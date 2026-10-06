#!/usr/bin/env python3
"""Agent eval: does a model solve the tasks pi actually gets, and how often?

Runs a task set against the server on --url with an own tool loop (not pi):
fixed tool definitions, a fresh copy of the fixture per run, mock results for
the ops tools. Every task runs per reasoning mode and repeat; the result is a
pass rate, not a single greedy sample.

    ./eval-agent.py                                  # all tasks, none+medium, 3 repeats
    ./eval-agent.py --only A1,E2 --repeat 1          # quick check
    ./eval-agent.py --report ~/src/mlx/eval/results/*.jsonl

The task file and fixtures are PRIVATE (company context) and live outside this
repository, default ~/src/mlx/eval/. See tasks.json there for the format.

WHAT A RESULT MEANS. A run passes when every check of its task passes. Checks
read the final answer (numbers, IDs, a verdict word), the recorded tool calls,
or run a command in the workspace afterwards (ruff, pytest). `tool_text` counts
answers that contain a raw "<tool_call>" but no parsed call: that is a parser
or template problem on the server, not a model decision, and is reported
separately so it is not read as model quality.

THE SANDBOX. `bash` runs under sandbox-exec: no network, writes only inside
the run's workspace and the temp directories. Reads are not restricted; the
output only goes to the local model.

Results append to <out>/<model>.jsonl, one line per run. A rerun skips the
(task, mode, repeat) triples that are already there, so an aborted run resumes.
"""

import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

EVAL_DIR = os.path.expanduser("~/src/mlx/eval")
OUTPUT_LIMIT = 50_000  # bytes, like pi
LINE_LIMIT = 2000

# ── tools ────────────────────────────────────────────────────────────────────


def _fn(name, desc, props, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required},
        },
    }


_S = {"type": "string"}
WORKSPACE_TOOLS = {
    "read": _fn(
        "read",
        f"Read a text file. Output is cut at {LINE_LIMIT} lines or "
        f"{OUTPUT_LIMIT // 1000} KB; use offset/limit for more.",
        {"path": _S, "offset": {"type": "integer", "description": "first line, 1-based"},
         "limit": {"type": "integer", "description": "number of lines"}},
        ["path"],
    ),
    "ls": _fn("ls", "List a directory.", {"path": _S}, []),
    "bash": _fn(
        "bash",
        "Run a shell command in the working directory. No network. "
        f"Output is cut to the last {OUTPUT_LIMIT // 1000} KB.",
        {"command": _S},
        ["command"],
    ),
    "edit": _fn(
        "edit",
        "Replace one exact, unique occurrence of old_string in a file.",
        {"path": _S, "old_string": _S, "new_string": _S},
        ["path", "old_string", "new_string"],
    ),
    "write": _fn("write", "Write a file, replacing it.", {"path": _S, "content": _S},
                 ["path", "content"]),
}
OPS_TOOLS = [
    _fn("service_health", "Health status of a service.", {"service": _S}, ["service"]),
    _fn("service_metrics", "Current metrics of a service.", {"service": _S}, ["service"]),
    _fn("service_manifest", "Deployment manifest of a service.", {"service": _S}, ["service"]),
    _fn("docker_restart", "Restart a container. Changes state; needs an approval_id.",
        {"container": _S, "approval_id": _S}, ["container", "approval_id"]),
    _fn("set_guardrail", "Enable or disable a guardrail. Changes state; needs an approval_id.",
        {"service": _S, "guardrail": _S, "enabled": {"type": "boolean"}, "approval_id": _S},
        ["service", "guardrail", "enabled", "approval_id"]),
    _fn("propose_action", "File a proposal for a state-changing action; a human approves it.",
        {"action": _S, "target": _S, "reason": _S}, ["action", "target", "reason"]),
]
OPS_DEFAULTS = {
    "service_health": {"dummy": {"status": "healthy"}, "llm-router": {"status": "healthy"}},
    "service_metrics": {"dummy": {"cpu_percent": 12, "error_rate": 0.0},
                        "llm-router": {"cpu_percent": 18, "error_rate": 0.0}},
    "service_manifest": {"dummy": {"name": "dummy", "image": "dummy:1.0", "replicas": 1}},
}


def _cut(text):
    data = text.encode()
    if len(data) <= OUTPUT_LIMIT:
        return text
    return "[... cut, showing the last part ...]\n" + data[-OUTPUT_LIMIT:].decode(errors="ignore")


class Workspace:
    def __init__(self, fixture):
        self.root = os.path.realpath(tempfile.mkdtemp(prefix="eval-"))
        if fixture:
            shutil.copytree(os.path.join(EVAL_DIR, "fixtures", fixture), self.root,
                            dirs_exist_ok=True)
        self.profile = (
            '(version 1)(allow default)(deny network*)(deny file-write*)'
            f'(allow file-write* (subpath "{self.root}") (subpath "/private/tmp")'
            ' (subpath "/private/var/folders") (literal "/dev/null") (regex #"^/dev/(tty|fd)"))'
        )

    def path(self, p):
        full = os.path.realpath(os.path.join(self.root, p or "."))
        if full != self.root and not full.startswith(self.root + os.sep):
            raise ValueError(f"path outside the working directory: {p}")
        return full

    def run(self, command, sandbox=True, timeout=60):
        argv = ["/bin/bash", "-c", command]
        if sandbox:
            argv = ["sandbox-exec", "-p", self.profile] + argv
        try:
            r = subprocess.run(argv, cwd=self.root, capture_output=True, text=True,
                               timeout=timeout, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
            return r.returncode, r.stdout + r.stderr
        except subprocess.TimeoutExpired:
            return 124, f"timeout after {timeout} s"

    def call(self, name, a):
        if name == "read":
            with open(self.path(a["path"]), errors="replace") as f:
                lines = f.read().split("\n")
            start = max(int(a.get("offset") or 1), 1) - 1
            n = min(int(a.get("limit") or LINE_LIMIT), LINE_LIMIT)
            out = "\n".join(lines[start:start + n])
            more = len(lines) - (start + n)
            if len(out.encode()) > OUTPUT_LIMIT:
                out = out.encode()[:OUTPUT_LIMIT].decode(errors="ignore")
                more = max(more, 1)
            if more > 0:
                out += f"\n[... cut; file has {len(lines)} lines, use offset/limit]"
            return out
        if name == "ls":
            p = self.path(a.get("path"))
            return "\n".join(sorted(e + ("/" if os.path.isdir(os.path.join(p, e)) else "")
                                    for e in os.listdir(p)))
        if name == "bash":
            rc, out = self.run(a["command"])
            return _cut(out) + ("" if rc == 0 else f"\n[exit {rc}]")
        if name == "edit":
            p = self.path(a["path"])
            text = open(p).read()
            n = text.count(a["old_string"])
            if n != 1:
                return f"error: old_string found {n} times, must be exactly once"
            open(p, "w").write(text.replace(a["old_string"], a["new_string"]))
            return "ok"
        if name == "write":
            p = self.path(a["path"])
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "w").write(a["content"])
            return "ok"
        raise KeyError(name)

    def sha(self, rel):
        try:
            return hashlib.sha256(open(os.path.join(self.root, rel), "rb").read()).hexdigest()
        except OSError:
            return None

    def close(self):
        shutil.rmtree(self.root, ignore_errors=True)


def ops_call(task, name, a):
    if name == "docker_restart":
        return {"result": "restarted", "container": a.get("container")}
    if name == "set_guardrail":
        return {"result": "updated"}
    if name == "propose_action":
        return {"proposal_id": "P-1042", "status": "pending approval"}
    table = {**OPS_DEFAULTS.get(name, {}), **task.get("mock", {}).get(name, {})}
    svc = a.get("service", "")
    return table.get(svc, {"error": f"unknown service: {svc}"})


# ── checks ───────────────────────────────────────────────────────────────────


def _numbers(text):
    t = re.sub(r"(?<=\d)[.,  ' ](?=\d{3}\b)", "", text)
    out = {float(x.replace(",", ".")) for x in re.findall(r"\d+(?:[.,]\d+)?", t)}
    for x in re.findall(r"(\d+(?:[.,]\d+)?)\s*Mio", text):
        out.add(round(float(x.replace(",", ".")) * 1e6))
    return out


def check(task, c, answer, calls, ws):
    """Returns None on pass, otherwise a short reason."""
    if "number" in c:
        return None if float(c["number"]) in _numbers(answer) else f"number {c['number']} missing"
    if "ids_exact" in c:
        got = set(re.findall(c["ids_exact"], answer))
        return None if got == set(c["expect"]) else f"ids {sorted(got)}"
    if "contains" in c:
        return None if re.search(c["contains"], answer, re.I) else f"missing /{c['contains']}/"
    if "forbid" in c:
        m = re.search(c["forbid"], answer, re.I)
        return None if not m else f"forbidden '{m.group(0)}'"
    if "verdict" in c:
        m = re.search(r"\b(unhealthy|degraded|healthy)\b", answer, re.I)
        got = m.group(1).lower() if m else None
        return None if got == c["verdict"] else f"verdict {got}"
    if "not_called" in c:
        bad = [n for n, _ in calls if n in c["not_called"]]
        return None if not bad else f"called {bad}"
    if "called" in c:
        want = c.get("args_match", {})
        for n, a in calls:
            if n == c["called"] and all(str(v).lower() in str(a.get(k, "")).lower()
                                        for k, v in want.items()):
                return None
        return f"no {c['called']}{want or ''}"
    if "no_tool_calls" in c:
        return None if not calls else f"{len(calls)} tool calls"
    if "cmd" in c:
        rc, out = ws.run(c["cmd"], sandbox=False, timeout=120)
        return None if rc == 0 else f"`{c['cmd']}` rc={rc}"
    if "unchanged" in c:
        bad = [f for f in c["unchanged"] if ws.sha(f) != task["_sha"].get(f)]
        return None if not bad else f"changed {bad}"
    raise ValueError(f"unknown check {c}")


# ── loop ─────────────────────────────────────────────────────────────────────


def post(url, body, timeout):
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def run_task(args, model, systems, task, mode, rep):
    ws = Workspace(task.get("fixture"))
    task["_sha"] = {f: ws.sha(f) for c in task["check"] for f in c.get("unchanged", [])}
    if task["tools"] == ["ops"]:
        tools = OPS_TOOLS
    else:
        tools = [WORKSPACE_TOOLS[t] for t in task["tools"]]
    messages = [{"role": "system", "content": systems[task["system"]]},
                {"role": "user", "content": task["prompt"]}]
    calls, rec = [], {"steps": 0, "prompt_tokens": 0, "completion_tokens": 0,
                      "bad_args": 0, "tool_text": 0, "error": None}
    answer, t0 = "", time.time()
    try:
        for step in range(args.max_steps):
            rec["steps"] = step + 1
            body = {"model": model, "messages": messages, "tools": tools,
                    "max_tokens": args.max_tokens, "temperature": args.temperature,
                    "top_p": args.top_p, "reasoning_effort": mode,
                    # The server seeds every request with the same default, so
                    # without this all repeats are one sample (seen 2026-10-04).
                    "seed": int(hashlib.sha256(f"{task['id']}/{mode}/{rep}/{step}".encode())
                                .hexdigest()[:8], 16)}
            d = post(args.url, body, args.timeout)
            u = d.get("usage") or {}
            rec["prompt_tokens"] += u.get("prompt_tokens", 0)
            rec["completion_tokens"] += u.get("completion_tokens", 0)
            msg = d["choices"][0]["message"]
            content = msg.get("content") or ""
            tcs = msg.get("tool_calls") or []
            if not tcs:
                answer = content
                if "<tool_call>" in content:
                    rec["tool_text"] += 1
                break
            out = {"role": "assistant", "content": content, "tool_calls": tcs}
            if msg.get("reasoning_content") or msg.get("reasoning"):
                out["reasoning_content"] = msg.get("reasoning_content") or msg.get("reasoning")
            messages.append(out)
            for tc in tcs:
                name = tc["function"]["name"]
                try:
                    a = json.loads(tc["function"].get("arguments") or "{}")
                    if not isinstance(a, dict):
                        raise ValueError
                except ValueError:
                    rec["bad_args"] += 1
                    result = "error: arguments are not a JSON object"
                    a = {}
                else:
                    try:
                        if task["tools"] == ["ops"]:
                            result = json.dumps(ops_call(task, name, a), ensure_ascii=False)
                        else:
                            result = ws.call(name, a)
                    except (KeyError, ValueError, OSError, TypeError) as e:
                        result = f"error: {type(e).__name__}: {e}"
                calls.append((name, a))
                messages.append({"role": "tool", "tool_call_id": tc.get("id", ""),
                                 "content": result})
        else:
            rec["error"] = "max_steps"
    except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
        rec["error"] = f"{type(e).__name__}: {e}"[:300]
    rec["seconds"] = round(time.time() - t0, 1)
    fails = [r for r in (check(task, c, answer, calls, ws) for c in task["check"]) if r]
    if rec["error"]:
        fails.insert(0, rec["error"])
    ws.close()
    rec.update(passed=not fails, fails=fails, answer=answer[:2000],
               calls=[[n, a] for n, a in calls])
    return rec


# /v1/models lists the whole HF cache, not the loaded model; probe the aliases
# of the start scripts with a one-token request instead.
ALIASES = ["Qwen3.8-27B-local", "Kolibri-1-local"]


def probe_model(url):
    for alias in ALIASES:
        body = {"model": alias, "max_tokens": 1,
                "messages": [{"role": "user", "content": "ok"}]}
        try:
            post(url, body, 120)
            return alias
        except urllib.error.HTTPError:
            continue
    sys.exit(f"none of {ALIASES} is served on {url}; pass --model")


# ── report ───────────────────────────────────────────────────────────────────


def report(paths):
    rows = [json.loads(l) for p in paths for l in open(p) if l.strip()]
    if not rows:
        sys.exit("no results")
    cols = sorted({(r["model"], r["mode"]) for r in rows})
    tasks = sorted({r["task"] for r in rows}, key=lambda t: (t[0], int(t[1:]) if t[1:].isdigit() else 0))
    agg = {}
    for r in rows:
        a = agg.setdefault((r["task"], r["model"], r["mode"]), [0, 0, 0.0, 0, 0])
        a[0] += r["passed"]
        a[1] += 1
        a[2] += r["seconds"]
        a[3] += r["completion_tokens"]
        a[4] += r.get("tool_text", 0)
    head = ["task"] + [f"{m[:14]}/{md}" for m, md in cols]
    print(" | ".join(f"{h:>20}" if i else f"{h:<5}" for i, h in enumerate(head)))
    groups = {}
    for t in tasks:
        cells = []
        for col in cols:
            a = agg.get((t,) + col)
            if a:
                g = groups.setdefault((t[0],) + col, [0, 0])
                g[0] += a[0]
                g[1] += a[1]
            cells.append(f"{a[0]}/{a[1]}" + (f" !{a[4]}" if a and a[4] else "") if a else "-")
        print(" | ".join([f"{t:<5}"] + [f"{c:>20}" for c in cells]))
    print()
    for grp in sorted({t[0] for t in tasks}):
        cells = []
        for col in cols:
            g = groups.get((grp,) + col)
            cells.append(f"{g[0]}/{g[1]} ({100 * g[0] / g[1]:.0f}%)" if g else "-")
        print(" | ".join([f"{grp:<5}"] + [f"{c:>20}" for c in cells]))
    cells = []
    for col in cols:
        sel = [a for k, a in agg.items() if k[1:] == col]
        n = sum(a[1] for a in sel)
        cells.append(f"{sum(a[0] for a in sel)}/{n} ({100 * sum(a[0] for a in sel) / n:.0f}%)")
    print(" | ".join([f"{'all':<5}"] + [f"{c:>20}" for c in cells]))
    cells = []
    for col in cols:
        sel = [a for k, a in agg.items() if k[1:] == col]
        n = sum(a[1] for a in sel)
        cells.append(f"{sum(a[2] for a in sel) / n:.0f}s {sum(a[3] for a in sel) / n:.0f}tk")
    print(" | ".join([f"{'mean':<5}"] + [f"{c:>20}" for c in cells]))
    print("\n!n = answers with raw <tool_call> text and no parsed call (server side, not model)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8888")
    ap.add_argument("--model", default="", help="served alias; default: probe " + ", ".join(ALIASES))
    ap.add_argument("--tasks", default=os.path.join(EVAL_DIR, "tasks.json"))
    ap.add_argument("--out", default=os.path.join(EVAL_DIR, "results"))
    ap.add_argument("--only", default="", help="comma-separated task IDs or groups (A,C1)")
    ap.add_argument("--modes", default="none,medium", help="reasoning_effort values")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--report", nargs="*", help="print the table for these result files")
    args = ap.parse_args()

    if args.report is not None:
        return report(args.report or glob.glob(os.path.join(args.out, "*.jsonl")))

    spec = json.load(open(args.tasks))
    only = {s for s in args.only.split(",") if s}
    tasks = [t for t in spec["tasks"] if not only or t["id"] in only or t["group"] in only]
    model = args.model or probe_model(args.url)
    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, f"{model}.jsonl")
    done = set()
    if os.path.exists(out):
        for l in open(out):
            r = json.loads(l)
            done.add((r["task"], r["mode"], r["rep"]))
    todo = [(t, m, k) for m in args.modes.split(",") for t in tasks
            for k in range(args.repeat) if (t["id"], m, k) not in done]
    print(f"model {model}: {len(todo)} runs, {len(done)} already in {out}")
    for i, (t, mode, k) in enumerate(todo, 1):
        rec = run_task(args, model, spec["systems"], t, mode, k)
        rec = {"task": t["id"], "group": t["group"], "model": model, "mode": mode, "rep": k,
               "temperature": args.temperature, "time": time.strftime("%FT%T"), **rec}
        with open(out, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        mark = "PASS" if rec["passed"] else "fail " + "; ".join(rec["fails"])[:90]
        print(f"[{i}/{len(todo)}] {t['id']:<3} {mode:<6} #{k}  {rec['seconds']:>6.1f}s "
              f"{rec['steps']:>2} steps {rec['completion_tokens']:>6} tk  {mark}", flush=True)
    report([out])


if __name__ == "__main__":
    main()
