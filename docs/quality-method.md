# How we measure the quality of a quantized build

Shared method for [`kolibri-quality/`](kolibri-quality/README.md) and
[`apertus-quality/`](apertus-quality/README.md). Instrument:
[`measure-quality.py`](../measure-quality.py), with one profile per model. The
commit, mlx and mlx-vlm versions of each run are in the `versions` field of
the result JSON.

## Arms

- **Reference**: the original release (Kolibri: block-FP8, dequantized to
  bf16; Apertus: bf16).
- **Noise floor**: the reference again, but each sequence runs through a layer
  in 512-token pieces against a KV cache, the way a server prefills, instead
  of in one pass. The weights and the math are the same; only the order of
  floating-point operations changes. Whatever this arm differs from the
  reference is numerical noise, and a build within it cannot be told apart
  from the original. The arm is called `<reference>-chunked`.
- **The shipped build**, and a **control** (Kolibri: uniform 3-bit; Apertus:
  round-to-nearest 4-bit, group size 64).

All arms go through the same layer-streamed forward pass: one decoder layer
is loaded, run over all sequences, and dropped. For the single-pass arms this
is bit-identical to the ordinary in-memory forward (`check`). Logits are
computed in fp32 on the CPU from the final hidden state and each arm's own LM
head; on the GPU an fp32 matmul deviates by ~1e-2, the size of what is being
measured.

## Distribution comparison (primary)

KL(reference ‖ arm) per token and whether the most likely next token is the
same, on every position of each text. This is more sensitive than accuracy
and needs less data. Reported as `llama-perplexity --kl-divergence` does:
mean, median, p90, p99, p99.9 (the worst token in a thousand), max, "same
top"; percentiles by nearest rank. Perplexity of both arms alongside.

| set | content | tokens (Apertus / Kolibri) |
|---|---|---|
| `text` | German and English Wikipedia prose at pinned revisions, 4096-token windows | 82k / 76k |
| `calib-v5` | Calibration v5 (bartowski / tristandruyen), mixed text, code and noise | 114k / 120k |
| `flores-eu` | the first 20 FLORES passages per EU language, via Belebele; 23 languages, Irish is not in Belebele | 70k / 91k |
| `chat` | OpenAssistant oasst2 threads (de, en), best-ranked path, chained into 16k-token dialogues through the chat template | 65k / 65k |
| `tools` | hermes-function-calling-v1 multi-turn conversations with tool calls and responses, tools passed to the chat template | 49k / 59k |

`chat` and `tools` are also reported for the assistant turns alone: the
tokens of each assistant message's content, found in the rendered text; turns
that are only tool calls use the template's turn boundaries. Data revisions
and SHA-256 hashes are in `sources` per set in the result JSON.

## Multiple choice

- **Sets**: Belebele `deu_Latn` / `eng_Latn` (900 each), Global-MMLU-Lite
  `de` / `en` (400 each, split into culturally sensitive and agnostic), and
  Belebele in four less-represented EU languages: Maltese, Latvian, Estonian,
  Lithuanian (900 each). For those four, passage, question and options are in
  the language and the instruction is the English one, so no translation of
  ours enters the prompt.
- **Position bias**: the four German and English sets are run a second time
  with the options shifted by one position (A→B, B→C, C→D, D→A, answer
  remapped). An answer that follows the content picks the same option in both
  orders; the report gives that share, the accuracy in both orders and the
  distribution of the predicted letters.
- **Prompt**: zero-shot, through the model's chat template as a single user
  turn, with the options as `A) … D)`:

  ```
  [Text:\n<passage>\n\n]Frage: <question>\n\nA) …\nB) …\nC) …\nD) …\n\nAntworte nur mit dem Buchstaben der richtigen Antwort.
  [Passage:\n<passage>\n\n]Question: <question>\n\nA) …\nB) …\nC) …\nD) …\n\nAnswer with the letter of the correct answer only.
  ```

  Template settings: Kolibri `reasoning_effort=none`; Apertus the template
  default (deliberation disabled).
- **Answer**: the letter among A-D with the highest logit at the first
  answer position (log-likelihood, nothing is generated or parsed). "Letter
  top-1" is how often the overall most likely token is one of the four
  letters.
- **Against the reference, per question**:
  - flips, split into right → wrong and wrong → right; symmetric flips are
    noise, an imbalance is a systematic loss;
  - the difference in accuracy with a paired 95% interval (Newcombe 1998,
    method 10, checked against the example in the paper);
  - McNemar's exact test (no difference detectable or not);
  - **equivalence** by TOST with a margin of ±1 percentage point, fixed
    before the runs: the 90% interval of the difference lies inside ±1 pp.
- **Wording**: "equivalent within ±1 pp" only when TOST passes; otherwise "no
  difference detectable" (or the measured difference). A p-value of 1.00 does
  not mean "equally good".

These are likelihood scores without reasoning: they show what the
quantization changes and are not comparable with published benchmark scores.

## Not covered yet

Long generation (reasoning chains, long context, instruction following,
code), baselines beyond our own controls, Irish (not in Belebele); see
here-be-dragons-ai/sovereign-models#8.
