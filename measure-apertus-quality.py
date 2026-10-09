#!/usr/bin/env python3
"""Measure what the 8-bit quantization costs Apertus 1.5 8B, against the bf16 original.

measure-quality.py with the `apertus` profile; see there for the method, the
data and the scoring. Three arms: the bf16 release (swiss-ai/Apertus-v1.5-8B)
as the reference, the shipped 8-bit omni build and a round-to-nearest 4-bit
control. Only the language model is measured; the image and audio tokenizers
are float32 in every arm.

    ./measure-apertus-quality.py prepare
    ./measure-apertus-quality.py check   --ckpt 8bit
    ./measure-apertus-quality.py forward --ckpt bf16
    ./measure-apertus-quality.py forward --ckpt 8bit
    ./measure-apertus-quality.py forward --ckpt 4bit
    ./measure-apertus-quality.py report

Run it with the Apertus venv; only the apertus1p5 branch of mlx-vlm knows the
model:

    ~/src/mlx/.venv-apertus/bin/python ./measure-apertus-quality.py ...

The multiple-choice prompts go through the template with its default,
"Deliberation: disabled". The "pre" / "post" split of the text set is
Kolibri's knowledge cutoff (2026-06-18), not Apertus'.

Data and hidden states live in ~/src/mlx/apertus-quality/. Stop the Apertus
server first.
"""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "_mq", Path(__file__).resolve().parent / "measure-quality.py"
)
mq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mq)

if __name__ == "__main__":
    mq.main("apertus", __doc__)
