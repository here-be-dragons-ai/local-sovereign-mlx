#!/usr/bin/env python3
"""Measure what the 3-bit quantization costs Kolibri 1, against the FP8 original.

measure-quality.py with the `kolibri` profile; see there for the method. Three
arms: the FP8 release (73 GB, dequantized lazily layer by layer), the shipped
3/6-bit build and a uniform 3-bit control (convert-kolibri.py --other-bits 3).

    ./measure-kolibri-quality.py prepare                 # texts + benchmarks
    ./measure-kolibri-quality.py check   --ckpt 3bit     # streamed == in-memory?
    ./measure-kolibri-quality.py forward --ckpt fp8      # all sets, resumable
    ./measure-kolibri-quality.py forward --ckpt 3bit
    ./measure-kolibri-quality.py forward --ckpt 3bit-uniform
    ./measure-kolibri-quality.py report                  # KL, PPL, accuracy

Data and hidden states live in ~/src/mlx/kolibri-quality/. Results: issue #8,
docs/kolibri-quality/.
"""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "_mq", Path(__file__).resolve().parent / "measure-quality.py"
)
mq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mq)

if __name__ == "__main__":
    mq.main("kolibri", __doc__)
