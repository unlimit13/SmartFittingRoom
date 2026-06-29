"""Compare two try-on output images for TP-vs-single equivalence.

    .venv/bin/python parallel/compare_outputs.py <a.png> <b.png>

Prints max/mean absolute per-pixel difference (0-255 scale). TP reorders the
attention reduction (per-rank partial sums + all_reduce vs one full matmul), so
bit-identical is not expected; a handful of +/-1 LSB pixels is a PASS.
"""
import sys

import numpy as np
from PIL import Image


def main(a_path, b_path):
    a = np.asarray(Image.open(a_path).convert("RGB"), dtype=np.int16)
    b = np.asarray(Image.open(b_path).convert("RGB"), dtype=np.int16)
    if a.shape != b.shape:
        print(f"SHAPE MISMATCH {a.shape} vs {b.shape}")
        return 1
    d = np.abs(a - b)
    maxd, meand = int(d.max()), float(d.mean())
    frac_gt1 = float((d > 1).mean())
    print(f"shape={a.shape} max|diff|={maxd} mean|diff|={meand:.4f} "
          f"frac(|diff|>1)={frac_gt1:.6f}")
    # TP reorders float accumulation, so a few LSB-level pixel diffs are expected
    # and benign; a real logic bug shows up as large diffs over many pixels.
    ok = maxd <= 5 and meand < 0.05 and frac_gt1 < 0.005
    print("EQUIVALENT (within LSB float-reordering noise)" if ok
          else "DIFFERS -- investigate")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
