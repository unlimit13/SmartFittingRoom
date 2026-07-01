"""Row-band (height-axis) partition math for spatial parallelism.

Pure integer geometry, deliberately free of torch so the split logic can be
unit-tested (NFR-03) without the heavy vton venv. sp_common.py re-exports these.
"""
import os


def band_bounds(H, rank, world):
    """Contiguous row band [h0, h1) this rank owns. Splits H as evenly as possible;
    earlier ranks get the +1 when H is not divisible (kept deterministic so every
    rank agrees on every other rank's band size).

    For world==2 an uneven split can be requested via SP_BAND_FRAC0 (rank 0's
    fraction of H, default 0.5) -- useful when one Pi has less free RAM. The split
    stays deterministic and contiguous, so halo/offset math is unaffected."""
    if world == 2:
        frac0 = float(os.environ.get("SP_BAND_FRAC0", "0.5"))
        split = max(1, min(H - 1, int(round(H * frac0))))
        return (0, split) if rank == 0 else (split, H)
    base, rem = divmod(H, world)
    h0 = rank * base + min(rank, rem)
    h1 = h0 + base + (1 if rank < rem else 0)
    return h0, h1


def band_size(H, rank, world):
    h0, h1 = band_bounds(H, rank, world)
    return h1 - h0
