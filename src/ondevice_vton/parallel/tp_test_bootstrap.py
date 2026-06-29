"""Localhost smoke test for parallel/tp_bootstrap.py.

Spawns 2 gloo ranks over loopback, runs init_tp -> all_reduce -> shutdown_tp,
and checks the collective summed correctly (rank r contributes r+1, so the sum
across world ranks must be world*(world+1)/2). Proves the bootstrap wiring the
real inference uses (PG up, default-group all_reduce, clean teardown) works.

    .venv/bin/python parallel/tp_test_bootstrap.py
"""
import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parallel.tp_bootstrap import init_tp, shutdown_tp  # noqa: E402


def worker(rank, world):
    # loopback rendezvous; no iface pin (let gloo pick lo)
    r, w = init_tp(rank, world, master_addr="127.0.0.1", master_port=29611, iface="")
    t = torch.full((4,), float(r + 1))
    dist.all_reduce(t, op=dist.ReduceOp.SUM)  # default group == the one init_tp made
    expected = world * (world + 1) / 2
    ok = torch.allclose(t, torch.full((4,), expected))
    if r == 0:
        print(f"[bootstrap] world={w} all_reduce sum={t.tolist()} expected={expected} "
              f"-> {'PASS' if ok else 'FAIL'}")
    shutdown_tp()
    assert ok


if __name__ == "__main__":
    mp.spawn(worker, args=(2,), nprocs=2)
    print("bootstrap smoke test: OK")
