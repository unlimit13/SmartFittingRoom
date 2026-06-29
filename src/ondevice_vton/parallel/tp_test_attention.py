"""Verify TP attention sharding math on ONE machine (2 gloo ranks via loopback).

Compares the full Attention output against the head-parallel TP output
(column-parallel q/k/v + row-parallel to_out + all_reduce). Same machine, so this
checks CORRECTNESS, not speed. Run:

    .venv/bin/python parallel/tp_test_attention.py
"""
import os
import sys
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # repo root

from Mobile_VTON.models.attention_processor import Attention  # noqa: E402
from parallel.tp_attention import shard_attention_, TPAttnProcessor  # noqa: E402


def _case(name, attn_kwargs, x_shape, enc_shape=None):
    torch.manual_seed(0)
    attn = Attention(**attn_kwargs).eval()
    x = torch.randn(*x_shape)
    enc = torch.randn(*enc_shape) if enc_shape else None
    with torch.no_grad():
        ref = attn(x, encoder_hidden_states=enc)
        tp = deepcopy(attn)
        shard_attention_(tp, dist.get_rank(), dist.get_world_size())
        tp.set_processor(TPAttnProcessor())
        out = tp(x, encoder_hidden_states=enc)
    if dist.get_rank() == 0:
        d = (out - ref).abs()
        ok = torch.allclose(out, ref, atol=1e-4)
        print(f"[{name}] max|diff|={d.max():.2e} mean|diff|={d.mean():.2e} "
              f"-> {'PASS' if ok else 'FAIL'}")
        return ok
    return True


def worker(rank, world):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29577")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    results = []
    # self-attention: 512 dim, 8 heads
    results.append(_case("self-attn",
                         dict(query_dim=512, heads=8, dim_head=64, bias=True),
                         x_shape=(2, 64, 512)))
    # cross-attention: query 512, context 768
    results.append(_case("cross-attn",
                         dict(query_dim=512, cross_attention_dim=768,
                              heads=8, dim_head=64, bias=True),
                         x_shape=(2, 64, 512), enc_shape=(2, 77, 768)))
    if rank == 0:
        print("ALL PASS" if all(results) else "SOME FAILED")
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(worker, args=(2,), nprocs=2)
