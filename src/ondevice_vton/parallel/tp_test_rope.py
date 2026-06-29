"""Verify TP attention == full attention WITH rope on the real denoiser.

The first 2-Pi run crashed because the real denoiser feeds image_rotary_emb into
attention (the per-module test in tp_test_denoiser.py never passed rope). rope is
applied per-head on head_dim and broadcasts over heads, so sharding query heads
must not change the result. This loads the real denoiser on 2 localhost gloo
ranks and checks TP-with-rope == full-with-rope for self- and cross-attention.

    .venv/bin/python parallel/tp_test_rope.py
"""
import os
import sys
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from Mobile_VTON.models.unets.unet_2d_condition_tryon import (  # noqa: E402
    UNet2DConditionModel as Unet_Tryon,
)
from parallel.tp_attention import (  # noqa: E402
    iter_attention_modules, tp_supported, shard_attention_, TPAttnProcessor,
)

CKPT = os.path.join(ROOT, "checkpoint")


def worker(rank, world):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29581")
    dist.init_process_group("gloo", rank=rank, world_size=world)

    model = Unet_Tryon.from_pretrained(
        CKPT, subfolder="denoiser", torch_dtype=torch.float32).eval()
    mods = [(n, a) for n, a in iter_attention_modules(model) if tp_supported(a, world)]
    # one self-attention + one cross-attention module
    self_m = next((na for na in mods if not na[1].is_cross_attention), None)
    cross_m = next((na for na in mods if na[1].is_cross_attention), None)
    targets = [m for m in (self_m, cross_m) if m is not None]

    S = 32
    npass = nfail = 0
    for name, attn in targets:
        head_dim = attn.inner_dim // attn.heads
        torch.manual_seed(7)
        x = torch.randn(1, S, attn.to_q.in_features)
        enc = (torch.randn(1, S, attn.to_k.in_features)
               if attn.is_cross_attention else None)
        # random rope table [S, head_dim]; same on both ranks (S_rope == S path)
        cos = torch.randn(S, head_dim)
        sin = torch.randn(S, head_dim)
        rope = (cos, sin)
        with torch.no_grad():
            ref = attn(x, encoder_hidden_states=enc, image_rotary_emb=rope)
            tp = deepcopy(attn)
            shard_attention_(tp, rank, world)
            tp.set_processor(TPAttnProcessor())
            out = tp(x, encoder_hidden_states=enc, image_rotary_emb=rope)
        if rank == 0:
            d = (out - ref).abs().max().item()
            kind = "cross" if attn.is_cross_attention else "self"
            ok = torch.allclose(out, ref, atol=1e-3)
            print(f"  [{kind}] {name}: max|diff|={d:.2e} -> {'PASS' if ok else 'FAIL'}")
            npass += ok
            nfail += (not ok)

    if rank == 0:
        print(f"[result] rope TP==full: pass={npass} fail={nfail} -> "
              f"{'ALL PASS' if nfail == 0 else 'SOME FAILED'}")
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(worker, args=(2,), nprocs=2)
