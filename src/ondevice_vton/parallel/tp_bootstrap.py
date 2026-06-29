"""Distributed bootstrap for 2-Pi tensor parallelism (TP).

Both Pis run the SAME inference script in SPMD: each loads the full pipeline and
shards ONLY the main denoiser's attention (head-parallel, see tp_attention.py).
Every other module (vae, text/image encoders, garment net) runs replicated and
identically on both ranks, so every tensor outside the sharded attention stays
bit-identical across ranks. Inside attention each rank computes its own head
slice and a single all_reduce per attention sums the partial outputs. Net result
is identical to single-Pi, with the per-step attention compute split across two
Pis.

Launch (rank 0 = this Pi 192.168.100.1, rank 1 = peer 192.168.100.2):
    GLOO_SOCKET_IFNAME=eth0 python inference_lowmem.py --tp --rank 0 --world_size 2 ...
    GLOO_SOCKET_IFNAME=eth0 python inference_lowmem.py --tp --rank 1 --world_size 2 ...

Both must agree on --master_addr / --master_port (rank 0's eth0 addr).
"""
import os

import torch.distributed as dist


def init_tp(rank, world_size, master_addr="192.168.100.1", master_port=29500,
            iface="eth0", backend="gloo"):
    """Bring up the gloo process group over the direct wired link.

    Pins gloo to `iface` (the direct GbE link, not wlan0) so the all_reduces ride
    the fast point-to-point cable. Returns the (rank, world_size) the group agreed
    on. Blocks until every rank has joined.
    """
    if iface:
        os.environ.setdefault("GLOO_SOCKET_IFNAME", iface)
    os.environ["MASTER_ADDR"] = str(master_addr)
    os.environ["MASTER_PORT"] = str(master_port)
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    return dist.get_rank(), dist.get_world_size()


def shutdown_tp():
    """Tear the process group down (no-op if it was never initialized)."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
