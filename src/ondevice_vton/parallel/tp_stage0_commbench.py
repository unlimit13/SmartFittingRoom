#!/usr/bin/env python3
"""Stage 0 PoC: measure torch.distributed (gloo) comm over the direct wired link.

This mirrors exactly what 2-way tensor parallelism (TP) of the UNet will need:
  - small-tensor latency  -> per-op sync overhead (many tiny syncs add up)
  - bulk all_reduce       -> Megatron-style MLP/attention reduction
  - bulk all_gather       -> channel-split conv block boundaries
at UNet activation sizes, then prints a TP feasibility verdict.

Run on BOTH Pis. Rank 0's eth0 IP is the master address. gloo is pinned to the
wired NIC via GLOO_SOCKET_IFNAME so the heavy traffic never leaks onto wlan0.

Example (rank0 eth0 = 192.168.100.1, rank1 eth0 = 192.168.100.2):
  # on rank 0:
  .venv/bin/python parallel/tp_stage0_commbench.py --rank 0 --master-addr 192.168.100.1
  # on rank 1:
  .venv/bin/python parallel/tp_stage0_commbench.py --rank 1 --master-addr 192.168.100.1

Both processes block in init until the peer joins (60s timeout); start them
within a minute of each other (order does not matter).
"""
import argparse
import datetime
import os
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True, help="0 or 1")
    ap.add_argument("--world-size", type=int, default=2)
    ap.add_argument("--master-addr", required=True, help="rank 0's eth0 IP")
    ap.add_argument("--master-port", default="29500")
    ap.add_argument("--iface", default="eth0", help="wired NIC for gloo")
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()

    # Force gloo onto the wired link (never wlan0).
    os.environ["GLOO_SOCKET_IFNAME"] = args.iface
    os.environ["TP_SOCKET_IFNAME"] = args.iface
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = str(args.master_port)
    os.environ["RANK"] = str(args.rank)
    os.environ["WORLD_SIZE"] = str(args.world_size)

    import torch
    import torch.distributed as dist

    torch.set_num_threads(4)
    print(f"[rank {args.rank}] connecting via {args.iface} to "
          f"{args.master_addr}:{args.master_port} ...", flush=True)
    dist.init_process_group(backend="gloo",
                            timeout=datetime.timedelta(seconds=60))
    rank, world = dist.get_rank(), dist.get_world_size()
    is0 = rank == 0
    if is0:
        print(f"[init OK] world={world} iface={args.iface}", flush=True)

    def bench(fn, warmup, iters):
        for _ in range(warmup):
            fn()
        dist.barrier()
        t = time.time()
        for _ in range(iters):
            fn()
        dist.barrier()
        return (time.time() - t) / iters

    # ---- latency: tiny all_reduce RTT ----
    x = torch.zeros(1)
    lat = bench(lambda: dist.all_reduce(x), warmup=10, iters=200)
    if is0:
        print(f"\n[latency] tiny all_reduce: {lat*1e3:.3f} ms/op "
              f"({lat*1e6:.0f} us)  <- per-sync overhead")

    # ---- bandwidth: all_reduce & all_gather at activation sizes ----
    if is0:
        print(f"\n{'op':10s}{'size':>8s}{'time':>11s}{'thrpt MB/s':>13s}")
    eff_ag16 = None
    for mb in [1, 4, 16, 64]:
        n = mb * 1024 * 1024 // 4  # fp32 elements
        t = torch.ones(n)
        dt = bench(lambda: dist.all_reduce(t), warmup=3, iters=args.iters)
        if is0:
            print(f"{'allreduce':10s}{mb:5d}MB{dt*1e3:9.1f}ms{mb/dt:13.1f}")
        chunk = torch.ones(n)
        outs = [torch.empty(n) for _ in range(world)]
        dt2 = bench(lambda: dist.all_gather(outs, chunk), warmup=3, iters=args.iters)
        if is0:
            print(f"{'allgather':10s}{mb:5d}MB{dt2*1e3:9.1f}ms{mb*world/dt2:13.1f}")
        if mb == 16:
            eff_ag16 = 16 / dt2  # MB/s effective (total/wall)

    # ---- TP feasibility verdict (rough planning estimate) ----
    if is0 and eff_ag16:
        EST_COMM_MB = 350.0   # rough per-forward all-gather volume, 2-way TP @1024x768
        COMPUTE_TP_S = 60.0   # ~half of the ~119s single-Pi per-step main denoiser
        BASE_S = 119.0
        comm_s = EST_COMM_MB / eff_ag16
        tp_step = COMPUTE_TP_S + comm_s
        speedup = BASE_S / tp_step
        print("\n[verdict] 2-way TP per-step estimate (rough):")
        print(f"  all_gather effective @16MB : {eff_ag16:.0f} MB/s")
        print(f"  est comm/forward (~{EST_COMM_MB:.0f}MB) : {comm_s:.1f} s")
        print(f"  est compute (TP, ~half)    : {COMPUTE_TP_S:.0f} s")
        print(f"  => TP step ~{tp_step:.0f}s vs single-Pi {BASE_S:.0f}s  -> {speedup:.2f}x")
        if speedup >= 1.4:
            print("  PASS: comm budget supports a real latency win -> proceed to Stage 1.")
        else:
            print("  MARGINAL: tune (fp16 comm / compute-comm overlap) or reconsider TP.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
