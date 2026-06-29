# Spatial Parallelism on 4 Pis

The spatial (H-band) parallel code is **world-agnostic** — the same `parallel/sp_*.py`
runs on any rank count. This doc covers what changed for N>2 and how to run on 4 Pis.

## What's already N-ready (no change needed)
The core math was written against `world` from the start and is verified at world=4
on localhost (`sp_test_conv.py 4` -> 14/14 PASS, max|diff| 1.9e-6):
- `band_bounds(H, rank, world)` — splits H into N contiguous bands (even split for
  N>2; the optional uneven `SP_BAND_FRAC0` knob applies only to N==2).
- `exchange_halo` — a 1-D neighbour chain: rank r swaps a halo row with r-1 and r+1.
  Point-to-point isend/irecv keyed by (peer, tag), so N pairs never collide. Interior
  ranks (1..N-2) exchange on both sides; ends zero-pad the outer edge.
- `gather_rows` / `_gather_seq` — all_gather over all N ranks (lean int size gather +
  per-rank broadcast), variable band sizes handled.
- `sp_groupnorm` — per-group (sum, sqsum, count) all_reduce over all N ranks.
- `sp_attention` — query band offset = `band_bounds(fullH, rank, N)[0]*W`; self-attn
  person K/V all_gathered over all N ranks; cross-attn query-parallel. No all_reduce.
- VAE decode — tiles round-robin over N ranks + 2 all_reduces (see VAE_SPATIAL_SPLIT.md).

## Launcher (new)
`parallel/run_sp_multi.sh` generalises `run_sp_pair.sh` to any peer count:
```
# 2 Pis:
bash parallel/run_sp_multi.sh 6 output/spatial single_data "192.168.100.2"
# 4 Pis:
bash parallel/run_sp_multi.sh 6 output/spatial single_data \
     "192.168.100.2 192.168.100.3 192.168.100.4"
```
rank 0 runs here (writer + gloo master 192.168.100.1); rank i runs on peer i over an
SSH channel held open locally until it exits. `world = 1 + #peers`.

## Network setup (the real 4-Pi requirement)
- The 2-Pi link is a single **direct cable** (point-to-point). **3+ Pis need a GbE
  switch.** Put every Pi on `192.168.100.0/24`: rank i -> `192.168.100.(i+1)`
  (rank0 .1, rank1 .2, rank2 .3, rank3 .4). Static IPs are NOT persistent across
  reboot — set on each Pi: `sudo ip addr add 192.168.100.<k>/24 dev eth0`.
- `GLOO_SOCKET_IFNAME=eth0` on every Pi (the launcher sets it).
- rank0 must have **passwordless SSH** to every peer IP.
- Each peer needs the repo at `$PEER_DIR` (default
  `/home/willtek/Mobile_VTON-ondevice-optmization`) with `.venv` + `checkpoint` +
  the synced `parallel/sp_*.py`, `inference_lowmem.py`, and `single_data`. Override
  the path with `PEER_DIR=...` (must be the same on all peers).

## Comm changes once a switch is in the path
On the 2-Pi direct cable comm was ~0. Through a switch the per-collective latency and
contention become real. Spatial's collectives per step: ~34 halo swaps (tiny, 1 row)
+ ~46 GroupNorm stat all_reduces (tiny) + per-self-attn person K/V all_gathers (MQA →
small) + the 2 VAE all_reduces. All are small payloads, but the COUNT × switch latency
(RTT through a switch ~0.1–0.3 ms) adds maybe tens of ms/step — still small vs ~68 s
of compute, but no longer exactly zero. Bandwidth is not the bottleneck; latency of
the many small syncs is. Batching adjacent halos / overlapping the K/V gather with
compute (TODO) matters more at N=4 than at N=2.

## Memory at N=4
Spatial keeps the MAIN denoiser weights **full** on every rank (it splits activations,
not weights). Activations shrink to 1/4, but the model-load peak is weight-dominated
and unchanged. The `--spatial` branch already channel-TP-shards the GARMENT denoiser
to claw back its weights at the load peak; note at N=4 some garment stages don't shard
(e.g. 14 query heads not divisible by 4) so they stay full — slightly higher peak than
N=2. If a Pi OOMs at load, options: (a) keep garment channel-TP (on by default),
(b) `MALLOC_ARENA_MAX=2` (launcher sets it), (c) free other RAM on that Pi. The band
being 1/4 helps the per-step (post-load) memory.

## Expected scaling (estimate, not yet measured on real 4 Pis)
- denoiser step: spatial splits ~all of the step, so ~N× minus the small
  unsplittable remainder and switch-latency overhead. 2 Pis measured 1.26× over
  channel-TP (= the channel-TP step 85s -> 68s). 4 Pis ideal ~2× over channel-TP,
  realistic less once switch latency on the many small collectives bites.
- VAE decode: ~N× (20 tiles / N).
- Unsplit: model load + garment single pass + scheduler — fixed.
Only validated on real hardware up to 2 Pis (we have 2). The world=4 path is
math-verified on localhost; running it for real needs the 4-Pi + switch setup above.
