# VAE Decode — Spatial Split Feasibility + Implementation

## TL;DR
**Yes, the VAE decode splits spatially — and trivially so.** It is the cleanest
parallel target in the whole pipeline: the decoder already runs in independent,
feather-blended **tiles** (`TiledVaeDecoder`), so we just hand each rank a slice of
the tile list and sum the results with one pair of `all_reduce`s. No halo, no
GroupNorm stat sync, no attention gather — none of the denoiser's complications.
Implemented in `inference_lowmem.py` (`TiledVaeDecoder`, gated by `--spatial`).

## Why it's the easy case
The full-res decode (latent 128×96 → image 1024×768) is the pipeline's second big
fixed cost (~2.5–3 min, NOT shrunk by denoiser parallelism or step-count cuts).
`TiledVaeDecoder` already decodes it as a grid of overlapping tiles (tile=32,
overlap=8 in latent space → **20 tiles** for 128×96), each decoded independently:

```
for (hi, wi) in tiles:            # 20 independent tiles
    img = decoder(latent[hi:hi+32, wi:wi+32])   # heavy conv decode of one tile
    out[oh:oh+th, ow:ow+tw] += img * feather_mask
    acc[oh:oh+th, ow:ow+tw] += feather_mask
return out / acc                  # weight-normalized blend
```

The tiles share NOTHING during decode — the only coupling is the final additive
feather-blend into `out`/`acc`. That makes the decode **embarrassingly parallel**
over tiles: partition the tile list across ranks, accumulate locally, then sum the
two buffers once.

## Implementation (done)
`TiledVaeDecoder.__init__(... sp_rank, sp_world, sp_group)` and in `__call__`:
- enumerate the deterministic tile list (same on every rank),
- **each rank decodes only `tile_index % sp_world == sp_rank`** (round-robin), into
  its own zero-initialised `out`/`acc`,
- a rank that drew no tiles still allocates zero buffers (must join the collectives),
- `all_reduce(out, SUM)` + `all_reduce(acc, SUM)` over the group, then `out/acc`.

Wired in `inference_lowmem.py`: when `--spatial`, the decoder is built with
`sp_rank=tp_rank, sp_world=tp_world`. Non-spatial path is unchanged (`sp_world=1`).

## Cost / payoff
- **Compute**: N ranks → ~N× on the decode (20 tiles / N per rank). 2 Pis ≈ 2×,
  4 Pis ≈ 4× (20 tiles split 5/5/5/5). Near-perfect; tiles are equal-cost.
- **Comm**: exactly **two all_reduces** of the full image. out [B,3,1024,768] fp32
  ≈ 9.4 MB, acc [1,1,1024,768] ≈ 3.1 MB → ~12.6 MB total, once. On the direct GbE
  link ~0.1 s — negligible vs the minutes of decode.
- **Memory**: unchanged peak (still one tile in flight at a time per rank); each
  rank just does fewer tiles. No extra full-res buffers beyond the existing out/acc.
- **Correctness**: identical to the single-process tiled decode up to float
  reduction order (the blend is associative SUM). Verify with `compare_outputs.py`.

## Equivalence is exact-ish
Same tiles, same masks, same `out/acc` math — only the *order* of the additive
accumulation changes (per-rank partial sums + all_reduce vs one sequential loop),
so expect ±1 LSB float-reorder noise, like every other stage.

## Why this matters
After denoiser spatial-TP (1.26× on the step loop) the VAE decode became a larger
*share* of the wall clock (it's a fixed ~2.5 min). Splitting it removes that
serialization too, so the per-image latency win compounds:
- denoiser step loop: split by spatial-TP (1.26× over channel-TP)
- VAE decode: split by tile distribution (~N×)
- only the model load + garment-net single pass + scheduler remain unsplit.

## Limits / notes
- Tile COUNT caps the parallelism: 20 tiles → useful up to ~20 ranks; beyond the
  tile count, extra ranks idle. For 1024×768 with tile=32/ov=8 that's plenty.
- Load balance is exact only when tiles divide evenly by N (20/4=5 ✓; 20/3 = 7/7/6).
- The decoder weights stay replicated on every rank (same as the rest of spatial).
