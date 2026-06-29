# Per-Pi Setup Checklist (4-Pi spatial cluster)

What each Pi needs before a distributed run. Standard cluster (2026-06-26+):
`192.168.100.1` = rank0/MAIN, `.2` = rank1, `.3` = rank2, `.4` = rank3.

## On EVERY Pi (rank 0..3)

### 1. Code — clone the repo + checkout master
```bash
git clone https://github.com/unlimit13/Mobile_VTON-ondevice-optmization.git
cd Mobile_VTON-ondevice-optmization && git checkout master && git pull
```
The spatial + channel-TP code (`parallel/sp_*.py`, `parallel/tp_*.py`, the
`--spatial`/`--tp` flags in `inference_lowmem.py`, the launchers) lives on master.
A `git pull` after any code change on rank0 must be run on every peer too.

### 2. Python env + dependencies
```bash
bash setup_env.sh          # creates .venv and installs requirements.txt
# (torch 2.12.1+cpu etc.; ~the same on every Pi)
```

### 3. Model checkpoint (~3.5 GB, NOT in git)
```bash
.venv/bin/python download_ckpt.py     # populates ./checkpoint/
```
`checkpoint/` is gitignored, so clone does NOT bring it — download on each Pi (or
rsync `checkpoint/` from rank0). Verify: `du -sh checkpoint` ≈ 3.5G.

### 4. Test data (`single_data/`, gitignored)
The runs read `--data_dir single_data` (test/image/*, test/cloth/*, manifests).
Build it once from the tracked samples, or rsync from rank0:
```bash
# option A: generate from a sample pair (also writes the manifests)
.venv/bin/python run_single_lowmem.py --person samples/person2.png \
  --cloth samples/cloth3.jpg --desc "..." --steps 1   # Ctrl-C after it builds single_data
# option B: rsync from rank0
rsync -a 192.168.100.1:/path/to/repo/single_data/ ./single_data/
```
`samples/person2.png` + `samples/cloth3.jpg` ARE tracked (in git), so they arrive
with the clone. The RGBA→RGB fix is in `inference_lowmem.py` (handles person2.png).

## Network (the cluster, not per-Pi software)
- All four on `192.168.100.0/24`, rank i → `192.168.100.(i+1)`. Set on each Pi
  (NOT persistent across reboot): `sudo ip addr add 192.168.100.<k>/24 dev eth0`.
- 3+ Pis need a **GbE switch** (the 2-Pi setup used a direct cable = point-to-point).
- `GLOO_SOCKET_IFNAME=eth0` (the launcher exports it).
- rank0 (.1) needs **passwordless SSH** to .2/.3/.4: `ssh-copy-id 192.168.100.<k>`.

## Run (from rank0 / .1)
```bash
bash parallel/run_sp_multi.sh 6 output/spatial          # 4-Pi default (.2 .3 .4)
# smoke-test first: 1 step, then 6
bash parallel/run_sp_multi.sh 1 output/spatial_smoke
```
PEER_DIR (peer repo path) defaults to `/home/willtek/Mobile_VTON-ondevice-optmization`;
override with `PEER_DIR=...` if peers clone elsewhere (must be identical on all peers).

## Quick readiness check (run on rank0)
```bash
for ip in 192.168.100.2 192.168.100.3 192.168.100.4; do
  echo "== $ip =="
  ssh -n -o BatchMode=yes -o ConnectTimeout=4 $ip \
    'cd ~/Mobile_VTON-ondevice-optmization 2>/dev/null && \
     echo code=$(git rev-parse --short HEAD) \
          venv=$(test -x .venv/bin/python && echo OK || echo MISSING) \
          ckpt=$(du -sh checkpoint 2>/dev/null | cut -f1) \
          sp=$(test -f parallel/sp_common.py && echo OK || echo MISSING) \
          data=$(test -d single_data/test && echo OK || echo MISSING)' \
    || echo "  SSH FAIL (set up passwordless ssh / static IP)"
done
```
All four fields OK + matching `code=` hash on every Pi → ready to launch.

## Status snapshot (2026-06-26)
Code is committed + pushed to origin/master @ 994d193, so a fresh clone now brings
the full spatial + channel-TP code and the cloth3/person2 test samples.
- .1 rank0: ready (this Pi; has the code, venv, ckpt).
- .2 rank1: ready (clone + venv + 3.5G ckpt + passwordless ssh). Run `git pull` to
  pick up 994d193 before the next run.
- .3 rank2: pings, but **passwordless SSH not set up** + repo/ckpt not provisioned
  (do the EVERY-Pi steps above).
- .4 rank3: **down** (no eth0 IP / power / cable) + not provisioned.
