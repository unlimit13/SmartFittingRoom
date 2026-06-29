#!/usr/bin/env bash
# Launch one N-Pi SPATIAL (H-band) parallel inference run. rank 0 runs here (the
# writer + gloo master at 192.168.100.1); ranks 1..N-1 run on the peer IPs given,
# each over an SSH connection kept open locally until that rank exits.
#
#   bash parallel/run_sp_multi.sh <steps> <outdir> <data> "<peer1_ip> <peer2_ip> ..."
#
# Examples:
#   2 Pis: bash parallel/run_sp_multi.sh 6 output/spatial single_data "192.168.100.2"
#   4 Pis: bash parallel/run_sp_multi.sh 6 output/spatial single_data \
#              "192.168.100.2 192.168.100.3 192.168.100.4"
#
# Requirements (see parallel/SPATIAL_4PI.md):
#  - every Pi on the 192.168.100.0/24 net (rank i -> .(i+1)); 3+ Pis need a SWITCH
#    (the 2-Pi direct cable is point-to-point). GLOO_SOCKET_IFNAME=eth0 on all.
#  - each peer has the repo at $PEER_DIR with .venv + checkpoint + the synced
#    parallel/sp_*.py + inference_lowmem.py + single_data.
set -euo pipefail
STEPS="${1:-1}"
OUT="${2:-output/spatial}"
DATA="${3:-single_data}"
# Default = 4-Pi standard: rank0 here (.1, main), ranks 1..3 on .2/.3/.4.
PEERS="${4:-192.168.100.2 192.168.100.3 192.168.100.4}"  # peer IPs for ranks 1..N-1

# All paths/interpreters are env-overridable so this same launcher works both from
# the original vton repo and vendored under SmartFittingRoom/src/ondevice_vton.
RANK0_DIR="${RANK0_DIR:-/home/willtek/bootcamp/SmartFittingRoom/src/ondevice_vton}"
PEER_DIR="${PEER_DIR:-/home/willtek/Mobile_VTON-ondevice-optmization}"
MASTER="${VTON_MASTER:-192.168.100.1}"
PORT="${VTON_PORT:-29500}"
PY="${VTON_PYTHON:-.venv/bin/python}"            # rank0 interpreter (rel to RANK0_DIR or absolute)
PEER_PY="${PEER_PYTHON:-.venv/bin/python}"        # peer interpreter (rel to PEER_DIR or absolute)
CKPT="${VTON_CHECKPOINT_PATH:-$RANK0_DIR/checkpoint}"
PEER_CKPT="${PEER_CHECKPOINT_PATH:-$PEER_DIR/checkpoint}"

read -ra PA <<< "$PEERS"
WORLD=$(( 1 + ${#PA[@]} ))

SHARED="--spatial --world_size ${WORLD} --master_addr ${MASTER} --master_port ${PORT} \
 --order unpaired --test_batch_size 1 --num_inference_steps ${STEPS} \
 --guidance_scale 2.5 --mixed_precision no"
ENV="CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 GLOO_SOCKET_IFNAME=eth0 \
 MALLOC_ARENA_MAX=2 SP_BAND_FRAC0=${SP_BAND_FRAC0:-0.5}"

cd "$RANK0_DIR"
mkdir -p logs "$OUT"
echo "[run_sp_multi] world=${WORLD} steps=${STEPS} peers=(${PEERS})"

PIDS=()
for i in "${!PA[@]}"; do
  r=$(( i + 1 )); ip="${PA[$i]}"
  ssh -n -o BatchMode=yes "$ip" "cd $PEER_DIR && mkdir -p logs $OUT && \
    ${ENV} ${PEER_PY} inference_lowmem.py --rank ${r} ${SHARED} \
    --checkpoint_path $PEER_CKPT --data_dir $PEER_DIR/$DATA \
    --output_dir $PEER_DIR/$OUT > logs/sp_rank${r}.log 2>&1" &
  PIDS+=($!)
  echo "[run_sp_multi] rank ${r} launched on ${ip}"
done

echo "[run_sp_multi] rank0 start: $(date '+%T')"
eval "${ENV} ${PY} inference_lowmem.py --rank 0 ${SHARED} \
  --checkpoint_path $CKPT --data_dir $RANK0_DIR/$DATA \
  --output_dir $OUT"
echo "[run_sp_multi] rank0 end: $(date '+%T')"

rc=0
for p in "${PIDS[@]}"; do wait "$p" || rc=$?; done
echo "[run_sp_multi] all peer ranks exited (rc=$rc)"
