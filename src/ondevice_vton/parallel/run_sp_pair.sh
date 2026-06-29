#!/usr/bin/env bash
# Launch one 2-Pi tensor-parallel inference run: rank 1 on the peer (background
# over SSH) + rank 0 here (foreground, the writer). gloo init_process_group
# blocks until both ranks join, so launch order doesn't matter.
#
#   bash parallel/run_sp_pair.sh <steps> <outdir>
set -euo pipefail
STEPS="${1:-1}"
OUT="${2:-output/tp}"
DATA="${3:-single_data}"   # data dir name, relative to each repo root

PEER=192.168.100.2
PEER_DIR=/home/willtek/Mobile_VTON-ondevice-optmization
RANK0_DIR=/home/willtek/bootcamp/vton/2026_CVPR_Mobile-VTON

# get_real_path resolves relative paths against WORK_DIR (one level above the
# repo), so checkpoint/data MUST be absolute -- and they differ per Pi.
SHARED="--spatial --world_size 2 --master_addr 192.168.100.1 --master_port 29500 \
 --order unpaired --test_batch_size 1 --num_inference_steps ${STEPS} \
 --guidance_scale 2.5 --mixed_precision no"
ENV="CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 GLOO_SOCKET_IFNAME=eth0 MALLOC_ARENA_MAX=2 SP_BAND_FRAC0=${SP_BAND_FRAC0:-0.4}"

# rank 1 on the peer: keep the ssh connection OPEN for the whole rank-1 run
# (background the ssh LOCALLY, don't background on the remote -- remote
# backgrounding leaves the ssh channel open and deadlocks the launcher). rank 1
# logs to the peer's own file; ssh returns when rank 1 exits.
cd "$RANK0_DIR"
mkdir -p logs "$OUT"
ssh -n -o BatchMode=yes "$PEER" "cd $PEER_DIR && mkdir -p logs $OUT && \
  ${ENV} .venv/bin/python inference_lowmem.py --rank 1 ${SHARED} \
  --checkpoint_path $PEER_DIR/checkpoint --data_dir $PEER_DIR/$DATA \
  --output_dir $PEER_DIR/$OUT > logs/tp_rank1.log 2>&1" &
SSH_PID=$!
echo "[run_sp_pair] rank1 launched on $PEER (ssh pid $SSH_PID, steps=$STEPS)"

# rank 0 here (foreground, writer) -- runs concurrently with rank 1
echo "[run_sp_pair] rank0 start: $(date '+%T')"
eval "${ENV} .venv/bin/python inference_lowmem.py --rank 0 ${SHARED} \
  --checkpoint_path $RANK0_DIR/checkpoint --data_dir $RANK0_DIR/$DATA \
  --output_dir $OUT"
echo "[run_sp_pair] rank0 end: $(date '+%T')"
wait "$SSH_PID"; echo "[run_sp_pair] rank1 (ssh) exited rc=$?"
