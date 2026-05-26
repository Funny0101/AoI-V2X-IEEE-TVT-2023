#!/bin/bash
# Run 3 new improvement algorithms in parallel
set -e

REPO="/archive/hot2/chj/workspace/homework/AoI-V2X-IEEE-TVT-2023"
PYTHON="/home/chenhuaji@corp.sse.tongji.edu.cn/.conda/envs/aoi-v2x/bin/python"

GPU_A=0   # 5-Modified MADDPG with PER
GPU_B=5   # 7-Modified MADDPG with NoiseDecay
GPU_C=6   # 8-Modified MADDPG with Nstep-TD

mkdir -p "$REPO/5-Modified MADDPG with PER/model/marl_model"
mkdir -p "$REPO/7-Modified MADDPG with NoiseDecay/model/marl_model"
mkdir -p "$REPO/8-Modified MADDPG with Nstep-TD/model/marl_model"

run_alg() {
    local gpu=$1
    local dir="$REPO/$2"
    local name="$2"
    local log="$dir/train.log"
    echo "[START] $name on GPU $gpu (log: $log)"
    CUDA_VISIBLE_DEVICES=$gpu $PYTHON "$dir/Main.py" > "$log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "[DONE]  $name finished successfully"
    else
        echo "[FAIL]  $name exited with code $rc"
    fi
}

echo "=========================================="
echo " Starting 3 algorithms in parallel"
echo " GPU: PER=$GPU_A  NoiseDecay=$GPU_B  Nstep-TD=$GPU_C"
echo "=========================================="

run_alg $GPU_A "5-Modified MADDPG with PER" &
PIDA=$!
run_alg $GPU_B "7-Modified MADDPG with NoiseDecay" &
PIDB=$!
run_alg $GPU_C "8-Modified MADDPG with Nstep-TD" &
PIDC=$!

wait $PIDA $PIDB $PIDC

echo "=========================================="
echo " All done! Check */train.log for results"
echo "=========================================="
