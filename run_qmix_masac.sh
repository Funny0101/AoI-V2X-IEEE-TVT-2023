#!/bin/bash
# Run QMIX (9) and MASAC (10) in parallel
set -e

REPO="/archive/hot2/chj/workspace/homework/AoI-V2X-IEEE-TVT-2023"
PYTHON="/home/chenhuaji@corp.sse.tongji.edu.cn/.conda/envs/aoi-v2x/bin/python"

GPU_A=1   # 9-Modified MADDPG with QMIX
GPU_B=2   # 10-Modified MADDPG with MASAC

mkdir -p "$REPO/9-Modified MADDPG with QMIX/model/marl_model"
mkdir -p "$REPO/10-Modified MADDPG with MASAC/model/marl_model"

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
echo " Starting QMIX and MASAC in parallel"
echo " GPU: QMIX=$GPU_A  MASAC=$GPU_B"
echo "=========================================="

run_alg $GPU_A "9-Modified MADDPG with QMIX" &
PIDA=$!
run_alg $GPU_B "10-Modified MADDPG with MASAC" &
PIDB=$!

wait $PIDA $PIDB

echo "=========================================="
echo " All done! Check */train.log for results"
echo "=========================================="
