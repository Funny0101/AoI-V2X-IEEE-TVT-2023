#!/bin/bash
# Run the 3 new algorithms in parallel
# Usage: bash run_new.sh

set -e

REPO="/archive/hot2/chj/workspace/homework/AoI-V2X-IEEE-TVT-2023"
PYTHON="/home/chenhuaji@corp.sse.tongji.edu.cn/.conda/envs/aoi-v2x/bin/python"

# ============ GPU assignment (modify here) ============
# Available GPUs: 0, 3, 5 (as of 2026-05-18)
# ALG7 shares GPU with ALG6 since they are lightweight (~159MB each)

GPU_ALG5=0   # 5-Modified MADDPG with Attention
GPU_ALG6=3   # 6-Modified MADDPG with AoI-Enhanced
GPU_ALG7=5   # 7-Modified MADDPG with Attention+AoI

# =====================================================

mkdir -p "$REPO/5-Modified MADDPG with Attention/model/marl_model"
mkdir -p "$REPO/6-Modified MADDPG with AoI-Enhanced/model/marl_model"
mkdir -p "$REPO/7-Modified MADDPG with Attention+AoI/model/marl_model"

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
echo " Starting 3 new algorithms in parallel"
echo " GPU: ALG5=$GPU_ALG5  ALG6=$GPU_ALG6  ALG7=$GPU_ALG7"
echo "=========================================="

run_alg $GPU_ALG5 "5-Modified MADDPG with Attention" &
PID5=$!
run_alg $GPU_ALG6 "6-Modified MADDPG with AoI-Enhanced" &
PID6=$!
run_alg $GPU_ALG7 "7-Modified MADDPG with Attention+AoI" &
PID7=$!

wait $PID5 $PID6 $PID7

echo "=========================================="
echo " All done! Results in each algorithm's model/marl_model/"
echo " Logs: */train.log"
echo "=========================================="
