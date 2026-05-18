#!/bin/bash
# Parallel training script for AoI-V2X
# Usage: bash run_all.sh
# Modify GPU assignments below before running

set -e

REPO="/archive/hot2/chj/workspace/homework/AoI-V2X-IEEE-TVT-2023"
PYTHON="/home/chenhuaji@corp.sse.tongji.edu.cn/.conda/envs/aoi-v2x/bin/python"

# ============ GPU assignment (modify here) ============
# Available GPUs: check with `nvidia-smi`
# Currently free: 0, 3, 5 (as of 2026-05-17)
# When all 4 GPUs are free, change to e.g. (0 1 2 3)

GPU_ALG1=0   # 1-Modified MADDPG with TDec
GPU_ALG2=3   # 2-Modified MADDPG
GPU_ALG3=5   # 3-MADDPG_FDec
GPU_ALG4=5   # 4-DDPG  (shares GPU with ALG3, only ~159MB each)

# =====================================================

mkdir -p "$REPO/1-Modified MADDPG with TDec/model/marl_model"
mkdir -p "$REPO/2-Modified MADDPG/model/marl_model"
mkdir -p "$REPO/3-MADDPG_FDec/model/marl_model"
mkdir -p "$REPO/4-DDPG/model/marl_model"

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
echo " Starting 4 algorithms in parallel"
echo " GPU allocation: ALG1=$GPU_ALG1  ALG2=$GPU_ALG2  ALG3=$GPU_ALG3  ALG4=$GPU_ALG4"
echo "=========================================="

run_alg $GPU_ALG1 "1-Modified MADDPG with TDec" &
PID1=$!
run_alg $GPU_ALG2 "2-Modified MADDPG" &
PID2=$!
run_alg $GPU_ALG3 "3-MADDPG_FDec" &
PID3=$!
run_alg $GPU_ALG4 "4-DDPG" &
PID4=$!

wait $PID1 $PID2 $PID3 $PID4

echo "=========================================="
echo " All done! Results in each algorithm's model/marl_model/"
echo " Logs: */train.log"
echo "=========================================="
