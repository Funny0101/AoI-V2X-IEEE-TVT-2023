#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
export PYTHON_EXEC="/home/chenhuaji@corp.sse.tongji.edu.cn/.conda/envs/aoi-v2x/bin/python"

echo "Starting Solution 1 (AoI-ParamShare-Fixed) in background..."
cd "12-Modified MADDPG with AoI-ParamShare"
nohup $PYTHON_EXEC Main.py > train_sol1.log 2>&1 &
cd ..

echo "Starting Solution 3 (AoI-ParamShare-Gumbel) in background..."
cd "16-Modified MADDPG with Gumbel-Softmax"
nohup $PYTHON_EXEC Main.py > train_sol3.log 2>&1 &
cd ..

echo "Both processes started on GPU 2. You can check train_sol1.log and train_sol3.log for progress."
