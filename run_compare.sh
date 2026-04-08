#!/bin/bash
set -euo pipefail

SCRIPT="train_fsdp_tp.py"
LOG_FSDP_TP="log.txt"
LOG_FSDP_ONLY="ref.txt"
DEBUG_FSDP_TP="./debug_fsdp_tp"
DEBUG_FSDP_ONLY="./debug_fsdp_only"

MODEL_NAME="${MODEL_NAME:-hf-internal-testing/Mixtral-tiny}"
COMMON_ARGS="--model_name $MODEL_NAME --num_steps 20 --lr 3e-4 --seed 42"

# echo "=== Generating fixed batches for $MODEL_NAME ==="
# python generate_fixed_batches.py "$MODEL_NAME"

rm -rf ./checkpoints_tp ./checkpoints_fsdp

echo "=== Running FSDP+TP (8 GPUs: fsdp=4, tp=2) then FSDP-only (4 GPUs: fsdp=4) ==="

echo "--- FSDP+TP ---"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=29500 $SCRIPT $COMMON_ARGS --fsdp_size 4 --tp_size 2 --enable_sp --save_dir ./checkpoints_tp > "$LOG_FSDP_TP" 2>&1
echo "FSDP+TP done"

echo "--- FSDP-only ---"
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29501 $SCRIPT $COMMON_ARGS --fsdp_size 4 --save_dir ./checkpoints_fsdp > "$LOG_FSDP_ONLY" 2>&1
echo "FSDP-only done"


echo ""
echo "=== Loss & Grad Diff ==="
git diff --no-index --color --word-diff=color "$LOG_FSDP_TP" "$LOG_FSDP_ONLY" || true