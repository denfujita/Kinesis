#!/bin/bash
# Train PPO baseline (no DEP) for motion imitation
# Usage: ./scripts/train_baseline.sh [exp_name]
# Ensure conda env is active: conda activate kinesis

cd "$(dirname "$0")/.." || exit 1
EXP_NAME=${1:-kinesis_im_baseline}
python src/run.py \
    --config-name config_legs \
    exp_name=$EXP_NAME \
    learning.use_dep_exploration=False \
    no_log=True
