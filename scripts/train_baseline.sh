#!/bin/bash
# Train PPO baseline (no DEP) for motion imitation
# Usage: ./scripts/train_baseline.sh [exp_name]

EXP_NAME=${1:-kinesis_im_baseline}
python src/run.py \
    --config-name config_legs \
    exp_name=$EXP_NAME \
    learning.use_dep_exploration=False
