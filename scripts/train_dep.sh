#!/bin/bash
# Train PPO + DEP exploration for motion imitation
# Requires run=train_run_legs_dep (includes muscle_len, muscle_force in observations)
# Usage: ./scripts/train_dep.sh [exp_name]
# Ensure conda env is active: conda activate kinesis

cd "$(dirname "$0")/.." || exit 1
EXP_NAME=${1:-kinesis_im_dep}
python src/run.py \
    --config-name config_legs \
    run=train_run_legs_dep \
    exp_name=$EXP_NAME \
    learning.use_dep_exploration=True \
    no_log=True
