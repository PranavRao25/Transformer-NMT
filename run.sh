#!/bin/bash

# 1. Fail-fast mechanism
set -e 

# 2. (Optional) Activate your environment here if not already active in the shell
# source /path/to/your/venv/bin/activate
# or
# conda activate your_env

echo "=========================================="
echo "STARTING TRAINING FOR UNSCALED"
echo "=========================================="
# 3. Stream output to both terminal and a log file
python3 -m src.train 2>&1 | tee logs/train_unscaled.log

echo "=========================================="
echo "STARTING TRAINING FOR CONSTANT LR"
echo "=========================================="
# If src.train fails, the script will exit before reaching this point
python3 -m src.train_2 2>&1 | tee logs/train_constant_lr.log

echo "ALL PIPELINES COMPLETED SUCCESSFULLY."
