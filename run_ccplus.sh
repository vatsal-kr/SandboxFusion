#!/bin/bash
. /root/miniconda3/etc/profile.d/conda.sh
conda activate sandbox-runtime
PYTHONUNBUFFERED=1 python -m sandbox.ccplus --model_name $1 --language $2