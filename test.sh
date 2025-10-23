#!/bin/bash
. /root/miniconda3/etc/profile.d/conda.sh
conda activate sandbox-runtime
PYTHONUNBUFFERED=1 python -m sandbox.test