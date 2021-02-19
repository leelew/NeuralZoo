#!/bin/bash

# load module in TH-2
source /WORK/app/toolshs/cnmodule.sh
source /WORK/app/osenv/ln1/set2.sh
module load anaconda3/5.3.0
source activate tf-2.0-py36

cd '/WORK/sysu_yjdai_6/work/lilu/MetReg/experiments/ERA5/'

python3 _data_generator.py