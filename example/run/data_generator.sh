#!/bin/bash

source /WORK/app/toolshs/cnmodule.sh
source /WORK/app/osenv/ln1/set2.sh
module load anaconda3/5.3.0
source activate tf-2.0-py36


ROOT="/WORK/sysu_yjdai_6/work/lilu/MetReg/experiments/ERA5/"

cd ${ROOT}

python3 data_generator.py --preliminary_path "/WORK/sysu_yjdai_6/hard/lilu/ERA5/preliminary/" \
			  --input_path "/WORK/sysu_yjdai_6/hard/lilu/ERA5/7D/inputs/" \
			  --intervel 18 \
			  --len_inputs 10 \
			  --window_size 7
