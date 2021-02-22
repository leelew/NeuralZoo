#!/bin/bash

# load module in TH-2
#source /WORK/app/toolshs/cnmodule.sh
#source /WORK/app/osenv/ln1/set2.sh
#module load anaconda3/5.3.0
#source activate tf-2.0-py36
SERVER=""

ROOT="/work/lilu/MetReg/experiments/ERA5/"
num_jobs=200
mdl_name="ml.linear.ridge"
input_path="/hard/lilu/ERA5/7D/inputs/"
forecast_path="/hard/lilu/ERA5/7D/forecast/"
model_path="/hard/lilu/ERA5/7D/save/"
score_path="/hard/lilu/ERA5/7D/score/" 

cd '/work/lilu/MetReg/experiments/ERA5/'

python3 inference.py --mdl_name ${mdl_name} \
	   	 			 --input_path $SERVER$input_path \
		 			 --forecast_path $SERVER$forecast_path \
					 --score_path $SERVER$score_path