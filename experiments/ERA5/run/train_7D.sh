#!/bin/bash

SERVER=""

ROOT="/work/lilu/MetReg/experiments/ERA5/"
num_jobs=200
mdl_name="ml.linear.ridge"
input_path="/hard/lilu/ERA5/7D/inputs/"
forecast_path="/hard/lilu/ERA5/7D/forecast/"
model_path="/hard/lilu/ERA5/7D/save/"
score_path="/hard/lilu/ERA5/7D/score/"  


cd ${ROOT}

for ((i=0; i<${num_jobs}; i++))
do
    python3 train.py --mdl_name ${mdl_name} \
	   	 --input_path $SERVER$input_path \
		 --forecast_path $SERVER$forecast_path \
		 --model_path $SERVER$model_path \
		 --batch_size 32 \
		 --epochs 50 \
		 --num_jobs ${i}
done
