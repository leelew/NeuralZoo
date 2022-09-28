#!/bin/bash

# Notes:: This file execute models costing overwhelming time.
#         i.e., SVM, RF, adaboost, gbdt

# load module in TH-2
source /WORK/app/toolshs/cnmodule.sh
source /WORK/app/osenv/ln1/set2.sh
module load anaconda3/5.3.0
source activate tf-2.0-py36

# "" for CoLM group server, "/WORK/sysu_yjdai_6" for TH-2 server.
SERVER="/WORK/sysu_yjdai_6"

ROOT="/work/lilu/MetReg/experiments/ERA5/"
num_jobs=200
mdl_name="ml.svr.svm"
input_path="/hard/lilu/ERA5/7D/inputs/"
forecast_path="/hard/lilu/ERA5/7D/forecast/"
model_path="/hard/lilu/ERA5/7D/save/"
score_path="/hard/lilu/ERA5/7D/score/"
hostlist=`yhrun -N 40 -n 200 hostname`
batch_size=32
epochs=50

part=1
space=','
j=0

for i in $hostlist
do
    k=$(($j/$part))
    if [ -z ${host[$k]} ]
    then
        host[$k]=${host[$k]}${i}
    else
        host[$k]=${host[$k]}${space}${i}
    fi
    j=$((j+1))
done

cd $server$ROOT

for ((i=0; i<${num_jobs}; i++))
do
    mpirun -np 1 -host ${host[$i]} python train.py --mdl_name ${mdl_name} \
	   	 --input_path $SERVER$input_path \
		 --forecast_path $SERVER$forecast_path \
		 --model_path $SERVER$model_path \
		 --batch_size ${batch_size} \
		 --epochs ${epochs} \
		 --num_jobs ${i} &
done
    mpirun -np 1 -host ${host[num_jobs-1]} python train.py --mdl_name ${mdl_name} \
	   	 --input_path $SERVER$input_path \
		 --forecast_path $SERVER$forecast_path \
		 --model_path $SERVER$model_path \
		 --batch_size ${batch_size} \
		 --epochs ${epochs} \
		 --num_jobs ${num_jobs}
wait