#!/bin/bash

# Notes:: This file execute models costing overwhelming time.
#         i.e., SVM, RF, adaboost, gbdt

# load module in TH-2
source /WORK/app/toolshs/cnmodule.sh
source /WORK/app/osenv/ln1/set2.sh
module load anaconda3/5.3.0
source activate tf-2.0-py36

hostlist=`yhrun -N 20 -n 100 hostname`
num_jobs=100
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

cd '/WORK/sysu_yjdai_6/work/lilu/experiments/ERA5/'


mdl_name="dl.rnn.lstm"

for ((i=0; i<${num_jobs}; i++))
do
    mpirun -np 1 -host ${host[$i]} python main.py --mdl_name ${mdl_name} &
done
    mpirun -np 1 -host ${host[num_jobs-1]} python main.py --mdl_name ${mdl_name}
wait
