#!/bin/bash

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

path_inputs="/WORK/sysu_yjdai_6/lilu/inputs/SMAP_lstm/"
path_outputs="/WORK/sysu_yjdai_6/lilu/outputs/lstm/"
epoch=40

cd '/WORK/sysu_yjdai_6/lilu/'
for ((i=0; i<${num_jobs}; i++))
do
    mpirun -np 1 -host ${host[$i]} python ./src/main.py --path_inputs ${path_inputs} \
                                                        --path_outputs ${path_outputs} \
                                                        --num_jobs $i \
                                                        --epoch ${epoch} \
                                                        --split_ratio 0.3 &
done
    mpirun -np 1 -host ${host[num_jobs-1]} python ./src/main.py --path_inputs ${path_inputs} \
                                                                --path_outputs ${path_outputs} \
                                                                --num_jobs ${num_jobs} \
                                                                --epoch ${epoch} \
                                                                --split_ratio 0.3
wait
