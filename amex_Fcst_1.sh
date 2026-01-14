#!/bin/sh

#### SBATCH -o gpu-job-%j.output
#SBATCH -o gpu-job-train-1.output
#SBATCH -p RTXA6Kq
#SBATCH --gpus-per-node=1

#SBATCH -n 1
#SBATCH -c 8
#SBATCH -w node08

# export PYTHONPATH=/path/to/project_root:$PYTHONPATH
# export CUDA_LAUNCH_BLOCKING=1

for lr in 1e-4
do
    echo "lr: "$lr
    for seed in 42
    do 
        echo "seed: "$seed
        CUDA_VISIBLE_DEVICES=6,7 python amex_train.py \
                                        --lrate $lr \
                                        --sampling "100pct" \
                                        --data_type "original" \
                                        --num_nodes 223 \
                                        --es_patience 3 \
                                        --seed $seed \
                                        --train \
                                        --test \
                                        --predict \
                                        --batch_size 32 \
                                        --num_workers 16 \
                                        --epochs 20 
    done
done

