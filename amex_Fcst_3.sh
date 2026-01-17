#!/bin/sh

#### SBATCH -o gpu-job-%j.output
#SBATCH -o gpu-job-train-3.output
#SBATCH -p PA100q
#SBATCH --gpus-per-node=1

#SBATCH -n 1
#SBATCH -c 16
#SBATCH -w node02

# export PYTHONPATH=/path/to/project_root:$PYTHONPATH
# export CUDA_LAUNCH_BLOCKING=1

for lr in 1e-5
do
    echo "lr: "$lr
    for seed in 42
    do 
        echo "seed: "$seed
        CUDA_VISIBLE_DEVICES=3 python amex_train.py \
                                        --lrate $lr \
                                        --sampling "100pct" \
                                        --data_type "original" \
                                        --num_nodes 223 \
                                        --es_patience 3 \
                                        --seed $seed \
                                        --train \
                                        --predict \
                                        --submit \
                                        --batch_size 256 \
                                        --num_workers 16 \
                                        --epochs 20 
    done
done

