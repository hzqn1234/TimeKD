#!/bin/sh

#### SBATCH -o gpu-job-%j.output
#SBATCH -o gpu-job-train-2.output
#SBATCH -p NV100q
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1

#SBATCH -n 1
#SBATCH -c 16
#SBATCH -w node18

# export PYTHONPATH=/path/to/project_root:$PYTHONPATH
# export CUDA_LAUNCH_BLOCKING=1


for lr in 1e-4
do
    echo "lr: "$lr
    # for seed in 42
    for seed in 42 420 4200 42000 420000
    do 
        echo "seed: "$seed
        CUDA_VISIBLE_DEVICES=1 python amex_train.py \
                                        --lrate $lr \
                                        --sampling "10pct" \
                                        --data_type "13month" \
                                        --batch_size 16 \
                                        --es_patience 3 \
                                        --num_nodes 223 \
                                        --seed $seed \
                                        --predict \
                                        --num_workers 16 \
                                        --epochs 20
    done
done

