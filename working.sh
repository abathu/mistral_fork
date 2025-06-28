#!/bin/bash

#SBATCH --job-name=mistral_train_shuffle_nondeterministic_test
#SBATCH --output=/home/s2678328/mistral_impossible/logs/slurm-%j.out
#SBATCH --error=/home/s2678328/mistral_impossible/logs/slurm-%j.out



# conda init
# conda activate mistral

export WANDB_API_KEY=d2df06328f6b7569ace98561f5128290edde5943


echo "🚀 Starting training job..."
echo "⏱️ Training start time:"
python -c "from datetime import datetime; print(datetime.now())"

which python
cd /home/s2678328/mistral_impossible


# CUDA_VISIBLE_DEVICES=0 python3 train.py --config conf/train_reverse_partial_100M_randinit_seed41.yaml --nnodes 1 --nproc_per_node 1 --training_arguments.fp16 true --training_arguments.warmup_steps 300 --training_arguments.max_steps 3000


# --config /home/s2678328/mistral_impossible/conf/datasets/dataset_shuffle_control_100M_seed41.yaml \


CUDA_LAUNCH_BLOCKING=1 python train.py \
    --config  /home/s2678328/mistral_impossible/conf/train_shuffle_nondeterministic_100M_randinit_seed41.yaml \
    --nnodes 1 \
    --nproc_per_node 1 \
    --training_arguments.fp16 true \
    --training_arguments.per_device_train_batch_size 2 \
    --training_arguments.warmup_steps 300 \
    --training_arguments.max_steps  3000 \
 


# CUDA_LAUNCH_BLOCKING=1 python  train.py \
#     --config /home/s2678328/mistral/conf/train_shuffle_control_100M_randinit_seed42.yaml \
#     --nnodes 1 \
#     --nproc_per_node 1 \
#     --training_arguments.fp16 true \
#     --training_arguments.per_device_train_batch_size 2 \
#     --training_arguments.warmup_steps 10 \
#     --training_arguments.max_steps 1000 \


echo "⏱️ Training end time:"
python -c "from datetime import datetime; print(datetime.now())"
echo "✅ Training finished"



# once extra training on the english data 