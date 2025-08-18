#!/bin/bash

#============================#
# Advanced Queue
#============================#
#SBATCH --account=lkara
#SBATCH --partition=advanced
#SBATCH --qos=adv_4gpu_qos
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=FLARE

# #============================#
# # General Queue
# #============================#
# #SBATCH --account=lkara
# #SBATCH --partition=general
# #SBATCH --qos=general_qos
# #SBATCH --gres=gpu:1
# #SBATCH --time=12:00:00
# #SBATCH --job-name=FLARE

# #============================#
# # Preempt Queue
# #============================#
# #SBATCH --account=lkara
# #SBATCH --partition=preempt
# #SBATCH --qos=preempt_qos
# #SBATCH --gres=gpu:1
# #SBATCH --time=24:00:00
# #SBATCH --job-name=FLARE

#============================#
# Setup
#============================#
echo "Setting up environment"
source ~/.bash_profile
cd /project/community/vedantpu/FLARE.py

#============================#
# Run interactively with
# $ srun --pty --overlap --jobid <job_id> bash
#============================#
# sleep 24h
#============================#

# uv run python -m pdebench --train true \
#     --dataset darcy --model_type 3 --exp_name model_3_darcy
#======================================================================#
DATASET=darcy
EPOCH=500
BATCH_SIZE=2
WEIGHT_DECAY=1e-5

NUM_BLOCKS=8
NUM_CHANNELS=64
NUM_LATENTS=256
NUM_HEADS=8

uv run python -m pdebench --dataset ${DATASET} --train true --model_type 2 \
    --epochs ${EPOCH} --weight_decay ${WEIGHT_DECAY} --batch_size ${BATCH_SIZE} \
    --channel_dim ${NUM_CHANNELS} --num_latents ${NUM_LATENTS} --num_blocks ${NUM_BLOCKS} --num_heads ${NUM_HEADS} \
    --seed 0 --exp_name model_2_${DATASET}_B_${NUM_BLOCKS}_C_${NUM_CHANNELS}_M_${NUM_LATENTS}_H_${NUM_HEADS}

#======================================================================#

#============================#
wait
#============================#
#