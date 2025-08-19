#
# MODEL_TYPES:
#
# 0: Transolver
# 1: LNO
# 2: FLARE
# 3: Vanilla Transformer
# 4: GNOT
# 5: UPT (not implemented)
# 6: PerceiverIO
#
#======================================================================#
# Comparison models
# hyperparameters are hard coded in pdebench/__main__.py for now
# for all models but FLARE.
#======================================================================#

for DATASET in elasticity darcy airfoil_steady pipe drivaerml_40k lpbf; do
for MODEL_TYPE in 0 1 4 6; do

    uv run python -m pdebench --dataset ${DATASET} --train true \
        --model_type ${MODEL_TYPE} --exp_name model_${MODEL_TYPE}_${DATASET}

done
done

###
# Transolver with conv2d and/or unified_pos
###

uv run python -m pdebench --dataset darcy --train true \
    --conv2d true --unified_pos true --model_type 0 --exp_name model_0conv_darcy

uv run python -m pdebench --dataset airfoil_steady --train true \
    --conv2d true --model_type 0 --exp_name model_0conv_airfoil_steady

uv run python -m pdebench --dataset pipe --train true \
    --conv2d true --model_type 0 --exp_name model_0conv_pipe

###
# Vanilla Transformer
###

uv run python -m pdebench --dataset elasticity --train true \
    --model_type 3 --exp_name model_3_elasticity

uv run python -m pdebench --dataset darcy --train true \
    --model_type 3 --exp_name model_3_darcy

uv run python -m pdebench --dataset airfoil_steady --train true \
    --model_type 3 --exp_name model_3_airfoil_steady

#======================================================================#
# FLARE
#======================================================================#
DATASET=elasticity
EPOCH=500
BATCH_SIZE=2
WEIGHT_DECAY=1e-5

NUM_BLOCKS=8
NUM_CHANNELS=64
NUM_LATENTS=64
NUM_HEADS=8

uv run python -m pdebench --dataset ${DATASET} --train true --model_type 2 \
    --epochs ${EPOCH} --weight_decay ${WEIGHT_DECAY} --batch_size ${BATCH_SIZE} \
    --channel_dim ${NUM_CHANNELS} --num_latents ${NUM_LATENTS} --num_blocks ${NUM_BLOCKS} --num_heads ${NUM_HEADS} \
    --seed 0 --exp_name model_2_${DATASET}_B_${NUM_BLOCKS}_C_${NUM_CHANNELS}_M_${NUM_LATENTS}_H_${NUM_HEADS}

#======================================================================#
DATASET=darcy
EPOCH=500
BATCH_SIZE=2
WEIGHT_DECAY=1e-5

NUM_BLOCKS=8
NUM_CHANNELS=64
NUM_LATENTS=256
NUM_HEADS=16

uv run python -m pdebench --dataset ${DATASET} --train true --model_type 2 \
    --epochs ${EPOCH} --weight_decay ${WEIGHT_DECAY} --batch_size ${BATCH_SIZE} \
    --channel_dim ${NUM_CHANNELS} --num_latents ${NUM_LATENTS} --num_blocks ${NUM_BLOCKS} --num_heads ${NUM_HEADS} \
    --seed 0 --exp_name model_2_${DATASET}_B_${NUM_BLOCKS}_C_${NUM_CHANNELS}_M_${NUM_LATENTS}_H_${NUM_HEADS}

#======================================================================#
DATASET=airfoil_steady
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
DATASET=pipe
EPOCH=500
BATCH_SIZE=2
WEIGHT_DECAY=1e-5

NUM_BLOCKS=8
NUM_CHANNELS=64
NUM_LATENTS=128
NUM_HEADS=8

uv run python -m pdebench --dataset ${DATASET} --train true --model_type 2 \
    --epochs ${EPOCH} --weight_decay ${WEIGHT_DECAY} --batch_size ${BATCH_SIZE} \
    --channel_dim ${NUM_CHANNELS} --num_latents ${NUM_LATENTS} --num_blocks ${NUM_BLOCKS} --num_heads ${NUM_HEADS} \
    --seed 0 --exp_name model_2_${DATASET}_B_${NUM_BLOCKS}_C_${NUM_CHANNELS}_M_${NUM_LATENTS}_H_${NUM_HEADS}

#======================================================================#
DATASET=drivaerml_40k
EPOCH=500
BATCH_SIZE=1
WEIGHT_DECAY=1e-4

NUM_BLOCKS=8
NUM_CHANNELS=64
NUM_LATENTS=256
NUM_HEADS=8

uv run python -m pdebench --dataset ${DATASET} --train true --model_type 2 \
    --epochs ${EPOCH} --weight_decay ${WEIGHT_DECAY} --batch_size ${BATCH_SIZE} \
    --channel_dim ${NUM_CHANNELS} --num_latents ${NUM_LATENTS} --num_blocks ${NUM_BLOCKS} --num_heads ${NUM_HEADS} \
    --seed 0 --exp_name model_2_${DATASET}_B_${NUM_BLOCKS}_C_${NUM_CHANNELS}_M_${NUM_LATENTS}_H_${NUM_HEADS}

#======================================================================#
DATASET=lpbf
EPOCH=250
BATCH_SIZE=1
WEIGHT_DECAY=1e-4

NUM_BLOCKS=8
NUM_CHANNELS=64
NUM_LATENTS=128
NUM_HEADS=8

uv run python -m pdebench --dataset ${DATASET} --train true --model_type 2 \
    --epochs ${EPOCH} --weight_decay ${WEIGHT_DECAY} --batch_size ${BATCH_SIZE} \
    --channel_dim ${NUM_CHANNELS} --num_latents ${NUM_LATENTS} --num_blocks ${NUM_BLOCKS} --num_heads ${NUM_HEADS} \
    --seed 0 --exp_name model_2_${DATASET}_B_${NUM_BLOCKS}_C_${NUM_CHANNELS}_M_${NUM_LATENTS}_H_${NUM_HEADS}

#======================================================================#
#