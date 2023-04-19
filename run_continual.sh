#!/bin/bash

#SBATCH -J NERF_CON_2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=16G
#SBATCH --time=4-0
#SBATCH -o %x_%j_%a.out
#SBATCH -e %x_%j_%a.err

date
ulimit -n 65536
### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=$(($RANDOM+32769))
export WORLD_SIZE=$SLURM_NNODES

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

source /data/moonjunyyy/init.sh
conda activate visionnerf

python train_continual.py \
    --expname NERF_CON_2 \
    --logdir ./log \
    --ckptdir ./save \
    --data_path /local_datasets/NMR_Dataset \
    --batch_size 1 \
    --chunk_size 2048 \
    --mlp_block_num 6 \
    --im_feat_dim 512 \
    --lrate_feature 1e-5 \
    --lrate_mlp 1e-4 \
    --sample_mode bbox_sample_full \
    --bbox_steps 50000 \
    --use_warmup \
    --warmup_steps 10000 \
    --lrate_decay_factor 0.1 \
    --lrate_decay_steps 450000 \
    --n_iters 500000 \
    --data_type dvr \
    --train_src_views 2 2 2 \
    --train_indices 3 1053 2103 3153 4203 \
    --train_tgt_views 1 6 10 \
    --img_hw 64 64 \
    --val_indices 2 228 456 677 903 \
    --val_src_views 2 2 2 \
    --val_tgt_views 1 6 10