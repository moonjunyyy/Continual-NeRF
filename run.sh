#!/bin/bash

#SBATCH -J DataCondensation
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

python train.py --expname NERF --logdir ./log --ckptdir ./save --data_path /local_datasets/NMR_Dataset --batch_size 32 --data_type dvr --train_tgt_views 6 9 12 --img_hw 64 64 --val_src_views 3 3 3 --val_tgt_views 6 9 12