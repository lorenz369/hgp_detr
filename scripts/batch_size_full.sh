#!/bin/bash -l
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --qos debug
#SBATCH --time=00:59:00
#SBATCH --account=ntrain3
#SBATCH --output=/global/homes/m/marcolz/DETR/batch_size_%j.out

python="python /global/homes/m/marcolz/DETR/hgp_detr/main.py"

for batch_size in 2 4 8 16 32 64 128 256
do
    output="/global/homes/m/marcolz/DETR/batch_size_{$batch_size}_full.out"
    $python --epochs 3 --batch_size $batch_size --dataset_file hgp --use_amp | tee -a $output
done