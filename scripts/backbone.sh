#!/bin/bash -l
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --qos debug
#SBATCH --time=01:00:00
#SBATCH --account=ntrain3
#SBATCH --output=/global/homes/m/marcolz/DETR/backbone_%j.out

python="python /global/homes/m/marcolz/DETR/hgp_detr/main.py"

for backbone in resnet18 resnet34 resnet50 resnet101 resnet152
do
    output="/global/homes/m/marcolz/DETR/$backbone.out"
    $python --epochs 3  --backbone $backbone --dataset_file hgp --use_amp | tee -a $output
done