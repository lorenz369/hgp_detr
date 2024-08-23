#!/bin/bash -l
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --qos debug
#SBATCH --time=00:30:00
#SBATCH --account=m2845
#SBATCH --output=/global/homes/m/marcolz/DETR/batch_size_%j.out

module load conda
conda activate detr_12.2
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=12345

python="python /global/homes/m/marcolz/DETR/hgp_detr/main.py"

for batch_size in 2 4 8 16 32 64 128 256
do
    output="/global/homes/m/marcolz/DETR/batch_size_amp_$batch_size.out"
    $python --epochs 3 --batch_size $batch_size --dataset_file hgp | tee -a $output
done