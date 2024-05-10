#!/bin/bash -l
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --qos debug
#SBATCH --time=00:30:00
#SBATCH --account=m3930
#SBATCH --output=/global/homes/m/marcolz/DETR/gpu_reports/GPU1/slurm/slurm_%j.out

# --------------------------------------------------------------------------------
# This file has been modified from its original version. It retains the original
# copyright notice, terms, and disclaimer as specified in the Lawrence Berkeley 
# National Labs BSD variant license. The modifications are provided under the same 
# license terms as the original, with no endorsements from the original authors or 
# affiliated institutions unless expressly stated.
# --------------------------------------------------------------------------------

# pre-run
module load gpu
module load conda
conda activate detr_12.2
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=12345
dcgmi profile --pause

# Time
metrics="sm__cycles_elapsed.avg,\
sm__cycles_elapsed.avg.per_second,"

# DP
metrics+="sm__sass_thread_inst_executed_op_dadd_pred_on.sum,\
sm__sass_thread_inst_executed_op_dfma_pred_on.sum,\
sm__sass_thread_inst_executed_op_dmul_pred_on.sum,"

# SP
metrics+="sm__sass_thread_inst_executed_op_fadd_pred_on.sum,\
sm__sass_thread_inst_executed_op_ffma_pred_on.sum,\
sm__sass_thread_inst_executed_op_fmul_pred_on.sum,"

# HP
metrics+="sm__sass_thread_inst_executed_op_hadd_pred_on.sum,\
sm__sass_thread_inst_executed_op_hfma_pred_on.sum,\
sm__sass_thread_inst_executed_op_hmul_pred_on.sum,"

# Tensor Core
metrics+="sm__inst_executed_pipe_tensor.sum,"

# DRAM, L2 and L1
metrics+="dram__bytes.sum,\
lts__t_bytes.sum,\
l1tex__t_bytes.sum"

# ncu flags
export lc=100

# detr parameters

export epochs=3
export backbone="resnet18"
export enc_layers=2
export dec_layers=2
export dim_ff=512
export hidden_dim=64
# export nheads=4
export queries=20
export file="hgp"
export batch_size=2


python="python /global/homes/m/marcolz/DETR/hgp_detr/main.py"
args=" --batch_size $batch_size --epochs $epochs --backbone $backbone --enc_layers $enc_layers --dec_layers $dec_layers --dim_feedforward $dim_ff --hidden_dim $hidden_dim --num_queries $queries --dataset_file $file"

dir=/global/homes/m/marcolz/DETR/gpu_reports/GPU1
pp_dir=/global/homes/m/marcolz/DETR/hgp_detr/roofline-on-nvidia-gpus/custom-scripts

for nheads in 4 8 16 32
do
    output=output_nheads$nheads.txt
    echo nheads: $nheads
    echo "$python --nheads $nheads $args > $dir/txt/$output 2>&1"
    $python --nheads $nheads $args > $dir/txt/$output 2>&1
done


# for nheads in 4 8 16 32
# do
#     output=output_nheads$nheads.csv
#     profileargstr="--target-processes all --nvtx --kernel-id :::6 --launch-count $lc --metrics $metrics --csv"
#     echo nheads: $nheads
#     echo "ncu $profileargstr $python --nheads $nheads $args > $pp_dir/$output 2>&1"
#     ncu $profileargstr $python --nheads $nheads $args > $pp_dir/$output 2>&1
# done


# cd $pp_dir
# srun -n1 python /global/homes/m/marcolz/DETR/hgp_detr/roofline-on-nvidia-gpus/custom-scripts/postprocess.py
