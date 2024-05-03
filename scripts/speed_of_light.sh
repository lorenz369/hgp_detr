#!/bin/bash -l
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --qos debug
#SBATCH --time=00:30:00
#SBATCH --account=m3930
#SBATCH --output=/global/homes/m/marcolz/DETR/gpu_reports/GPU1/slurm/slurm_%j.out

# --------------------------------------------------------------------------------
# File added by Marco Lorenz in April 2024 to profile different configurations of hyperparameters
# with Nvidia Nsight Compute on a slurm system (NERSC-9, Perlmutter).
# This modification is made under the terms of the Apache License 2.0, which is the license
# originally associated with this file. All original copyright, patent, trademark, and
# attribution notices from the Source form of the Work have been retained, excluding those 
# notices that do not pertain to any part of the Derivative Works.
# --------------------------------------------------------------------------------

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
export k="regex:elementwise"
export ls=2
export lc=100
export root="/global/homes/m/marcolz/DETR/gpu_reports/GPU1/ncu"
export branch1="debug_target_processes"
export branch2="debug_no_target_processes"

# detr parameters

export epochs=1
export backbone="resnet18"
export enc_layers=2
export dec_layers=2
export dim_ff=512
export hidden_dim=64
export nheads=4
export queries=20
export file="hgp"
export ratio=0.1
export cupy="cupy"
export csv="csv"

input="python /global/homes/m/marcolz/DETR/hgp_detr/main.py"
input+=" --epochs $epochs --backbone $backbone --enc_layers $enc_layers --dec_layers $dec_layers --dim_feedforward $dim_ff --hidden_dim $hidden_dim --nheads $nheads --num_queries $queries --dataset_file $file"

# pre-run
module load gpu
module load conda
conda activate detr_12.2
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=12345
dcgmi profile --pause
    

echo " ncu --profile-from-start off --target-processes all --nvtx --kernel-id :::1 --kernel-name $k --launch-count $lc --metrics $metrics --export=\"$root/sol_startoff_targetall\" --csv $input "
ncu --profile-from-start off --target-processes all --nvtx --kernel-id :::1 --kernel-name $k --launch-count $lc --metrics $metrics --export=\"$root/sol_startoff_targetall\" --csv $input

echo " ncu  --nvtx --kernel-id :::1 --kernel-name $k --launch-count $lc --metrics $metrics --export=\"$root/sol_wo_fstartoff_targetall\" --csv $input "
ncu --nvtx --kernel-id :::1 --kernel-name $k --launch-count $lc --metrics $metrics --export=\"$root/sol_wo_fstartoff_targetall\" --csv $input


