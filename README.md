--------------------------------------------------------------------------------
Modified by Marco Lorenz in April 2024.
The following contains a fork of https://github.com/facebookresearch/detr to apply the detection transformer (DETR)
to the Hands, Guns and Phones dataset (https://paperswithcode.com/dataset/hgp), and to 
examine and profile its execution.

All modifications are made under the terms of the Apache License 2.0, which is the license
originally associated with this file and repository. All original copyright, patent, 
trademark, and attribution notices from the Source form of the Work have been retained, 
excluding those notices that do not pertain to any part of the Derivative Works.
--------------------------------------------------------------------------------

# Table of contents
* [Introduction](#introduction)
* [DETR - Background](#Background)
* [Usage - Training with HGP Dataset](#Usage)
* [Demo - Inference with trained model](#demo)
* [Profiling - Roofline Methodology for Perlmutter](#Profiling)

# Introduction
If you would like to explore the original DETR and its associated COCO Dataset, please refer to the original repository. For details see [End-to-End Object Detection with Transformers](https://ai.facebook.com/research/publications/end-to-end-object-detection-with-transformers) by Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko.

This repository presents instructions how to train DETR with the Hands, Guns and Phones dataset, and how to profile it with [Nvidia Nsight Compute](https://developer.nvidia.com/nsight-compute). 
Furthermore, it includes an extension of the roofline methodology to profile [Nvidia A100 Tensor Core GPUs](https://www.nvidia.com/en-us/data-center/a100/), which is based on [this NERSC repository](https://gitlab.com/NERSC/roofline-on-nvidia-gpus). The profiling scripts and instructions are designed for NERSC's current High-Performance-Computing System, Perlmutter, but can be extended to other systems and GPUs.

For details on the roofline methodology see [Hierarchical Roofline Performance Analysis for Deep Learning Applications](https://arxiv.org/abs/2009.05257) by Charlene Yang, Yunsong Wang, Steven Farrell, Thorsten Kurth, and Samuel Williams
For details on NERSC and Perlmutter see [Getting started at NERSC](https://docs.nersc.gov/getting-started/).

# Background
Original documentation of https://github.com/facebookresearch/detr:
**DE⫶TR**: End-to-End Object Detection with Transformers
========

[![Support Ukraine](https://img.shields.io/badge/Support-Ukraine-FFD500?style=flat&labelColor=005BBB)](https://opensource.fb.com/support-ukraine)

PyTorch training code and pretrained models for **DETR** (**DE**tection **TR**ansformer).
We replace the full complex hand-crafted object detection pipeline with a Transformer, and match Faster R-CNN with a ResNet-50, obtaining **42 AP** on COCO using half the computation power (FLOPs) and the same number of parameters. Inference in 50 lines of PyTorch.

![DETR](.github/DETR.png)

**What it is**. Unlike traditional computer vision techniques, DETR approaches object detection as a direct set prediction problem. It consists of a set-based global loss, which forces unique predictions via bipartite matching, and a Transformer encoder-decoder architecture. 
Given a fixed small set of learned object queries, DETR reasons about the relations of the objects and the global image context to directly output the final set of predictions in parallel. Due to this parallel nature, DETR is very fast and efficient.

**About the code**. We believe that object detection should not be more difficult than classification,
and should not require complex libraries for training and inference.
DETR is very simple to implement and experiment with, and we provide a
[standalone Colab Notebook](https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_demo.ipynb)
showing how to do inference with DETR in only a few lines of PyTorch code.
Training code follows this idea - it is not a library,
but simply a [main.py](main.py) importing model and criterion
definitions with standard training loops.

Additionnally, we provide a Detectron2 wrapper in the d2/ folder. See the readme there for more information.

For details see [End-to-End Object Detection with Transformers](https://ai.facebook.com/research/publications/end-to-end-object-detection-with-transformers) by Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko.

See our [blog post](https://ai.facebook.com/blog/end-to-end-object-detection-with-transformers/) to learn more about end to end object detection with transformers.

# Usage

## Getting started
There are no extra compiled components in DETR and package dependencies are minimal,
so the code is very simple to use. We provide instructions how to install dependencies via conda.
First, clone the repository locally:
```
git clone https://github.com/lorenz369/hgp_detr.git #--branch amp/no-cupy
```
To test [Automatic Mixed Precision](https://pytorch.org/docs/stable/notes/amp_examples.html) provided by PyTorch, check out branch 'amp'.
To get rid of the cupy dependency, check out branch 'no_cupy'.

Then, install PyTorch 1.5+ and torchvision 0.6+:
```
conda create -n detr -c pytorch pytorch torchvision
conda activate detr
```
Install pycocotools (for evaluation on COCO), cuda (for cupy-annotations) and scipy (for training):
```
conda install cython scipy
conda install cuda -c nvidia

pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```
That's it, should be good to train and evaluate detection models.


To train a lightweight example configuration on the HGP dataset (GPU):
```
python main.py --batch_size 2 --epochs 3  --backbone resnet18 --enc_layers 1 --dec_layers 1 --dim_feedforward 512 --hidden_dim 64 --nheads 2 --num_queries 5 --dataset_file hgp
```

To train on the HGP dataset (GPU):
```
python main.py --batch_size 2 --epochs 300  --backbone resnet18 --enc_layers 2 --dec_layers 2 --dim_feedforward 2048 --hidden_dim 256 --nheads 32 --num_queries 5 --dataset_file hgp
```

To train a lightweight example configuration on the HGP dataset (CPU):
```
python main.py --batch_size 2 --epochs 3  --backbone resnet18 --enc_layers 1 --dec_layers 1 --dim_feedforward 512 --hidden_dim 64 --nheads 2 --num_queries 5 --device cpu --dataset_file hgp
```

To obtain a checkpoint at your output_dir:
```
python main.py --dataset_file hgp --output_dir /your/output_dir
```

To resume from previously obtained checkpoint at your input_dir:
```
python main.py --dataset_file hgp --resume /your/input_dir
```

To evaluate a previously trained model at your input_dir:
```
python main.py --batch_size 2 --no_aux_loss --eval --dataset_file hgp --resume /your/input_dir
```

'results/checkpoints' contains a log file of a sample training run.

## Argument Help

| Flag | explanation | default |
| --------------- | --------------- | --------------- |
| --lr    | Learning rate transformer    | 1e-4    |
| --lr_backbone    | Learning rate backbone    | 1e-5    |
| --batch_size    | Batch size    | 2    |
| --weight_decay    | Weight decay    | 1e-4    |
| --epochs    | Training epochs   | 300    |
| --lr_drop    | Learning rate drop after epoch    | 200    |
| ----clip_max_norm   | Gradient clipping max norm    | 0.1    |

| Model parameters | explanation | default |
| --------------- | --------------- | --------------- |
| --frozen_weights    | Path to the pretrained model. If set, only the mask head will be trained    | None |

| Backbone | explanation | default |
| --------------- | --------------- | --------------- |
| --backbone    | Name of the convolutional backbone to use    | resnet50    |
| --dilation    | If true, we replace stride with dilation in the last convolutional block (DC5)    | False    |
| --position_embedding    | Type of positional embedding to use on top of the image features    | Row 1 Cell 2    |

| Transformer | explanation | default |
| --------------- | --------------- | --------------- |
| --enc_layers    | Number of encoding layers in the transformer    | 6    |
| --dec_layers    | Number of decoding layers in the transformer    | 6    |
| --dim_feedforward    | Intermediate size of the feedforward layers in the transformer blocks    | 2048    |
| --hidden_dim    | Size of the embeddings (dimension of the transformer)    | 256    |
| --dropout    | Dropout applied in the transformer    | 0.1    |
| --nheads    | Number of attention heads inside the transformer's attentions    | 8    |
| --num_queries    | Number of query slots    | 100    |
| --pre_norm    | Pre-Normalizattion before transformer layers    | False    |

| Loss | explanation | default |
| --------------- | --------------- | --------------- |
| --no_aux_loss    | Disables auxiliary decoding losses (loss at each layer)   | False    |

Matcher | explanation | default |
| --------------- | --------------- | --------------- |
| --set_cost_class    | Class coefficient in the matching cost    | 1    |
| --set_cost_bbox    | L1 box coefficient in the matching cost    | 5    |
| --set_cost_giou    | giou box coefficient in the matching cost    | 2    |

| Loss coefficients | explanation | default |
| --------------- | --------------- | --------------- |
| --bbox_loss_coef    | Coefficient of the bbox in the loss    | 5    |
| --giou_loss_coef    | giou coefficient in the loss    | 2    |
| --eos_coef    | Relative classification weight of the no-object class    | 0.1    |

| Dataset parameters | explanation | default |
| --------------- | --------------- | --------------- |
| --dataset_file      | Path to the dataset  | coco   |
| --output_dir    | path where to save, empty for no saving   | ''   |
| --device    | device to use for training / testing    | cuda    |
| --seed    | Seed for reproduceability    | 42    |
| --resume    | resume from checkpoint    | ''    |
| --start_epoch    | start epoch    | 0    |
| --eval    | Evaluation    | False    |
| --num_workers    | num_workers    | 2    |
| --fast_dev_run | Sample a random subset of the dataset for faster training (only for profiling purposes) | 1.0 |

| Distributed training parameters | explanation | default |
| --------------- | --------------- | --------------- |
| --world_size    | number of distributed processes    | 1    |
| --dist_url    | url used to set up distributed training    | 'env://'    |


# Demo

Subdirectory 'results' contains 'detr_hands_on.ipynb', which shows how to generate and visualize predictions and the underlying attention mechanisms.
To run it, you will need to modify the following line:
```
checkpoint = torch.load('/Users/marcolorenz/Programming/DETR/hgp_detr/checkpoints/nqueries20_resnet18/checkpoint0399.pth', map_location='cpu')  # Use 'cuda' if using GPU
```
Please include the path to your own pretrained model obtained in the previous step, and play around with the sample images, or insert your own.


# Profiling

A working installation of CUDA and Nvidia Nsight Systems/Compute are prerequisites for profiling successfully.
Nvidia Nsight Compute in particular is necessary to produce roofline charts for Nvidia GPUs.
Most experiments associated with this repository were conducted on NERSC-9, Perlmutter. 
First, we will show the manual commands to conduct experiments on Perlmutter. Second, we will point to scripts to automate these experiments.
All commands and scripts can be adjusted to different accounts, systems or GPUs.

Happy hacking!

## Training and Profiling on NERSC-9, Perlmutter
The following contains sample commands for Perlmutter.

### initial:
Please note: A NERSC account is necessary to access Perlmutter. For details on NERSC and Perlmutter see [Getting started at NERSC](https://docs.nersc.gov/getting-started/).
```
ssh username@saul.nersc.gov
conda create -n "detr_12.2" python cython pycocotools pytorch torchvision pytorch scipy conda-forge::nvtx -c pytorch -c nvidia
git clone https://github.com/lorenz369/hgp_detr.git
```

### recurring:
```
module load conda
conda activate detr_12.2
cd /global/your/path/to/hgp_detr
```

### Profiling of 1 GPU
```
salloc --nodes 1 --gpus=1 --qos debug --time 00:20:00 --constraint gpu --account=myAccount
cd /global/your/path/to/hgp_detr
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=12345
dcgmi profile --pause
```

Nsight Systems
```
srun nsys profile --stats=true -t nvtx,cuda --output=../gpu_reports/perlmutter/GPU1/nsys/__report_name__ --force-overwrite true python main.py --epochs 1  --backbone resnet18 --dataset_file hgp
| tee your/path/log.txt 
```

Nsight Compute
```
ncu --target-processes all  -k regex:elementwise --launch-skip 10 --launch-count 10 --set default --section SourceCounters --metrics smsp__cycles_active.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.avg --export=your/path/file python main.py --epochs 1 --dataset_file hgp | tee your/path/log.txt

ncu  -k regex:elementwise --launch-skip 10 --launch-count 10 --set default --section SourceCounters --metrics smsp__cycles_active.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.avg --export=your/path/file python main.py --epochs 1 --dataset_file hgp | tee your/path/log.txt
``` 

### Profiling of 2 GPUs
```
salloc --nodes 1 --gpus=2 --qos debug --time 00:15:00 --constraint gpu --account=myAccount
cd /global/your/path/to/hgp_detr
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=12345
dcgmi profile --pause
```

Nsight Systems
```
srun nsys profile --stats=true -t nvtx,cuda --output=your/path/file --force-overwrite true python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --epochs 1 --dataset_file hgp
| tee tee your/path/log.txt
```

Nsight Compute
```
srun ncu --export=your/path/file--set default --section SourceCounters --metrics smsp__cycles_active.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.avg python main.py -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --epochs 1 --dataset_file hgp
| tee tee your/path/log.txt    
```

### Output Sync
```
rsync -avz username@saul.nersc.gov:/global/homes/u/username/dir /Users/marcolorenz/Programming/DETR/gpu_reports/perlmutter

```

## Roofline Charts
'roofline-on-nvidia-gpus/custom-scripts' contains python scripts for producing roofline charts with csv output files obtained from profiling runs with Nvidia Nsight Compute.
'Postprocess.py' contains a parsing scripts from csv data to a Pandas DataFrame. To produce roofline charts, it will then hand these DataFrames to roofline function. 
There are two predefined functions available: 'roofline.py' for basic roofline charts, and 'roofline_pu.py' for more information processing units. They can also be used as a starting point for own function definitions.


To run simple execute 'postprocess.py' in a directory containing one or multiple files that read 'output....csv', or adjust the script to meet your own requirements.

Adjust the following line to process different directories:
```
datadir="."
```

Adjust the following lines to produce different types of roofline charts, or to call your own function:
```
from roofline_pu import roofline_pu

roofline_pu(title, FLOPS, AI, AIHBM, AIL2, AIL1, LABELS, PU, flag)
```

## Automating Roofline Profiling

'roofline-on-nvidia-gpus/custom-scripts' furthermore contains a set of scripts to profile and compare different aspects of DETR like hyperparameters, or sections of the training loop.
These scripts are designed for a 'slurm' scheduled system like Perlmutter with a preconfigured conda environment.

At the very least, you will need to modify the first lines specifying the slurm parameters, particularly the account and the output directory:
```
#!/bin/bash -l
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --qos debug
#SBATCH --time=00:30:00
#SBATCH --account=m3930
#SBATCH --output=/global/homes/m/marcolz/DETR/gpu_reports/GPU1/slurm/slurm_%j.out
```

Next, run with:
```
sbatch myscript.sh
```

To watch execution, optionally run
```
watch -n 3 sqs
```





















## Training on Octane (Private cluster of Heidelberg University, not externally accessible)

### initial:
```
ssh octane
git clone https://github.com/lorenz369/hgp_detr.git
```

### conda for brook/dev nodes
```
conda create --name detr_clone --clone base #for brook and dev
conda activate detr_clone
conda install conda-forge::pycocotools
```

### Profiling, e.g. dev
```
srun -p dev -N 1 --gres=gpu:1 --cpus-per-task 1 --mem 4G --pty bash -i
module load anaconda/3
module load cuda/11.4
conda activate detr_clone
cd /home/mlorenz/hgp_detr/
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=12345
nsys profile -o /home/mlorenz/octane/dev/nsys/__report_name__ --stats=true -t nvtx,cuda --force-overwrite true python main.py --batch_size 2 --epochs 3  --backbone resnet18 --enc_layers 1 --dec_layers 1 --dim_feedforward 512 --hidden_dim 64 --nheads 2 --num_queries 5 --num_workers 1 --dataset_file hgp 
| tee /home/mlorenz/octane/dev/txt/output.txt
```

### Output Sync
```
rsync -avz mlorenz@ceg-octane:/home/mlorenz/octane /Users/marcolorenz/Programming/DETR/gpu_reports
```