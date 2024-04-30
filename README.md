--------------------------------------------------------------------------------
Modified by Marco Lorenz in April 2024.
I've forked https://github.com/facebookresearch/detr to apply the detection transformer 
to the Hands, Guns and Phones dataset (https://paperswithcode.com/dataset/hgp), and to 
examine and profile its execution.
All modifications are made under the terms of the Apache License 2.0, which is the license
originally associated with this file and repository. All original copyright, patent, 
trademark, and attribution notices from the Source form of the Work have been retained, 
excluding those notices that do not pertain to any part of the Derivative Works.
--------------------------------------------------------------------------------

# Usage - Object detection
There are no extra compiled components in DETR and package dependencies are minimal,
so the code is very simple to use. We provide instructions how to install dependencies via conda.
First, clone the repository locally:
```
git clone https://github.com/facebookresearch/detr.git
```
Then, install PyTorch 1.5+ and torchvision 0.6+:
```
conda install -c pytorch pytorch torchvision
```
Install pycocotools (for evaluation on COCO), nvtx (for annotations) and scipy (for training):
```
conda install cython scipy
conda install conda-forge::nvtx

pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```
That's it, should be good to train and evaluate detection models.

To train a lightweight example configuration on the HGP dataset (GPU):
```
python main.py --batch_size 2 --epochs 3  --backbone resnet18 --enc_layers 1 --dec_layers 1 --dim_feedforward 512 --hidden_dim 64 --nheads 2 --num_queries 5 --dataset_file hgp
```

To train a lightweight example configuration on the HGP dataset (CPU):
```
python main.py --batch_size 2 --epochs 3  --backbone resnet18 --enc_layers 1 --dec_layers 1 --dim_feedforward 512 --hidden_dim 64 --nheads 2 --num_queries 5 --device cpu --dataset_file hgp
```

# Argument Help

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


# Perlmutter

## initial:
```
ssh marcolz@saul.nersc.gov
conda create -n "detr_12.2" python cython pycocotools pytorch torchvision pytorch scipy conda-forge::nvtx -c pytorch -c nvidia
git clone https://github.com/lorenz369/hgp_detr.git
git clone https://github.com/lorenz369/gpu_reports.git
```

## recurring:
```
module load conda
conda activate detr_12.2
cd /global/homes/m/marcolz/DETR/hgp_detr
```

## Profiling of 1 GPU
```
salloc --nodes 1 --gpus=1 --qos debug --time 00:15:00 --constraint gpu --account=m3930
cd /global/homes/m/marcolz/DETR/hgp_detr
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=12345
dcgmi profile -–pause
```

Nsight Systems
```
srun nsys profile --stats=true -t nvtx,cuda --output=../gpu_reports/perlmutter/GPU1/nsys/__report_name__ --force-overwrite true python main.py --epochs 1  --backbone resnet18 --enc_layers 1 --dec_layers 1 --dim_feedforward 512 --hidden_dim 64 --nheads 2 --num_queries 5 --dataset_file hgp
| tee ../output/perlmutter_1gpu.txt  
```

Nsight Compute
```
ncu --target-processes all --nvtx --nvtx-include rng --range-filter :0:[5] -k regex:elementwise --launch-skip 10 --launch-count 10 --set default --section SourceCounters --metrics smsp__cycles_active.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.avg --export=/global/homes/m/marcolz/DETR/gpu_reports/GPU1/ncu/perlmutter1 python main.py --epochs 1  --backbone resnet18 --enc_layers 1 --dec_layers 1 --dim_feedforward 512 --hidden_dim 64 --nheads 2 --num_queries 5 --dataset_file hgp | tee /global/homes/m/marcolz/DETR/gpu_reports/GPU1/perlmutter1.txt

ncu --nvtx --nvtx-include rng --range-filter :0:[5] -k regex:elementwise --launch-skip 10 --launch-count 10 --set default --section SourceCounters --metrics smsp__cycles_active.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.avg --export=/home/coder/coder/ncu/__report_name__ python main.py --epochs 1  --backbone resnet18 --enc_layers 1 --dec_layers 1 --dim_feedforward 512 --hidden_dim 64 --nheads 2 --num_queries 5 --dataset_file hgp | tee /home/coder/coder/txt/coder.txt
```


 

## Profiling of 2 GPUs
```
salloc --nodes 1 --gpus=2 --qos interactive --time 00:15:00 --constraint gpu --account=m3930
cd /global/homes/m/marcolz/DETR/hgp_detr
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=12345
dcgmi profile -–pause
```

Nsight Systems
```
srun nsys profile --stats=true -t nvtx,cuda --output=../gpu_reports/perlmutter/GPU2/nsys/__report_name__ --force-overwrite true python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --epochs 1  --backbone resnet18 --enc_layers 1 --dec_layers 1 --dim_feedforward 512 --hidden_dim 64 --nheads 2 --num_queries 5 --dataset_file hgp
| tee ../output/perlmutter_2gpu.txt  
```

Nsight Compute
```
srun ncu --nvtx --nvtx-include --kernel-id :::1 --export=../gpu_reports/perlmutter/GPU2/ncu/__report_name__ --set default --section SourceCounters --metrics smsp__cycles_active.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.avg python main.py -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --epochs 1  --backbone resnet18 --enc_layers 1 --dec_layers 1 --dim_feedforward 512 --hidden_dim 64 --nheads 2 --num_queries 5 --dataset_file hgp
| tee ../output/perlmutter_2gpu.txt      
```

## Output Sync
```
rsync -avz marcolz@saul.nersc.gov:/global/homes/m/marcolz/DETR/gpu_reports/perlmutter /Users/marcolorenz/Programming/DETR/gpu_reports
```
Nsight Systems:
  nsys-ui report_name.nsys-rep

# Experimental: Coder

## initial
ssh coder.DETR.main
conda install cuda -c nvidia
conda install nvtx pycocotools -c conda-forge

git clone https://github.com/lorenz369/hgp_detr.git

## Find the correct ncu installation path sing the find or locate Command
If you're unsure of the installation path, you can use the find or locate command to search for ncu across your system. Try out all the options to find the correct one and configure the environment in the next step.

Using find:
```
sudo find / -name ncu 2>/dev/null
```

## Set the Correct `ncu` Path Permanently

Since you've identified the correct `ncu` executable for your system, you should configure your environment to use this path by default.

### Update `.bashrc` or `.bash_profile`

1. **Open Your Configuration File**: Open your `.bashrc` or `.bash_profile` in a text editor. You can use `nano` for a simple text editing experience:

    ```bash
    nano ~/.bashrc  # or ~/.bash_profile
    ```

2. **Add the `ncu` Path**: Add the following line at the end of the file to update your `PATH` environment variable:

    ```bash
    export PATH="/opt/miniconda/envs/DL/nsight-compute/2024.1.1/target/linux-desktop-glibc_2_11_3-x64:$PATH"
    ```

3. **Save and Exit**: Save your changes and exit the editor. If you are using `nano`, you can press `Ctrl + O` to write the changes, press `Enter` to confirm, and then `Ctrl + X` to exit.

4. **Activate the Changes**: To make the changes take effect, source your updated configuration file:

    ```bash
    source ~/.bashrc  # or source ~/.bash_profile
    ```

This setup ensures that the correct version of `ncu` is available system-wide in any new terminal session.

## Profiling
```
cd /home/coder/hgp_detr
```

Nsight Compute
```
ncu --range-filter :0:[5] --nvtx --nvtx-include rng -k regex:elementwise --launch-count 10 --set default --section SourceCounters --metrics smsp__cycles_active.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.avg --export=/home/coder/coder/ncu/__report_name__ python main.py --epochs 1  --backbone resnet18 --enc_layers 1 --dec_layers 1 --dim_feedforward 512 --hidden_dim 64 --nheads 2 --num_queries 5 --dataset_file hgp | tee /home/coder/coder/txt/coder.txt      
```

## Output Sync
```
rsync -avz /home/coder/coder /Users/marcolorenz/Programming/DETR/gpu_reports
```

# Octane

## initial:
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

## Profiling, e.g. dev
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

## Output Sync
```
rsync -avz mlorenz@ceg-octane:/home/mlorenz/octane /Users/marcolorenz/Programming/DETR/gpu_reports
```
    

# Helpful commands 
tee, screen or tmux
    


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
# Model Zoo
We provide baseline DETR and DETR-DC5 models, and plan to include more in future.
AP is computed on COCO 2017 val5k, and inference time is over the first 100 val5k COCO images,
with torchscript transformer.

<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>backbone</th>
      <th>schedule</th>
      <th>inf_time</th>
      <th>box AP</th>
      <th>url</th>
      <th>size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DETR</td>
      <td>R50</td>
      <td>500</td>
      <td>0.036</td>
      <td>42.0</td>
      <td><a href="https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detr/logs/detr-r50_log.txt">logs</a></td>
      <td>159Mb</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DETR-DC5</td>
      <td>R50</td>
      <td>500</td>
      <td>0.083</td>
      <td>43.3</td>
      <td><a href="https://dl.fbaipublicfiles.com/detr/detr-r50-dc5-f0fb7ef5.pth">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detr/logs/detr-r50-dc5_log.txt">logs</a></td>
      <td>159Mb</td>
    </tr>
    <tr>
      <th>2</th>
      <td>DETR</td>
      <td>R101</td>
      <td>500</td>
      <td>0.050</td>
      <td>43.5</td>
      <td><a href="https://dl.fbaipublicfiles.com/detr/detr-r101-2c7b67e5.pth">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detr/logs/detr-r101_log.txt">logs</a></td>
      <td>232Mb</td>
    </tr>
    <tr>
      <th>3</th>
      <td>DETR-DC5</td>
      <td>R101</td>
      <td>500</td>
      <td>0.097</td>
      <td>44.9</td>
      <td><a href="https://dl.fbaipublicfiles.com/detr/detr-r101-dc5-a2e86def.pth">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detr/logs/detr-r101-dc5_log.txt">logs</a></td>
      <td>232Mb</td>
    </tr>
  </tbody>
</table>

COCO val5k evaluation results can be found in this [gist](https://gist.github.com/szagoruyko/9c9ebb8455610958f7deaa27845d7918).

The models are also available via torch hub,
to load DETR R50 with pretrained weights simply do:
```python
model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)
```


COCO panoptic val5k models:
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>backbone</th>
      <th>box AP</th>
      <th>segm AP</th>
      <th>PQ</th>
      <th>url</th>
      <th>size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DETR</td>
      <td>R50</td>
      <td>38.8</td>
      <td>31.1</td>
      <td>43.4</td>
      <td><a href="https://dl.fbaipublicfiles.com/detr/detr-r50-panoptic-00ce5173.pth">download</a></td>
      <td>165Mb</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DETR-DC5</td>
      <td>R50</td>
      <td>40.2</td>
      <td>31.9</td>
      <td>44.6</td>
      <td><a href="https://dl.fbaipublicfiles.com/detr/detr-r50-dc5-panoptic-da08f1b1.pth">download</a></td>
      <td>165Mb</td>
    </tr>
    <tr>
      <th>2</th>
      <td>DETR</td>
      <td>R101</td>
      <td>40.1</td>
      <td>33</td>
      <td>45.1</td>
      <td><a href="https://dl.fbaipublicfiles.com/detr/detr-r101-panoptic-40021d53.pth">download</a></td>
      <td>237Mb</td>
    </tr>
  </tbody>
</table>

Checkout our [panoptic colab](https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/DETR_panoptic.ipynb)
to see how to use and visualize DETR's panoptic segmentation prediction.

# Notebooks

We provide a few notebooks in colab to help you get a grasp on DETR:
* [DETR's hands on Colab Notebook](https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_attention.ipynb): Shows how to load a model from hub, generate predictions, then visualize the attention of the model (similar to the figures of the paper)
* [Standalone Colab Notebook](https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_demo.ipynb): In this notebook, we demonstrate how to implement a simplified version of DETR from the grounds up in 50 lines of Python, then visualize the predictions. It is a good starting point if you want to gain better understanding the architecture and poke around before diving in the codebase.
* [Panoptic Colab Notebook](https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/DETR_panoptic.ipynb): Demonstrates how to use DETR for panoptic segmentation and plot the predictions.


## Data preparation

Download and extract COCO 2017 train and val images with annotations from
[http://cocodataset.org](http://cocodataset.org/#download).
We expect the directory structure to be the following:
```
path/to/coco/
  annotations/  # annotation json files
  train2017/    # train images
  val2017/      # val images
```

## Training
To train baseline DETR on a single node with 8 gpus for 300 epochs run:
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --coco_path /path/to/coco 
```
A single epoch takes 28 minutes, so 300 epoch training
takes around 6 days on a single machine with 8 V100 cards.
To ease reproduction of our results we provide
[results and training logs](https://gist.github.com/szagoruyko/b4c3b2c3627294fc369b899987385a3f)
for 150 epoch schedule (3 days on a single machine), achieving 39.5/60.3 AP/AP50.

We train DETR with AdamW setting learning rate in the transformer to 1e-4 and 1e-5 in the backbone.
Horizontal flips, scales and crops are used for augmentation.
Images are rescaled to have min size 800 and max size 1333.
The transformer is trained with dropout of 0.1, and the whole model is trained with grad clip of 0.1.


## Evaluation
To evaluate DETR R50 on COCO val5k with a single GPU run:
```
python main.py --batch_size 2 --no_aux_loss --eval --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth --coco_path /path/to/coco
```
We provide results for all DETR detection models in this
[gist](https://gist.github.com/szagoruyko/9c9ebb8455610958f7deaa27845d7918).
Note that numbers vary depending on batch size (number of images) per GPU.
Non-DC5 models were trained with batch size 2, and DC5 with 1,
so DC5 models show a significant drop in AP if evaluated with more
than 1 image per GPU.

## Multinode training
Distributed training is available via Slurm and [submitit](https://github.com/facebookincubator/submitit):
```
pip install submitit
```
Train baseline DETR-6-6 model on 4 nodes for 300 epochs:
```
python run_with_submitit.py --timeout 3000 --coco_path /path/to/coco
```

# Usage - Segmentation

We show that it is relatively straightforward to extend DETR to predict segmentation masks. We mainly demonstrate strong panoptic segmentation results.

## Data preparation

For panoptic segmentation, you need the panoptic annotations additionally to the coco dataset (see above for the coco dataset). You need to download and extract the [annotations](http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip).
We expect the directory structure to be the following:
```
path/to/coco_panoptic/
  annotations/  # annotation json files
  panoptic_train2017/    # train panoptic annotations
  panoptic_val2017/      # val panoptic annotations
```

## Training

We recommend training segmentation in two stages: first train DETR to detect all the boxes, and then train the segmentation head.
For panoptic segmentation, DETR must learn to detect boxes for both stuff and things classes. You can train it on a single node with 8 gpus for 300 epochs with:
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --coco_path /path/to/coco  --coco_panoptic_path /path/to/coco_panoptic --dataset_file coco_panoptic --output_dir /output/path/box_model
```
For instance segmentation, you can simply train a normal box model (or used a pre-trained one we provide).

Once you have a box model checkpoint, you need to freeze it, and train the segmentation head in isolation.
For panoptic segmentation you can train on a single node with 8 gpus for 25 epochs:
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --masks --epochs 25 --lr_drop 15 --coco_path /path/to/coco  --coco_panoptic_path /path/to/coco_panoptic  --dataset_file coco_panoptic --frozen_weights /output/path/box_model/checkpoint.pth --output_dir /output/path/segm_model
```
For instance segmentation only, simply remove the `dataset_file` and `coco_panoptic_path` arguments from the above command line.

# License
DETR is released under the Apache 2.0 license. Please see the [LICENSE](LICENSE) file for more information.

# Contributing
We actively welcome your pull requests! Please see [CONTRIBUTING.md](.github/CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](.github/CODE_OF_CONDUCT.md) for more info.
