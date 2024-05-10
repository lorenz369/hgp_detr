# --------------------------------------------------------------------------------
# Modified by Marco Lorenz in 2024.
# Changes made: 
# Added CuPy profiler to the training loop in line 65 to profile the backwards computation.
# Added support of the Hands, Guns and Phones dataset (HGP), including the following
# - import of the build_evaluator method to support the HGP dataset in line 24
# - call of the build_evaluator method to support the HGP dataset in line 88
# This modification is made under the terms of the Apache License 2.0, which is the license
# originally associated with this file. All original copyright, patent, trademark, and
# attribution notices from the Source form of the Work have been retained, excluding those 
# notices that do not pertain to any part of the Derivative Works.
# --------------------------------------------------------------------------------

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch

import util.misc as utils
from datasets.__init__ import build_evaluator # Added by Marco Lorenz on April 2nd, 2024
from datasets.panoptic_eval import PanopticEvaluator

import cupy.cuda.runtime # Added by Marco Lorenz on May 2nd, 2024
import time # Added by Marco Lorenz on April 2nd, 2024

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, profiling_section: str = None):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    # Time accumulators added by Marco Lorenz on April 2nd, 2024
    total_process_time = 0
    total_loss_time = 0
    total_backward_time = 0
    iterations = 0

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):

        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        start_time = time.time()  # Added by Marco Lorenz on April 2nd, 2024
        
        if profiling_section == 'forward' or profiling_section == 'all': # Added by Marco Lorenz on April 2nd, 2024
            cupy.cuda.runtime.profilerStart()
        outputs = model(samples)
        if profiling_section == 'forward': # Added by Marco Lorenz on April 2nd, 2024
            cupy.cuda.runtime.profilerStop()

        process_time = time.time() - start_time # Added by Marco Lorenz on April 2nd, 2024
        total_process_time += process_time # Added by Marco Lorenz on April 2nd, 2024

        if profiling_section == 'loss': # Added by Marco Lorenz on April 2nd, 2024
            cupy.cuda.runtime.profilerStart()

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if profiling_section == 'loss': # Added by Marco Lorenz on April 2nd, 2024
            cupy.cuda.runtime.profilerStop()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)
        
        loss_time = time.time() - start_time - process_time # Added by Marco Lorenz on April 2nd, 2024
        total_loss_time += loss_time # Added by Marco Lorenz on April 2nd, 2024

        optimizer.zero_grad()

        if profiling_section == 'backward': # Added by Marco Lorenz on April 2nd, 2024
            cupy.cuda.runtime.profilerStart()
        losses.backward()
        if profiling_section == 'backward':
            cupy.cuda.runtime.profilerStop() # Added by Marco Lorenz on April 2nd, 2024

        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        
        if profiling_section == 'optimizer': # Added by Marco Lorenz on April 2nd, 2024
            cupy.cuda.runtime.profilerStart()
        optimizer.step()
        if profiling_section == 'optimizer' or profiling_section == 'all': # Added by Marco Lorenz on April 2nd, 2024
            cupy.cuda.runtime.profilerStop()

        backward_time = time.time() - start_time - process_time - loss_time  # Added by Marco Lorenz on April 2nd, 2024
        total_backward_time += backward_time # Added by Marco Lorenz on April 2nd, 2024
        iterations += 1  # Increment iteration count

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # Average the accumulated times, added by Marco Lorenz on April 2nd, 2024
    avg_process_time = total_process_time / iterations
    avg_loss_time = total_loss_time / iterations
    avg_backward_time = total_backward_time / iterations

    print(f"Average processing time per iteration: {avg_process_time:.6f} seconds")
    print(f"Average loss computation time per iteration: {avg_loss_time:.6f} seconds")
    print(f"Average backward pass time per iteration: {avg_backward_time:.6f} seconds")

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, profiling_section: str = None):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = build_evaluator(base_ds, iou_types) # Modified by Marco Lorenz on April 2nd, 2024
    coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if profiling_section == 'forward' or profiling_section == 'all': # Added by Marco Lorenz on April 2nd, 2024
            cupy.cuda.runtime.profilerStart()
        outputs = model(samples)
        if profiling_section == 'forward' or profiling_section == 'all': # Added by Marco Lorenz on April 2nd, 2024
            cupy.cuda.runtime.profilerStop()
            
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator
