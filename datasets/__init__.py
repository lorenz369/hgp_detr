# --------------------------------------------------------------------------------
# Modified by Marco Lorenz on April 2nd, 2024.
# Changes made: Added support of the Hands, Guns and Phones dataset (HGP), including the following
# - Custom imports
# - Support of HGPDetection class by modifiying the get_coco_api_from_dataset(dataset)
# - Support of HGP dataset class by modifying the build_evaluator(base_ds, iou_types)
# - Support of HGP evaluator by introducing the method build_evaluator(base_ds, iou_types)
# Hands, Guns and Phones dataset: https://paperswithcode.com/dataset/hgp
# This modification is made under the terms of the Apache License 2.0, which is the license
# originally associated with this file. All original copyright, patent, trademark, and
# attribution notices from the Source form of the Work have been retained, excluding those 
# notices that do not pertain to any part of the Derivative Works.
# --------------------------------------------------------------------------------

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision

from .coco import build as build_coco

# Custom imports, added by Marco Lorenz on April 2nd, 2024
from .coco_eval import CocoEvaluator
from pycocotools.coco import COCO

from .hgp import HGPDetection, build_hgp
from .hgp_eval import HGPEvaluator
from .hgp_ann import HGP


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, HGPDetection): # Added by Marco Lorenz on April 2nd, 2024
            return dataset.hgp
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco
    
def build_dataset(image_set, args):
    if args.dataset_file == 'hgp': # Added by Marco Lorenz on April 2nd, 2024
        return build_hgp(image_set)
    if args.dataset_file == 'coco':
        return build_coco(image_set, args)
    if args.dataset_file == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')

# Added by Marco Lorenz on April 2nd, 2024
def build_evaluator(base_ds, iou_types): 
    if isinstance(base_ds, HGP): 
        return HGPEvaluator(base_ds, iou_types)
    if isinstance(base_ds, torchvision.datasets.CocoDetection):
        return CocoEvaluator(base_ds, iou_types)
    raise NotImplementedError(f"no evaluator for {base_ds}")
