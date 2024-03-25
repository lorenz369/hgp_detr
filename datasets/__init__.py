# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision

from .coco import build as build_coco
from .coco_eval import CocoEvaluator
from pycocotools.coco import COCO

from .hgp import HGPDetection, build_hgp
from .hgp_eval import HGPEvaluator
from .hgp_ann import HGP


def get_coco_api_from_dataset(dataset):
    print("get_coco_api_from_dataset")
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        print("dataset", dataset)
        if isinstance(dataset, HGPDetection):
            return dataset.hgp
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco
    

def build_dataset(image_set, args):
    if args.dataset_file == 'hgp':
        return build_hgp(image_set)
    if args.dataset_file == 'coco':
        return build_coco(image_set, args)
    if args.dataset_file == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')

def build_evaluator(base_ds, iou_types):
    print("base_ds", base_ds)
    print("isinstance(base_ds, HGP)", isinstance(base_ds, HGP))
    print("isinstance(base_ds, torchvision.datasets.CocoDetection)", isinstance(base_ds, torchvision.datasets.CocoDetection))
    print("type(base_ds)", type(base_ds))  
    if isinstance(base_ds, HGP):
        print("HGPEvaluator")
        return HGPEvaluator(base_ds, iou_types)
    if isinstance(base_ds, torchvision.datasets.CocoDetection):
        print("CocoEvaluator")
        return CocoEvaluator(base_ds, iou_types)
    raise NotImplementedError(f"no evaluator for {base_ds}")
