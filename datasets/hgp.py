# --------------------------------------------------------------------------------
# Modified by Marco Lorenz on April 2nd, 2024.
# Custom dataset for HGP dataset. Mostly copy-paste from coco.py
# Changes made: Support of the HGP dataset for the DETR model by introducing the HGPDetection class
# Hands, Guns and Phones dataset: https://paperswithcode.com/dataset/hgp
# This modification is made under the terms of the Apache License 2.0, which is the license
# originally associated with this file. All original copyright, patent, trademark, and
# attribution notices from the Source form of the Work have been retained, excluding those 
# notices that do not pertain to any part of the Derivative Works.
# --------------------------------------------------------------------------------

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
from torchvision.datasets import VisionDataset

from .hgp_ann import HGP

import datasets.transforms as T

from os import path
from PIL import Image
from typing import Any, List, Tuple, Dict


class HGPDetection(VisionDataset):
    def __init__(self, img_folder: str, lab_folder: str, ann_file: str, image_set: str, transforms):
        super().__init__(img_folder)
        self.hgp = HGP(ann_file)
        self.ids = list(sorted(self.hgp.imgs.keys()))
        self.img_folder = img_folder
        self.lab_folder = lab_folder
        self.ann_file = ann_file
        self.file_prefix = 'T' if image_set == 'train' else 'V'
        self._transforms = transforms
        self.prepare = PrepareFormat() 
    
    def _load_image(self, id: int) -> Image.Image:
        img_path = path.join(self.img_folder, f"{self.file_prefix}{id:08}.png")
        return Image.open(img_path).convert('RGB')

    def _load_target(self, id: int) -> List[Any]:
        return self.hgp.loadAnns(self.hgp.getAnnIds(id))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image_id = self.ids[index]
        image = self._load_image(index)
        target = self._load_target(index)

        target = {'image_id': image_id, 'annotations': target}
        image, target = self.prepare(image, target)
        if self._transforms is not None:
            image, target = self._transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.ids)
    
class PrepareFormat(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target

def make_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

def build_hgp(image_set):
    file_dir = Path(__file__).resolve().parent.parent
    root = Path(file_dir, 'HGP')
    mode = 'instances'
    print(f'Dataset-root specified: {root}')
    assert root.exists(), f'provided path {root} does not exist'
    PATHS = {
        "train": (root / "images" / "train2017", root / "labels" / "train2017", root / "images" / "annotations"/ f"{mode}_train2017.json"),
        "val": (root / "images" / "val2017", root / "labels" / "val2017", root / "images" / "annotations" / f"{mode}_val2017.json")
    }

    img_folder, lab_folder, ann_file = PATHS[image_set]
    dataset = HGPDetection(img_folder, lab_folder, ann_file, image_set, transforms=make_transforms(image_set))
    return dataset
