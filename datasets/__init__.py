# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision

from .coco import build as build_coco
from .Test_on_SIngle_image import test_sample_images

def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco

def model_test_images(model,args):
    return test_sample_images(model,args)

def build_dataset(image_set, args):
    # if you want to use original dataset filw which original Facebook DETR team worked on this - detection task
    if args.dataset_file == 'coco':
        return build_coco(image_set, args)
    
    # if you want to use original dataset file which original Facebook DETR team worked on this - segmentatation task
    if args.dataset_file == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')