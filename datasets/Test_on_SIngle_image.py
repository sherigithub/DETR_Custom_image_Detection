#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 20:28:56 2023

@author: sheri
"""

import json
import random
import os
from pathlib import Path

from PIL import Image
import matplotlib.pyplot as plt

import torch

import torchvision.transforms as T
#torch.set_grad_enabled(False);

class test_img_preparation():
    def __init__(self):
        
        # standard PyTorch mean-std input image normalization
        self.transform = T.Compose([
            T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    
        self.CLASSES = ['bus','car','motorbike','person','truck']

        # colors for visualization
        self.COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
                       [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]] 

    # for output bounding box post-processing

    def box_cxcywh_to_xyxy(self,x):
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    def rescale_bboxes(self,out_bbox, size):
        img_w, img_h = size
        b = self.box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b

    def plot_results(self,pil_img, prob, boxes):
        plt.figure(figsize=(16,10))
        plt.imshow(pil_img)
        ax = plt.gca()
        colors = self.COLORS * 100
        for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                       fill=False, color=c, linewidth=3))
            cl = p.argmax()
            text = f'{self.CLASSES[cl]}: {p[cl]:0.2f}'
            ax.text(xmin, ymin, text, fontsize=15,
                    bbox=dict(facecolor='yellow', alpha=0.5))
        plt.axis('off')
        plt.show()



    def test_sample_images(self, model,args):
    
        model.eval()
        root = Path(args.coco_path)
        
        fi = open(root/"annotations"/"valid_annotations_coco.json")
        test_data = json.load(fi)
        fi.close()
        
        test_imag_pths = test_data['images']
        ln_max = len(test_imag_pths)
        
        rnd_ls = random.sample(range(0, ln_max), 4)
    
        for i in rnd_ls:
            im_pth = test_imag_pths[i]['file_name']
            img_path = root/'valid'/im_pth
            im = Image.open(img_path)
                
            # mean-std normalize the input image (batch-size: 1)
            img = self.transform(im).unsqueeze(0)
            
            # propagate through the model
            outputs = model(img)
            
            # keep only predictions with 0.7+ confidence
            probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
            keep = probas.max(-1).values > 0.8 #args.threshold_value
            
            # convert boxes from [0; 1] to image scales
            bboxes_scaled = self.rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
            self.plot_results(im, probas[keep], bboxes_scaled)
            
        return 