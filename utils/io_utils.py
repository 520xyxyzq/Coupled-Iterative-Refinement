# Input/output utility functions
# Author: Ziqi Lu ziqilu@mit.edu
# Copyright 2023 The Ambitious Folks of the MRG

import json
import os
import sys
import torch
import pandas as pd
from pathlib import Path
import numpy as np

sys.path.append(str(Path(".").resolve()))
from detector.tensor_collection import PandasTensorCollection


def rle_to_binary_mask(rle):
    """
    Converts a COCOs run-length encoding (RLE) to binary mask.
    @param rle (dict): Mask in RLE format
    @return mask (HxW): a 2D binary numpy array where '1's represent the object
    """
    binary_array = np.zeros(np.prod(rle.get('size')), dtype=bool)
    counts = rle.get('counts')

    start = 0
    for i in range(len(counts)-1):
        start += counts[i] 
        end = start + counts[i+1] 
        binary_array[start:end] = (i + 1) % 2

    binary_mask = binary_array.reshape(*rle.get('size'), order='F')
    return binary_mask


def readDetections(json_file):
    """
    Read 2D detections from file
    @param json_file (str): Path to the detection json file
    @return dets (PandasTensorCollect): 2D Bboxes + 2D Segmentations + labels 
    """
    assert os.path.isfile(json_file), f"Error: {json_file} doesn't exist"
    with open(json_file) as jf:
        dets_lst = json.load(jf)

    bboxes, masks, dets = {}, {}, {}
    for det in dets_lst:
        image_id = det["image_id"]
        obj_id = det["category_id"]
        label = f"obj_{obj_id:06g}"
        score = det["score"]
        det_dict = dict(batch_im_id=0, label=label, score=score)

        bbox = det["bbox"]
        # TODO: COCO [x1, y1, h, w] -> torchvision bbox [x1, y1, x2, y2]
        bbox[-2] += bbox[0]
        bbox[-1] += bbox[1]
        mask_rle = det["segmentation"]
        mask = rle_to_binary_mask(mask_rle)
        # Organize the detections by Image ID
        if dets.get(image_id, False):
            dets[image_id].append(det_dict)
            bboxes[image_id].append(torch.tensor(bbox))
            masks[image_id].append(torch.tensor(mask))
        else:
            dets[image_id] = [det_dict]
            bboxes[image_id] = [torch.tensor(bbox)]
            masks[image_id] = [torch.tensor(mask)]

    for image_id in dets.keys():
        masks[image_id] = torch.stack(masks[image_id])
        bboxes[image_id] = torch.stack(bboxes[image_id])
        # TODO: consider the case where masks don't exist in json
        dets[image_id] = PandasTensorCollection(
            infos=pd.DataFrame(dets[image_id]), bboxes=bboxes[image_id],
            masks=masks[image_id]
        )
        # The detections have to be processed in GPU
        dets[image_id] = dets[image_id].cuda()
    return dets


def returnEmptyDetection(height=None, width=None, has_mask=False):
    """
    Create an empty detection PandasTensorCollection
    @param height (int): Image height (for mask)
    @param width (int): Image width (for mask)
    @param has_mask (bool): Whether to create a mask tensor
    @return det (PandasTensorCollection): Empty detection
    """
    if has_mask:
        assert height is not None and width is not None,\
            "Error: must have image shape to make masks!"
    infos = dict(score=[], label=[], batch_im_id=[])
    bboxes = torch.empty(0, 4).cuda().float()
    masks = torch.empty(0, height, width, dtype=torch.bool).cuda()

    outputs = PandasTensorCollection(
        infos=pd.DataFrame(infos), bboxes=bboxes, masks=masks
    )
    return outputs
        
