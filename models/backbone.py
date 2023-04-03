# ------------------------------------------------------------------------
# PoET: Pose Estimation Transformer for Single-View, Multi-Object 6D Pose Estimation
# Copyright (c) 2022 Thomas Jantos (thomas.jantos@aau.at), University of Klagenfurt - Control of Networked Systems (CNS). All Rights Reserved.
# Licensed under the BSD-2-Clause-License with no commercial use [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE_DEFORMABLE_DETR for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
Backbone modules.
"""

import torch
from ultralytics import CNS_YOLO, YOLO
from torch import nn
from typing import List
from util.misc import NestedTensor

import torch.nn.functional as F
from .position_encoding import build_position_encoding
from .backbone_maskrcnn import build_maskrcnn
# from .yolov4.cns_yolo import build_cns_yolo


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        if hasattr(backbone, 'model'):
            self.num_channels = backbone.model.num_channels
        else:
            self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        # TODO: Dirty, fix it
        # TODO: Currently the Object detector backbone has to be pretrained. Extend code to make object detectors
        #  trainable.
        if hasattr(self[0], 'train_backbone') and self[0].train_backbone:
            raise NotImplementedError
        else:
            self[0].eval()
            preds = self[0](tensor_list)
            if type(self[0]).__name__ == 'YOLO':
                results, predictions, xs = preds
                for prediction in predictions:
                    prediction = torch.clone(prediction)
                    if prediction is not None:
                        prediction[:, 5] += 1
                # out: Dict[str, NestedTensor] = {} # replaced by xs
                for name, x in xs.items():
                    m = tensor_list.mask
                    # print(f'{name=},{x=}')
                    if isinstance(x, tuple):
                        x = x[1][1]     # chosen kinda arbitrarily, if it doens't work, change this
                    assert m is not None
                    mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
                    xs[name] = NestedTensor(torch.clone(x), mask)
            else:
                self[0].eval()
                predictions, xs = self[0](tensor_list)
                
        out: List[NestedTensor] = []
        pos = []
        for name, x in sorted(xs.items()):
            out.append(x)

        # position encoding
        for x in out:
            # pos.append(self[1](x).to(x.tensors.dtype))
            pos.append(self[1](x))

        return out, pos, predictions

def build_yolov8(args):
    if not args.backbone_weights:
        backbone_weights = '/home/nils/ultralytics/runs/detect/train7/weights/best.pt' 
    else:
        backbone_weights = args.backbone_weights
    # backbone = CNS_YOLO(opts=args, model=backbone_weights, task='cns_detect', return_interim_layers=args.num_feature_levels>1).backbone
    backbone = YOLO(opts=args, model=backbone_weights, task='cns_detect', return_interim_layers=args.num_feature_levels>1)# .backbone
    backbone.strides = backbone.model.strides
    return backbone


def build_backbone(args):
    # Build the positional embedding
    position_embedding = build_position_encoding(args)

    # print(args.backbone)
    # Build the object detector backbone
    if args.backbone == "maskrcnn":
        backbone = build_maskrcnn(args)
    elif args.backbone == 'yolov4':
        # backbone = build_cns_yolo(args)
        raise NotImplementedError
    elif args.backbone == 'yolov8':
        backbone = build_yolov8(args)
    else:
        raise NotImplementedError
    model = Joiner(backbone, position_embedding)
    return model
