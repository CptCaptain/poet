# ------------------------------------------------------------------------
# PoET: Pose Estimation Transformer for Single-View, Multi-Object 6D Pose Estimation
# Copyright (c) 2022 Thomas Jantos (thomas.jantos@aau.at), University of Klagenfurt - Control of Networked Systems (CNS). All Rights Reserved.
# Licensed under the BSD-2-Clause-License with no commercial use [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE_DEFORMABLE_DETR in the LICENSES folder for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import util.misc as utils
import data_utils.samplers as samplers
from data_utils import build_dataset
from models import build_model
from evaluation_tools.pose_evaluator_init import build_pose_evaluator

from util.quaternion_ops import quat2rot
from data_utils.data_prefetcher import data_prefetcher

from evaluation_tools.metrics import get_src_permutation_idx, calc_rotation_error, calc_translation_error


def get_args_parser():
    parser = argparse.ArgumentParser('Pose Estimation Transformer', add_help=False)

    parser.add_argument('--checkpoint', help='Path to model checkpoint', required=True)
    parser.add_argument('--device', default='cuda',
                        help='device to use for inference')
    # * Backbone
    parser.add_argument('--backbone', default='yolov4', type=str, choices=['yolov4', 'maskrcnn'],
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--backbone_cfg', default='configs/ycbv_yolov4-csp.cfg', type=str,
                        help="Path to the backbone config file to use")
    parser.add_argument('--backbone_weights', default=None, type=str,
                        help="Path to the pretrained weights for the backbone."
                             "None if no weights should be loaded.")
    parser.add_argument('--backbone_conf_thresh', default=0.4, type=float,
                        help="Backbone confidence threshold which objects to keep.")
    parser.add_argument('--backbone_iou_thresh', default=0.5, type=float, help="Backbone IOU threshold for NMS")
    parser.add_argument('--backbone_agnostic_nms', action='store_true',
                        help="Whether backbone NMS should be performed class-agnostic")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # ** PoET configs
    parser.add_argument('--bbox_mode', default='gt', type=str, choices=('gt', 'backbone', 'jitter'),
                        help='Defines which bounding boxes should be used for PoET to determine query embeddings.')
    parser.add_argument('--reference_points', default='bbox', type=str, choices=('bbox', 'learned'),
                        help='Defines whether the transformer reference points are learned or extracted from the bounding boxes')
    parser.add_argument('--query_embedding', default='bbox', type=str, choices=('bbox', 'learned'),
                        help='Defines whether the transformer query embeddings are learned or determined by the bounding boxes')
    parser.add_argument('--rotation_representation', default='6d', type=str, choices=('6d', 'quat', 'silho_quat'),
                        help="Determine the rotation representation with which PoET is trained.")
    parser.add_argument('--class_mode', default='agnostic', type=str, choices=('agnostic', 'specific'),
                        help="Determine whether PoET ist trained class-specific or class-agnostic")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=10, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # * Matcher
    parser.add_argument('--matcher_type', default='pose', choices=['pose'], type=str)
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=1, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Loss coefficients
    # Pose Estimation losses
    parser.add_argument('--translation_loss_coef', default=1, type=float, help='Loss weighing parameter for the translation')
    parser.add_argument('--rotation_loss_coef', default=1, type=float, help='Loss weighing parameter for the rotation')


    # * Inference mode settings
    parser.add_argument('--visualize', type='store_true', help='Visualize results')
    parser.add_argument('--return_results', type='store_true', help='Return results for further use')
    return parser


def init_model_from_checkpoint(args, model, checkpoint):
    # TODO maybe strip training related stuff...
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
    unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
    if len(missing_keys) > 0:
        print('Missing Keys: {}'.format(missing_keys))
    if len(unexpected_keys) > 0:
        print('Unexpected Keys: {}'.format(unexpected_keys))
    return model



def main(args):
    device = torch.device(args.device)

    # Build the model and evaluator
    model, criterion, matcher = build_model(args)
    model.to(device)

    checkpoint = torch.load(args.checkpoint)
    model = init_model_from_checkpoint(args, model, checkpoint)


    if args.visualize:
        pass
    elif args.return_results:
        pass
    return


def write_to_csv(matcher, n_boxes_per_sample, out_csv_file, outputs, pred_end_time, rotation_mode):
    outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

    for idx in len(outputs_without_aux["pred_rotation"]):
        pred_translations = outputs_without_aux["pred_translation"][idx].detach().cpu().numpy()
        pred_rotations = outputs_without_aux["pred_rotation"][idx].detach().cpu().numpy()
        pred_boxes = outputs_without_aux["pred_boxes"][idx].detach().cpu().numpy()
        pred_classes = outputs_without_aux["pred_classes"][idx].detach().cpu().numpy()

        if rotation_mode in ['quat', 'silho_quat']:
            pred_rotations = quat2rot(pred_rotations)

        for cls_idx, img_file, pred_translation, pred_rotation in zip(pred_classes, pred_translations, pred_rotations):
            obj_id = cls_idx
            score = 1.0
            # CSV format: obj_id, score, R, t, time
            csv_str = "{},{},{} {} {} {} {} {} {} {} {}, {} {} {}, {}\n".format(obj_id, score,
                                                                                pred_rotation[0, 0], pred_rotation[0, 1], pred_rotation[0, 2],
                                                                                pred_rotation[1, 0], pred_rotation[1, 1], pred_rotation[1, 2],
                                                                                pred_rotation[2, 0], pred_rotation[2, 1], pred_rotation[2, 2],
                                                                                pred_translation[0] * 1000, pred_translation[1] * 1000, pred_translation[2] * 1000,
                                                                                pred_end_time)
            out_csv_file.write(csv_str)



def write_to_bop_csv(matcher, n_boxes_per_sample, out_csv_file, outputs, pred_end_time, rotation_mode, data_loader, targets):
    # CSV format: scene_id, im_id, obj_id, score, R, t, time

    outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

    pred_translations = outputs_without_aux["pred_translation"][idx].detach().cpu().numpy()
    pred_rotations = outputs_without_aux["pred_rotation"][idx].detach().cpu().numpy()

    indices = matcher(outputs_without_aux, targets, n_boxes_per_sample)
    idx = get_src_permutation_idx(indices)

    obj_classes_idx = torch.cat([t['labels'][i] for t, (_, i) in zip(targets, indices)],
                                dim=0).detach().cpu().numpy()

    img_files = [data_loader.dataset.coco.loadImgs(t["image_id"].item())[0]['file_name'] for t, (_, i) in
                 zip(targets, indices) for _ in range(0, len(i))]

    if rotation_mode in ['quat', 'silho_quat']:
        pred_rotations = quat2rot(pred_rotations)

    for cls_idx, img_file, pred_translation, pred_rotation in zip(obj_classes_idx, img_files, pred_translations, pred_rotations):
        file_info = img_file.split("/")
        scene_id = int(file_info[1])
        img_id = int(file_info[3][:file_info[3].rfind(".")])
        obj_id = cls_idx
        score = 1.0
        csv_str = "{},{},{},{},{} {} {} {} {} {} {} {} {}, {} {} {}, {}\n".format(scene_id, img_id, obj_id, score,
                                                                                  pred_rotation[0, 0], pred_rotation[0, 1], pred_rotation[0, 2],
                                                                                  pred_rotation[1, 0], pred_rotation[1, 1], pred_rotation[1, 2],
                                                                                  pred_rotation[2, 0], pred_rotation[2, 1], pred_rotation[2, 2],
                                                                                  pred_translation[0] * 1000, pred_translation[1] * 1000, pred_translation[2] * 1000,
                                                                                  pred_end_time)
        out_csv_file.write(csv_str)


@torch.no_grad()
def bop_evaluate(model, matcher, data_loader, image_set, bbox_mode, rotation_mode, device, output_dir):
    """
    Evaluate PoET on the dataset and store the results in the BOP format
    """
    model.eval()
    matcher.eval()

    output_eval_dir = output_dir + "/bop_" + bbox_mode + "/"
    Path(output_eval_dir).mkdir(parents=True, exist_ok=True)

    counter = 1
    n_images = len(data_loader.dataset.ids)
    # CSV format: scene_id, im_id, obj_id, score, R, t, time
    out_csv_file = open(output_eval_dir + 'ycbv.csv', 'w')
    out_csv_file.write("scene_id,im_id,obj_id,score,R,t,time")
    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        pred_start_time = time.time()
        outputs, n_boxes_per_sample = model(samples, targets)
        pred_end_time = time.time() - pred_start_time
        write_to_bop_csv(matcher, n_boxes_per_sample, out_csv_file, outputs, pred_end_time, rotation_mode, data_loader, targets)       

        print("Processed {}/{}".format(counter, n_images))
        counter += 1
    out_csv_file.close()


def predict_image(model, bbox_mode, rotation_mode, device, output_dir, sample):
        # do this when evaluating images as a precursor to RBOT
        pred_start_time = time.time()
        outputs, n_boxes_per_sample = model(sample)
        pred_end_time = time.time() - pred_start_time
        # CSV format: obj_id, score, R, t, time
        out_csv_file = open(output_eval_dir + 'sample.csv', 'w')
        out_csv_file.write("obj_id,score,R,t,time")
        write_to_csv(n_boxes_per_sample, out_csv_file, outputs, pred_end_time, rotation_mode)       


if __name__ == '__main__':
    parser = argparse.ArgumentParser('PoET training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)



