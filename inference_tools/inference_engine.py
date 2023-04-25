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

import json
import torch
import time
import util.misc as utils

from data_utils.data_prefetcher import data_prefetcher
from models import build_model
from inference_tools.dataset import build_dataset
from torch.utils.data import DataLoader, SequentialSampler


def benchmark_model(model, samples, targets, num_iterations=100):
    model.eval()
    total_time = 0
    # warmup
    for _ in range(50):
        _, _ = model(samples, targets)

    for _ in range(num_iterations):
        start_time = time.time()
        with torch.no_grad():
            _, _ = model(samples, targets)
        total_time += time.time() - start_time
    avg_time = 1000 * total_time / num_iterations
    return avg_time


def inference(args, accelerator):
    """
    Script for Inference with PoET. The datalaoder loads all the images and then iterates over them. PoET processes each
    image and stores the detected objects and their poses in a JSON file. Currently, this script allows only batch sizes
    of 1.
    """
    device = torch.device(args.device)
    model, criterion, matcher = build_model(args)
    model.to(device)
    model.eval()

    # Load model weights
    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)

    # Construct dataloader that loads the images for inference
    dataset_inference = build_dataset(args)
    sampler_inference = SequentialSampler(dataset_inference)
    data_loader_inference = DataLoader(dataset_inference, 1, sampler=sampler_inference,
                                       drop_last=False, collate_fn=utils.collate_fn, num_workers=0,
                                       pin_memory=True)
    model, data_loader_inference = accelerator.prepare(model, data_loader_inference)

    prefetcher = data_prefetcher(data_loader_inference, device, prefetch=False)
    samples, targets = prefetcher.next()
    targets = None
    results = {}
    benchmark_results = []
    # Iterate over all images, perform pose estimation and store results.
    for i, idx in enumerate(range(len(data_loader_inference)-900)):
        print("Processing {}/{}".format(i, len(data_loader_inference) - 1))
        outputs, n_boxes_per_sample = model(samples, targets)

        if args.benchmark:
            avg_time = benchmark_model(model, samples, targets)
            benchmark_results.append(avg_time)
            print(f"Average inference time: {avg_time:.4f} milliseconds")

        # Iterate over all the detected predictions
        img_file = data_loader_inference.dataset.image_paths[i]
        img_id = img_file[img_file.find("_")+1:img_file.rfind(".")]
        results[img_id] = {}
        for d in range(n_boxes_per_sample[0]):
            pred_t = outputs['pred_translation'][0][d].detach().cpu().tolist()
            pred_rot = outputs['pred_rotation'][0][d].detach().cpu().tolist()
            pred_box = outputs['pred_boxes'][0][d].detach().cpu().tolist()
            pred_class = outputs['pred_classes'][0][d].detach().cpu().tolist()
            results[img_id][d] = {
                "t": pred_t,
                "rot": pred_rot,
                "box": pred_box,
                "class": pred_class
            }

        samples, targets = prefetcher.next()

    # Store the json-file
    out_file_name = "results.json"
    with open(args.inference_output + out_file_name, 'w') as out_file:
        json.dump(results, out_file)

    if args.benchmark:
        out_file_name = "benchmark.json"
        with open(args.inference_output + out_file_name, 'w') as out_file:
            json.dump({'avg_time': f'{sum(benchmark_results) / len(benchmark_results):.4f}'}, out_file)
    return

