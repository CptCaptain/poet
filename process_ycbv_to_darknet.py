#!/usr/bin/env python

import os
from pathlib import Path
from glob import glob
import shutil
import json
from collections import defaultdict

scenes = glob("/home/nils/datasets/bop/output/*/bop_data/ycbv/train_pbr/*/")
output_path = Path("/media/Data1/BOP/ycbv_darknet/")
train_out_path = output_path / 'train'
val_out_path = output_path / 'val'
test_out_path = output_path / 'test'

img_path = output_path / "images"
label_path = output_path / "labels"

# ensure directories exist
def ensure_dir(dir):
    if not Path.exists(dir):
        os.makedirs(dir)

for dir in [train_out_path, val_out_path, test_out_path]:
    ensure_dir(dir / 'images')
    ensure_dir(dir / 'labels')

split = {
        'train': 0.8,
        'val': 0.1, 
        'test': 0.1,
        }

def write_output(scene: Path, images: list, img_id_labels: dict, output_path: Path, img_nr: int) -> int:
    added_imgs = 0
    for img_id, labels in img_id_labels.items():
        img_path = scene / images[img_id]['file_name']
        img_out_path = output_path / "images" / f"{img_nr:06}.jpg"
        ann_out_path = output_path / "labels" / f"{img_nr:06}.txt"
        shutil.copy(img_path, output_path / "images" / f'{img_nr:06}.jpg')
        with open(ann_out_path, 'w') as f:
            f.write('\n'.join(labels))
        # print(f'would copy img to {img_out_path}')
        # print(f'would write annotations to {ann_out_path}')
        img_nr += 1
        added_imgs += 1
    print(f'added {added_imgs} imgs to {output_path}')
    return img_nr

train_img_nr = 0
val_img_nr = 0
test_img_nr = 0
for scene in scenes:
    scene = Path(scene)
    print(f'processing scene {scene}')
    with open(scene/"scene_gt_coco.json", 'r') as f:
        gt_labels = json.load(f)
    # print(gt_labels)
    images = gt_labels['images']
    annotations = gt_labels['annotations']
    img_id_labels = defaultdict(list)
    for annotation in annotations:
        # calculate bbox format
        im_w, im_h = annotation['width'], annotation['height']
        x, y, w, h = annotation['bbox']
        # calculate center and normalize to [0..1]
        center_x = ((x + (x + w)) / 2) / im_w
        center_y = ((y + (y + h)) / 2) / im_h
        w = w / im_w
        h = h / im_h

        label_str = f'{annotation["category_id"]} {center_x} {center_y} {w} {h}'

        img_id_labels[annotation['image_id']].append(label_str)
    
    train_offset = int(split['train']*len(images))
    val_offset = train_offset + int(split['val']*len(images))
    test_offset = val_offset + int(split['test']*len(images))
    train_label_split = dict(list(img_id_labels.items())[:train_offset])
    val_label_split = dict(list(img_id_labels.items())[train_offset:val_offset])
    test_label_split = dict(list(img_id_labels.items())[val_offset:])
    
    train_img_nr = write_output(scene, images, train_label_split, train_out_path, train_img_nr)
    val_img_nr = write_output(scene, images, val_label_split, val_out_path, val_img_nr)
    test_img_nr = write_output(scene, images, test_label_split, test_out_path, test_img_nr)
        
print(f'wrote {train_img_nr} train images')
print(f'wrote {val_img_nr} val images')
print(f'wrote {test_img_nr} test images')
    



