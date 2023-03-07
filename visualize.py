# Original Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague
#######################################################################
# Adapted from bop_toolkit/scripts/vis_est_poses to work with non-BOP images

"""Visualizes object models in pose estimates saved in a BOP-like format."""

import os
import numpy as np
import itertools

from bop_toolkit_lib import config
from bop_toolkit_lib import misc
from bop_toolkit_lib import renderer
from bop_toolkit_lib import visualization


# PARAMETERS.
################################################################################
p = {
        # Top N pose estimates (with the highest score) to be visualized for each
        # object in each image.
        'n_top': 1,  # 0 = all estimates, -1 = given by the number of GT poses.

        # True = one visualization for each (im_id, obj_id), False = one per im_id.
        'vis_per_obj_id': True,

        # Indicates whether to render RGB image.
        'vis_rgb': True,

        # Indicates whether to resolve visibility in the rendered RGB images (using
        # depth renderings). If True, only the part of object surface, which is not
        # occluded by any other modeled object, is visible. If False, RGB renderings
        # of individual objects are blended together.
        'vis_rgb_resolve_visib': True,

        # Indicates whether to render depth image.
        'vis_depth_diff': False,

        # If to use the original model color.
        'vis_orig_color': False,

        # Type of the renderer (used for the VSD pose error function).
        'renderer_type': 'vispy',  # Options: 'vispy', 'cpp', 'python'.

        # Names of files with pose estimates to visualize (assumed to be stored in
        # folder config.eval_path). See docs/bop_challenge_2019.md for a description
        # of the format. Example results can be found at:
        # https://bop.felk.cvut.cz/media/data/bop_sample_results/bop_challenge_2019/
        'result_filenames': [
            '/path/to/csv/with/results',
            ],

        # Folder containing the BOP datasets.
        # 'datasets_path': config.datasets_path,

        # Folder for output visualisations.
        'vis_path': os.path.join(config.output_path, 'vis_est_poses'),

        # Path templates for output images.
        'vis_rgb_tpath': os.path.join(
            '{vis_path}', '{result_name}', '{img_name:06d}', '{vis_name}.jpg'),
        'vis_depth_diff_tpath': os.path.join(
            '{vis_path}', '{result_name}', '{img_name:06d}',
            '{vis_name}_depth_diff.jpg'),
        }
################################################################################


# Load colors.
colors_path = os.path.join(
        os.path.dirname(visualization.__file__),
        'colors.json'
        )
colors = inout.load_json(colors_path)

# TODO fill with sensible content
model_paths = {
        'id': 'Path'
        }

def vis(img: np.ndarray, model_ids: List, ests: List[dict], models: Optional[dict] = None, K=None):
    """ Visualizes the estimated poses of given objects in a given image

        :param img: image as array (or similar, TBD)
        :param model_ids: List of model ids that can be resolved into 3D models
        :param ests: List of dictionaries with required keys:
                - obj_id: Object ID (can be mapped to 3D model)
                - R: 3x3 Rotation matrix
                - t: 3x1 Translation matrix
        :param models: Optional dict with model IDs and paths, to be used instead of model_ids
        :param K: 3x3 matrix with camera intrinsics
    """
    # Rendering mode.
    # We only support RGB
    renderer_mode = 'rgb'

    # Create a renderer.
    width, height = img.shape[:2]
    ren = renderer.create_renderer(
          width, height, p['renderer_type'], mode=renderer_mode)

    # Load object models.
    models = {}
    for obj_id in model_ids:
        misc.log('Loading 3D model of object {}...'.format(obj_id))
        model_path = dp_model['model_tpath'].format(obj_id=obj_id)
        model_color = None
        if not p['vis_orig_color']:
            model_color = tuple(colors[(obj_id - 1) % len(colors)])
        ren.add_object(obj_id, model_path, surf_color=model_color)

    # Organize the pose estimates by scene, image and object.
    misc.log('Organizing pose estimates...')
    ests_org = {}
    for est in ests:
      ests_org.setdefault(est['scene_id'], {}).setdefault(
              est['im_id'], {}).setdefault(est['obj_id'], []).append(est)

    for scene_id, scene_ests in ests_org.items():
        for im_ind, (im_id, im_ests) in enumerate(scene_ests.items()):
            im_ests_vis = []
            im_ests_vis_obj_ids = []
            # Select the number of top estimated poses to visualize.
            if p['n_top'] == 0:  # All estimates are considered.
                n_top_curr = None
            elif p['n_top'] == -1:  # Given by the number of GT poses.
                n_gt = sum([gt['obj_id'] == obj_id for gt in scene_gt[im_id]])
                n_top_curr = n_gt
            else:  # Specified by the parameter n_top.
                n_top_curr = p['n_top']
            # We can't sort as all prediction of PoET have score 1.
            obj_ests_sorted = obj_ests[slice(0, n_top_curr)]

            # Get list of poses to visualize.
            for est in obj_ests_sorted:
                est['obj_id'] = obj_id

              # Text info to write on the image at the pose estimate.
                if p['vis_per_obj_id']:
                  est['text_info'] = [
                          {'name': '', 'val': est['score'], 'fmt': ':.2f'}
                          ]
                else:
                  val = '{}:{:.2f}'.format(obj_id, est['score'])
                est['text_info'] = [{'name': '', 'val': val, 'fmt': ''}]

            im_ests_vis.append(obj_ests_sorted)
            im_ests_vis_obj_ids.append(obj_id)

          # Join the per-object estimates if only one visualization is to be made.
        if not p['vis_per_obj_id']:
          im_ests_vis = [list(itertools.chain.from_iterable(im_ests_vis))]

        # Visualization name.
        if p['vis_per_obj_id']:
            vis_name = '{im_id:06d}_{obj_id:06d}'.format(
                    im_id=im_id, obj_id=im_ests_vis_obj_ids[ests_vis_id])
        else:
            vis_name = '{im_id:06d}'.format(im_id=im_id)

        # Path to the output RGB visualization.
        vis_rgb_path = None
        if p['vis_rgb']:
            vis_rgb_path = p['vis_rgb_tpath'].format(
                    vis_path=p['vis_path'], result_name=result_name, scene_id=scene_id,
                    vis_name=vis_name)

        # Visualization.
        visualization.vis_object_poses(
                poses=ests_vis, K=K, renderer=ren, rgb=img,
                vis_rgb_path=vis_rgb_path, vis_depth_diff_path=vis_depth_diff_path,
                vis_rgb_resolve_visib=p['vis_rgb_resolve_visib'])

misc.log('Done.')
