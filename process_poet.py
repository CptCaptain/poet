import cv2
import pickle
import torch
import os
import sys
from models import build_model
from main import get_args_parser

from accelerate import Accelerator

accelerator = Accelerator()


CHECKPOINT = './output_squirrel/checkpoint0049.pth'


def new_image_arrived(image_path: str, last_mtime: int):
    current_mtime = os.path.getmtime(image_path)
    return current_mtime > last_mtime


def process_image(mode: str):
    image_path = "/home/nils/poet_demo/input_image.png"
    image = cv2.imread(image_path)
    
    parser = get_args_parser()
    args = parser.parse_args()
    args.batch_size = 16
    args.dec_layers = 5
    args.nheads = 16
    args.device = 'cuda'
    args.bbox_mode = 'backbone'
    model, criterion, matcher = build_model(args)
    model.load_state_dict(torch.load(CHECKPOINT)['model'], strict=False)
    model.eval()
    model = accelerator.prepare(model)

    last_mtime = 0

    while True:
        if new_image_arrived(image_path, last_mtime):
            last_mtime = os.path.getmtime(image_path)
            image = cv2.imread(image_path)

            print(f'{image=}')
            print(f'{image.shape=}')
            print(f'{type(image)=}')
            img_tensor = torch.tensor(image.transpose(2, 0, 1)).to(args.device)
            print(f'{type(img_tensor)=}')
            with torch.no_grad():
                # check if this transposition is correct
                out, n_boxes_per_sample = model([img_tensor])
            translation_vector = out['pred_translation']
            rotation_matrix = out['pred_rotation']
            pred_boxes = out['pred_boxes']
            print(f"{translation_vector=}")
            print(f"{rotation_matrix=}")
            print(f"{pred_boxes=}")
            result = {
                "translation_vector": translation_vector.detach().cpu().numpy(),
                "rotation_matrix": rotation_matrix.detach().cpu().numpy(),
                "pred_boxes": pred_boxes,
            }

            result_bytes = pickle.dumps(result)
            print(result_bytes, end="", flush=True)

        if mode != 'continuous':
            break
        time.sleep(0.1)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "initialize"
    os.chdir('/home/nils/poet')
    process_image(mode)

