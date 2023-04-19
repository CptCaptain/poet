import cv2
import pickle
import torch
import os
import sys
from contextlib import contextmanager
from models import build_model
from main import get_args_parser

from accelerate import Accelerator

accelerator = Accelerator()


# CHECKPOINT = './output_squirrel/checkpoint0049.pth'
CHECKPOINT = './output_squirrel/checkpoint0199.pth'

@contextmanager
def suppress_stdout():
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = original_stdout


def new_image_arrived(image_path: str, last_mtime: int):
    current_mtime = os.path.getmtime(image_path)
    return current_mtime > last_mtime


def process_image(mode: str):
    # image_path = "/home/nils/poet_demo/input_image.png"
    image_path = "/home/nils/poet_demo/squirrel_2.png"
    image = cv2.imread(image_path)
    
    parser = get_args_parser()
    args = parser.parse_args()
    args.batch_size = 16
    args.dec_layers = 5
    args.nheads = 16
    args.device = 'cuda'
    args.bbox_mode = 'backbone'
    with suppress_stdout():
        model, criterion, matcher = build_model(args)
        checkpoint = {k: v for k, v in torch.load(CHECKPOINT)['model'].items() if not k.startswith('backbone')}
        model.load_state_dict(checkpoint, strict=False)
        model.eval()
        model = accelerator.prepare(model)

    last_mtime = 0

    while True:
        if new_image_arrived(image_path, last_mtime):
            last_mtime = os.path.getmtime(image_path)
            image = cv2.imread(image_path)

            img_tensor = torch.tensor(image.transpose(2, 0, 1)).to(args.device)
            with torch.no_grad():
                # check if this transposition is correct
                out, n_boxes_per_sample = model([img_tensor])
            translation_vector = out['pred_translation'][0]
            rotation_matrix = out['pred_rotation'][0]
            pred_boxes = out['pred_boxes'][0]
            pred_classes = out['pred_classes'][0]
            if True:
                print(pred_boxes)
                print(pred_classes)
            result = None
            for t, r, b, c in zip(translation_vector, rotation_matrix, pred_boxes, pred_classes):
                if not c == 21:
                    continue
                result = {
                    "translation_vector": t.detach().cpu().numpy().tolist(),
                    "rotation_matrix": r.detach().cpu().numpy().tolist(),
                    "pred_boxes": b.detach().cpu().numpy().tolist(),
                    "pred_classes": c.detach().cpu().numpy().tolist(),
                }

            # result_bytes = pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
            if result is not None:
                print(result, flush=True)
            else:
                print('No Squirrel detected!')

        if mode != 'continuous':
            break
        time.sleep(0.1)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "initialize"
    os.chdir('/home/nils/poet')
    process_image(mode)

