"""Segment one image on the evaluation grid, optionally using CUDA graph replay."""
import os
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE','1')
import argparse
from contextlib import nullcontext
from pathlib import Path
import cv2
import torch
from rescale.build import build_model
from rescale.config import load_config
from rescale.data.transforms import get_val_transforms
from rescale.inference.segmented import SegmentedRecursiveSession


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/rescale.yaml')
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--image',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--graph',action='store_true')
    a=p.parse_args()
    cfg=load_config(a.config)
    model=build_model(cfg).cuda().eval()
    ckpt=torch.load(a.checkpoint,map_location='cpu')
    model.load_state_dict(ckpt.get('model_state',ckpt),strict=True)
    image=cv2.imread(a.image,cv2.IMREAD_COLOR)
    if image is None: raise FileNotFoundError(a.image)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    x=get_val_transforms(cfg['data']['image_size'])(image=image)['image'][None].cuda()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        with SegmentedRecursiveSession(model) if a.graph else nullcontext():
            mask=model.predict(x).argmax(1)[0].cpu().numpy().astype('uint8')*255
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(a.output),mask): raise OSError(f'Cannot write {a.output}')
    print(f'Saved {mask.shape[1]} x {mask.shape[0]} mask: {a.output}')


if __name__=='__main__': main()
