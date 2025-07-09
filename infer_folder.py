import argparse
import os
from PIL import Image

import cv2
import numpy as np
import torch

from WinCLIP import WinClipAD


def load_model(args):
    device = f"cuda:{args.gpu_id}" if (torch.cuda.is_available() and args.use_cpu == 0) else "cpu"
    model = WinClipAD(out_size_h=args.resolution,
                      out_size_w=args.resolution,
                      device=device,
                      backbone=args.backbone,
                      pretrained_dataset=args.pretrained_dataset,
                      scales=args.scales,
                      img_resize=args.img_resize,
                      img_cropsize=args.img_cropsize)
    model = model.to(device)
    model.eval_mode()
    model.build_text_feature_gallery(args.class_name)
    return model


def infer_folder(model, image_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for name in os.listdir(image_dir):
        if not name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        path = os.path.join(image_dir, name)
        img = Image.open(path).convert("RGB")
        tensor = model.transform(img).unsqueeze(0).to(model.device)
        score = model(tensor)[0]
        score = (score - score.min()) / (score.max() - score.min() + 1e-8)
        heat = (score * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(np.array(img), 0.5, heatmap, 0.5, 0)
        cv2.imwrite(os.path.join(save_dir, name), overlay)


def get_args():
    parser = argparse.ArgumentParser(description="Infer images in a folder")
    parser.add_argument("--image-dir", type=str, required=True, help="folder with input images")
    parser.add_argument("--save-dir", type=str, default="results", help="where to save visualizations")
    parser.add_argument("--class-name", type=str, required=True, help="class name for prompts")
    parser.add_argument("--backbone", type=str, default="ViT-B-16-plus-240")
    parser.add_argument("--pretrained_dataset", type=str, default="laion400m_e32")
    parser.add_argument("--scales", nargs="+", type=int, default=[2, 3])
    parser.add_argument("--img-resize", type=int, default=240)
    parser.add_argument("--img-cropsize", type=int, default=240)
    parser.add_argument("--resolution", type=int, default=400)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--use-cpu", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    model = load_model(args)
    infer_folder(model, args.image_dir, args.save_dir)
