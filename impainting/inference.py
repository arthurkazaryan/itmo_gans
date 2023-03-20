import random
import sys
from argparse import ArgumentParser
from pathlib import Path

import cv2
import torch
import numpy as np
from PIL import Image

sys.path.append('./MAT')
from MAT import dnnlib
from MAT import legacy
from MAT.datasets.mask_generator_512 import RandomMask
from MAT.generate_image import copy_params_and_buffers
from MAT.networks.mat import Generator

# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cpu')


def restore_video(video_path: str, output_path: str, mask_path: str = None, seed: int = 240):

    def preprocess_image(array: np.ndarray):
        size = (512, 512)
        img = Image.fromarray(array)
        img.thumbnail(size)
        full_img = Image.new('RGB', size, (0, 0, 0))
        full_img.paste(
            img, (int((size[0] - img.size[0]) / 2), int((size[1] - img.size[1]) / 2))
        )
        return np.array(full_img)

    def preprocess_mask(array: np.ndarray):
        size = (512, 512)
        img = Image.fromarray(array)
        img.thumbnail(size)
        full_img = Image.new('L', size, 255)
        full_img.paste(
            img, (int((size[0] - img.size[0]) / 2), int((size[1] - img.size[1]) / 2))
        )
        return np.array(full_img)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    cap = cv2.VideoCapture(video_path)
    height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    min_side = min(height, width)
    net_res = 512 if min_side > 512 else min_side

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                                   (net_res, net_res))

    if mask_path:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        mask = preprocess_mask(mask) / 255
        mask = torch.from_numpy(mask).float().to(DEVICE).unsqueeze(0).unsqueeze(0)
    else:
        mask = RandomMask(net_res)  # adjust the masking ratio by using 'hole_range'
        mask = torch.from_numpy(mask).float().to(DEVICE).unsqueeze(0)

    with dnnlib.util.open_url('./MAT/pretrained/CelebA-HQ_512.pkl') as f:
        g_saved = legacy.load_network_pkl(f)['G_ema'].to(DEVICE).eval().requires_grad_(False)  # type: ignore

    generator = Generator(z_dim=512, c_dim=0, w_dim=512, img_resolution=net_res, img_channels=3)\
        .to(DEVICE).eval().requires_grad_(False)
    copy_params_and_buffers(g_saved, generator, require_all=True)
    label = torch.zeros([1, generator.c_dim], device=DEVICE)

    noise_mode = 'random' if min_side != 512 else 'const'

    frame_idx = 1
    with torch.no_grad():
        while cap.isOpened():
            print(f'Processing frame: {frame_idx}')
            ret, image = cap.read()
            if not ret:
                break
            image = preprocess_image(image)[:, :, ::-1]
            image = image.transpose(2, 0, 1)[:3]
            image = (torch.from_numpy(image.copy()).float().to(DEVICE) / 127.5 - 1).unsqueeze(0)
            z = torch.from_numpy(np.random.randn(1, generator.z_dim)).to(DEVICE)
            output = generator(image, mask, z, label, truncation_psi=1, noise_mode=noise_mode)
            output = (output.permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
            output = output[0].cpu().numpy()
            video_writer.write(output[:, :, ::-1])

            Image.fromarray(output, 'RGB').save('./debug.jpeg')

            frame_idx += 1

    cap.release()
    video_writer.release()


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--video_path", required=True, type=str, help="Path to a video")
    parser.add_argument("--mask_path", required=True, type=str, help="Path to a mask")
    parser.add_argument("--output_path", required=True, type=str, help="Path to save a video")

    args = parser.parse_args()

    assert Path(args.video_path).is_file(), f'Video "{args.video_path}" was not found.'
    assert Path(args.mask_path).is_file(), f'Mask "{args.video_path}" was not found.'

    restore_video(
        video_path=args.video_path,
        mask_path=args.mask_path,
        output_path=args.output_path
    )
