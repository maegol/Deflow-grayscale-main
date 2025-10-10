#!/usr/bin/env python3
"""
create_DeFlow_train_dataset_grayscale.py

Read images from source_dir, create downsampled images at requested scales
(using your existing load_at_multiple_scales function), and save single-channel
grayscale images to target_dir/<scale>x/ preserving filenames.

This version optionally enforces a fixed GT size (GT_size). If --gt_size is
given, the script will ensure:
  - images saved in target_dir/1x/ are exactly GT_size x GT_size
  - images saved in target_dir/<s>x/ are exactly (GT_size // s) x (GT_size // s)
Resizing uses Pillow with sensible resampling choices.
"""
import os
import argparse
import sys
from tqdm import tqdm
import numpy as np
from PIL import Image

# make sure your project path is correct for importing data.util
sys.path.insert(0, '../codes')
from data.util import is_image_file, load_at_multiple_scales
import cv2
import numpy as np

def _to_gray_if_needed(img):
    """Return a 2D uint8 grayscale numpy array regardless of input format."""
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    # squeeze channel dim if single-channel stored as HxWx1
    if img.ndim == 3 and img.shape[2] == 1:
        img = img.squeeze(axis=2)
    # if RGB/RGBA -> convert to gray using Rec.601
    if img.ndim == 3 and img.shape[2] >= 3:
        # OpenCV expects BGR; if loaded via cv2 it's BGR, if PIL->numpy it's RGB.
        # We do a neutral conversion using channels as-is (assuming RGB order).
        img = img[..., :3].astype(np.float32)
        gray = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
        gray = np.clip(gray, 0, 255).round().astype(np.uint8)
        return gray
    # if already 2D
    if img.ndim == 2:
        # ensure uint8
        if img.dtype != np.uint8:
            if np.issubdtype(img.dtype, np.floating):
                maxv = float(np.nanmax(img)) if img.size else 0.0
                if maxv <= 1.0 + 1e-8:
                    img = (np.clip(img, 0.0, 1.0) * 255.0).round().astype(np.uint8)
                else:
                    img = np.clip(img, 0, 255).round().astype(np.uint8)
            else:
                img = np.clip(img, 0, 255).astype(np.uint8)
        return img
    raise ValueError(f"Unsupported image shape: {getattr(img, 'shape', None)}")

def _enforce_fixed_sizes(lq_img, gt_img, GT_size, scale):
    """
    Ensure gt_img is (GT_size, GT_size) and lq_img is (GT_size//scale, GT_size//scale).
    Both outputs are 2D uint8 grayscale numpy arrays.
    """
    # convert to grayscale arrays
    gt = _to_gray_if_needed(gt_img)
    lq = _to_gray_if_needed(lq_img)

    desired_gt = int(GT_size)
    desired_lq = max(1, int(GT_size) // int(scale))

    # Resize GT if needed (use cubic for up/down)
    if gt.shape[:2] != (desired_gt, desired_gt):
        gt = cv2.resize(gt, (desired_gt, desired_gt), interpolation=cv2.INTER_CUBIC)

    # Resize LQ if needed (use area for downsampling)
    if lq.shape[:2] != (desired_lq, desired_lq):
        # if lq is larger than desired, use INTER_AREA, else BICUBIC
        if lq.shape[0] > desired_lq or lq.shape[1] > desired_lq:
            interp = cv2.INTER_AREA
        else:
            interp = cv2.INTER_CUBIC
        lq = cv2.resize(lq, (desired_lq, desired_lq), interpolation=interp)

    return lq, gt

def to_grayscale_uint8(img: np.ndarray) -> np.ndarray:
    """
    Convert an input image array to a 2D uint8 grayscale image (shape H,W).

    Handles:
      - shapes: (H,W), (H,W,1), (H,W,3), (H,W,4)
      - float arrays in [0,1] or in 0..255 range
      - integer arrays
    Returns:
      uint8 2D numpy array in range 0..255
    """
    # Convert PIL image to numpy if necessary
    if not isinstance(img, np.ndarray):
        img = np.array(img)

    # Squeeze singleton channel
    if img.ndim == 3 and img.shape[2] == 1:
        img = img.squeeze(axis=2)

    # If already 2D, use as-is (grayscale)
    if img.ndim == 2:
        gray = img.astype(np.float32)
    elif img.ndim == 3 and img.shape[2] in (3, 4):
        # Convert RGB/RGBA -> luminance (Rec. 601)
        rgb = img[..., :3].astype(np.float32)
        gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    else:
        raise ValueError(f"Unsupported image shape for grayscale conversion: {img.shape}")

    # Convert floats/ints -> uint8 with clipping
    if np.issubdtype(gray.dtype, np.floating):
        maxv = float(np.nanmax(gray)) if gray.size else 0.0
        # Heuristic: if values appear within [0,1], scale up to 0..255
        if maxv <= 1.0 + 1e-8:
            gray = np.clip(gray, 0.0, 1.0)
            gray = (gray * 255.0).round().astype(np.uint8)
        else:
            gray = np.clip(gray, 0.0, 255.0).round().astype(np.uint8)
    else:
        # integer types: clip and cast
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    return gray


def save_grayscale_pil(gray_uint8: np.ndarray, out_path: str):
    """
    Save a 2D uint8 numpy array as a single-channel grayscale image using Pillow.
    """
    if gray_uint8.ndim != 2 or gray_uint8.dtype != np.uint8:
        raise ValueError("save_grayscale_pil expects a 2D uint8 array.")
    pil = Image.fromarray(gray_uint8, mode='L')
    pil.save(out_path)


def pil_resize_grayscale(gray_uint8: np.ndarray, size: int, upsample_resample=Image.BICUBIC, downsample_resample=Image.LANCZOS):
    """
    Resize a 2D uint8 grayscale image to (size, size) using Pillow with
    intelligent resampling selection depending on up/down-sampling.
    Returns resized 2D uint8 numpy array.
    """
    if size <= 0:
        raise ValueError("size must be >= 1")
    if gray_uint8.ndim != 2:
        raise ValueError("pil_resize_grayscale expects 2D array")

    h, w = gray_uint8.shape
    if (h, w) == (size, size):
        return gray_uint8

    pil = Image.fromarray(gray_uint8, mode='L')
    # choose resample: use LANCZOS for downsample, BICUBIC for upsample
    if size < min(h, w):
        resample = downsample_resample
    else:
        resample = upsample_resample

    pil_resized = pil.resize((size, size), resample=resample)
    return np.array(pil_resized)


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-source_dir', required=True, help='path to directory containing HR images')
    p.add_argument('-target_dir', required=True, help='path to target directory')
    p.add_argument('-scales', nargs='+',  type=int, default=[1, 4], help='scales to downsample to')
    p.add_argument('--gt_size', type=int, default=None,
                   help='If set, enforce GT size for 1x images; LQ sizes will be gt_size//scale (min 1)')
    p.add_argument('--out_ext', type=str, default=None,
                   help='Output extension/format (e.g. .png). If not set, preserve input filename extension.')
    return p.parse_args()


def main():
    args = parse_args()

    source_dir = args.source_dir
    scales = args.scales
    gt_size = args.gt_size
    out_ext = args.out_ext

    # prepare output directories
    scale_dirs = []
    for scale in scales:
        dirpath = os.path.join(args.target_dir, f'{scale}x')
        os.makedirs(dirpath, exist_ok=True)
        scale_dirs.append(dirpath)

    files = sorted(os.listdir(source_dir))
    for fn in tqdm(files):
        if not is_image_file(fn):
            continue

        src_path = os.path.join(source_dir, fn)

        # Expect load_at_multiple_scales to return a list/tuple of numpy arrays (one per scale)
        try:
            images = load_at_multiple_scales(src_path, scales=scales)
        except Exception as e:
            tqdm.write(f"Error loading {src_path}: {e}. Skipping.")
            continue

        # Ensure the returned number matches the requested scales
        if len(images) != len(scale_dirs):
            tqdm.write(f"Warning: number of returned images ({len(images)}) "
                       f"!= number of scales ({len(scale_dirs)}) for {fn}. Truncating/padding as needed.")

        for img, out_dir, scale in zip(images, scale_dirs, scales):
            try:
                # convert to grayscale 2D uint8
                gray = to_grayscale_uint8(img)
            except Exception as e:
                tqdm.write(f"Skipping {fn} for {out_dir}: cannot convert to grayscale: {e}")
                continue

            # If GT size enforcement is requested, compute desired size for this scale
            if gt_size is not None:
                if scale == 1:
                    desired = gt_size
                else:
                    desired = max(1, gt_size // scale)
                if gray.shape[:2] != (desired, desired):
                    try:
                        gray = pil_resize_grayscale(gray, desired)
                        tqdm.write(f"Resized {fn} for scale {scale}x -> {desired}x{desired}")
                    except Exception as e:
                        tqdm.write(f"Failed resizing {fn} for scale {scale}x to {desired}: {e}")
                        continue

            # Build output path: preserve extension unless out_ext specified
            base_name, ext = os.path.splitext(fn)
            save_ext = out_ext if out_ext is not None else ext
            if not save_ext.startswith('.'):
                save_ext = '.' + save_ext
            out_path = os.path.join(out_dir, base_name + save_ext)

            try:
                save_grayscale_pil(gray, out_path)
            except Exception as e:
                tqdm.write(f"Failed to save {out_path}: {e}")

    print("Done.")


if __name__ == '__main__':
    main()
