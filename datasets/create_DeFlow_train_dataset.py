#!/usr/bin/env python3
"""
create_DeFlow_train_dataset_grayscale.py

Read images from source_dir, create downsampled images at requested scales
(using your existing load_at_multiple_scales function), and save single-channel
grayscale images to target_dir/<scale>x/ preserving filenames.

Saves images as true single-channel files using Pillow ('L' mode).
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
    # Validate ndarray
    if not isinstance(img, np.ndarray):
        # If load_at_multiple_scales returns a PIL Image or something else, convert
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


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-source_dir', required=True, help='path to directory containing HR images')
    parser.add_argument('-target_dir', required=True, help='path to target directory')
    parser.add_argument('-scales', nargs='+',  type=int, default=[1, 4], help='scales to downsample to')
    args = parser.parse_args()

    source_dir = args.source_dir
    scales = args.scales

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

        for img, out_dir in zip(images, scale_dirs):
            try:
                gray = to_grayscale_uint8(img)
            except Exception as e:
                tqdm.write(f"Skipping {fn} for {out_dir}: cannot convert to grayscale: {e}")
                continue

            out_path = os.path.join(out_dir, fn)
            try:
                save_grayscale_pil(gray, out_path)
            except Exception as e:
                tqdm.write(f"Failed to save {out_path}: {e}")

    print("Done.")


if __name__ == '__main__':
    main()
