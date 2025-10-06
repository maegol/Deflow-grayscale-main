# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 14:33:40 2025

@author: Maego
"""
"""
dicom_to_png_converter.py

Convert a repository organized as <patient>/<date>/<study> containing DICOM files
into a mirrored repository with PNG images.

Usage:
    - From a terminal/CLI:
        python dicom_to_png_converter.py /path/to/dicom_repo /path/to/output_repo --min-files 20

    - From an IDE (double-click / Run in IDE with no arguments): either a small GUI
      will open to select input and output folders, or you can set the IDE_* constants
      below so the script will run immediately with the hard-coded paths.

Requirements:
    pip install pydicom pillow numpy tqdm

What it does:
 - walks the input root recursively
 - attempts to read DICOM files (skips non-DICOM)
 - applies rescale (RescaleSlope/Intercept) and photometric interpretation
 - uses WindowCenter/WindowWidth if present for better visualization
 - supports multi-frame DICOMs (saves each frame as a separate PNG)
 - preserves the relative folder structure in the output directory
 - avoids re-writing existing PNGs unless --overwrite is used
 - creates the output repository if it does not exist
 - ignores any folder (study folder) that contains fewer than `min_files` DICOM files

"""

import argparse
import os
from pathlib import Path
import sys
import logging
from collections import defaultdict

try:
    import numpy as np
    from PIL import Image
    import pydicom
    from pydicom import dcmread
    from pydicom.errors import InvalidDicomError
except Exception as e:
    print("Missing required libraries. Please install: pydicom pillow numpy")
    raise

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **kw: x


# === EDIT THESE CONSTANTS WHEN RUNNING FROM AN IDE IF YOU WANT HARDCODED PATHS ===
# Leave either or both empty to fall back to the GUI selection dialogs.
IDE_INPUT_PATH = ""  # e.g. "/home/user/dicom_repo"
IDE_OUTPUT_PATH = ""  # e.g. "/home/user/png_repo"
# ============================================================================

# default minimum number of DICOM files required for a folder to be processed
DEFAULT_MIN_FILES = 20

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def is_dicom_file(path: Path) -> bool:
    """Quick check whether a file is DICOM by trying to read minimal header."""
    try:
        # Stop before reading pixel data for speed
        dcmread(str(path), stop_before_pixels=True)
        return True
    except InvalidDicomError:
        return False
    except Exception:
        return False


def apply_windowing(pixel_array: np.ndarray, ds: pydicom.dataset.FileDataset) -> np.ndarray:
    """Apply WindowCenter/WindowWidth if present; otherwise leave pixel_array as-is.
    Returns float array (not scaled to 0-255).
    """
    arr = pixel_array.astype(np.float32)

    wc = getattr(ds, 'WindowCenter', None)
    ww = getattr(ds, 'WindowWidth', None)

    if wc is None or ww is None:
        return arr

    # WindowCenter/WindowWidth may be multi-valued
    try:
        if isinstance(wc, pydicom.multival.MultiValue):
            wc = float(wc[0])
        else:
            wc = float(wc)
        if isinstance(ww, pydicom.multival.MultiValue):
            ww = float(ww[0])
        else:
            ww = float(ww)
    except Exception:
        return arr

    # Standard DICOM windowing formula
    lower = wc - 0.5 - (ww - 1) / 2.0
    upper = wc - 0.5 + (ww - 1) / 2.0

    arr = np.clip(arr, lower, upper)
    arr = (arr - lower) / (upper - lower)  # normalized 0..1
    return arr


def dicom_to_png_arrays(ds: pydicom.dataset.FileDataset) -> list:
    """Return a list of numpy arrays (uint8) representing frames to save as PNG.
    Handles MONOCHROME1/2 and RGB.
    """
    try:
        pixels = ds.pixel_array
    except Exception as e:
        raise RuntimeError(f"Cannot get pixel data: {e}")

    # Rescale using RescaleSlope/Intercept
    intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
    slope = float(getattr(ds, 'RescaleSlope', 1.0))

    # If pixels are multi-frame, shape might be (frames, H, W) or (frames, H, W, samples)
    arr = pixels.astype(np.float32) * slope + intercept

    photometric = getattr(ds, 'PhotometricInterpretation', '').upper()

    # If it's RGB and last dim is 3, just convert per-frame
    if photometric.startswith('RGB') or (arr.ndim == 3 and arr.shape[-1] == 3):
        # Ensure shape is (frames?, H, W, 3)
        if arr.ndim == 3:
            # single-frame RGB: (H, W, 3)
            arr_list = [arr]
        elif arr.ndim == 4:
            # multi-frame RGB: (frames, H, W, 3)
            arr_list = [arr[i] for i in range(arr.shape[0])]

        out_list = []
        for a in arr_list:
            # Normalize each channel to 0-255 independently by the max across channels
            a = a - np.min(a)
            if np.max(a) != 0:
                a = a / np.max(a)
            a8 = (a * 255.0).astype(np.uint8)
            out_list.append(a8)
        return out_list

    # For MONOCHROME images
    # If MONOCHROME1, invert later
    invert = photometric.startswith('MONOCHROME1')

    # If multi-frame
    if arr.ndim == 3:  # (frames, H, W)
        frames = [arr[i] for i in range(arr.shape[0])]
    elif arr.ndim == 2:
        frames = [arr]
    else:
        # Unexpected shape
        frames = [arr.reshape(arr.shape[0], -1)]

    out_list = []
    for f in frames:
        # Apply windowing if available (windowing uses the *pre-rescaled* pixel values semantics)
        f_win = apply_windowing(f, ds)
        # If windowing returned normalized 0..1, use it; otherwise normalize by min/max
        if np.max(f_win) <= 1.0 and np.min(f_win) >= 0.0:
            norm = f_win
        else:
            norm = f - np.min(f)
            if np.max(norm) != 0:
                norm = norm / np.max(norm)
            else:
                norm = norm

        if invert:
            norm = 1.0 - norm

        # Scale to uint8
        a8 = (norm * 255.0).astype(np.uint8)
        out_list.append(a8)

    return out_list


def save_array_as_png(arr: np.ndarray, out_path: Path) -> None:
    """Save a numpy array as PNG using PIL. Accepts 2D (L) or 3D (RGB) arrays."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if arr.ndim == 2:
        img = Image.fromarray(arr, mode='L')
    elif arr.ndim == 3 and arr.shape[2] == 3:
        img = Image.fromarray(arr)
    else:
        # try to squeeze or convert
        arr2 = np.squeeze(arr)
        if arr2.ndim == 2:
            img = Image.fromarray(arr2.astype(np.uint8), mode='L')
        else:
            # fallback: convert to 8-bit grayscale
            flat = arr2.astype(np.float32)
            flat = flat - flat.min()
            if flat.max() != 0:
                flat = flat / flat.max()
            img = Image.fromarray((flat * 255).astype(np.uint8), mode='L')

    img.save(str(out_path))


def process_file(in_path: Path, in_root: Path, out_root: Path, overwrite: bool = False) -> int:
    """Process one file. Returns number of PNGs written (0 if skipped or failed)."""
    try:
        if not is_dicom_file(in_path):
            return 0

        ds = dcmread(str(in_path))
        arrays = dicom_to_png_arrays(ds)

        # build output directory following relative structure
        rel = in_path.relative_to(in_root)
        # we want to preserve patient/date/study structure up to the study folder
        # so create same relative path but replace filename
        stem = in_path.stem
        written = 0
        for i, arr in enumerate(arrays):
            if len(arrays) == 1:
                out_name = f"{stem}.png"
            else:
                out_name = f"{stem}_frame{i:03d}.png"

            out_path = out_root.joinpath(rel.parent).with_suffix('') / out_name
            # ensure unique path
            out_path = Path(str(out_path))

            if out_path.exists() and not overwrite:
                logging.debug(f"Skipping existing {out_path}")
                continue

            save_array_as_png(arr, out_path)
            written += 1

        return written

    except Exception as e:
        logging.warning(f"Failed to convert {in_path}: {e}")
        return 0


def walk_and_convert(in_root: Path, out_root: Path, overwrite: bool = False, min_files: int = DEFAULT_MIN_FILES) -> None:
    """Walk the input tree, group files by their immediate parent folder and process
    only those folders that contain at least `min_files` DICOM files. This avoids
    creating unnecessary folders in the output repository.
    """
    all_files = [p for p in in_root.rglob('*') if p.is_file()]

    # Group files by their parent directory
    parents = defaultdict(list)
    for p in all_files:
        parents[p.parent].append(p)

    total_files_seen = 0
    written_total = 0
    folders_processed = 0
    folders_skipped = 0

    for parent, files in tqdm(list(parents.items()), desc='Folders'):
        # Count DICOM-like files in the folder quickly by extension first
        ext_count = sum(1 for f in files if f.suffix.lower() in ('.dcm', '.dicom'))
        dicom_count = ext_count

        # If extension-based count is below threshold, do a slow but accurate check
        if dicom_count < min_files:
            for f in files:
                # Stop early if we reach min_files to avoid excessive checks
                if dicom_count >= min_files:
                    break
                # If file has an uncommon extension, try a light DICOM header check
                if f.suffix.lower() not in ('.dcm', '.dicom'):
                    try:
                        if is_dicom_file(f):
                            dicom_count += 1
                    except Exception:
                        pass

        total_files_seen += len(files)

        if dicom_count < min_files:
            folders_skipped += 1
            logging.debug(f"Skipping folder {parent} (contains {dicom_count} DICOM files < {min_files})")
            continue

        # Process files in this folder
        folders_processed += 1
        for p in tqdm(files, desc=f'Processing {parent.name}', leave=False):
            written = process_file(p, in_root, out_root, overwrite=overwrite)
            written_total += written

    logging.info(f"Scanned {total_files_seen} files across {len(parents)} folders. ")
    logging.info(f"Processed {folders_processed} folders, skipped {folders_skipped} folders (min_files={min_files}).")
    logging.info(f"Wrote {written_total} PNG files to {out_root}")


# === New convenience wrapper so you can call from other Python code or from an IDE ===
def convert_repository(input_path, output_path, overwrite: bool = False, min_files: int = DEFAULT_MIN_FILES) -> None:
    """High-level function that creates the output repository if needed and runs the conversion.

    Parameters
    ----------
    input_path : str or Path
        Root of the DICOM repository organized as patient/date/study
    output_path : str or Path
        Root where PNGs will be saved (created if necessary)
    overwrite : bool
        Whether to overwrite existing PNG files
    min_files : int
        Minimum number of DICOM files that a folder must contain to be processed
    """
    in_root = Path(input_path).expanduser().resolve()
    out_root = Path(output_path).expanduser().resolve()

    if not in_root.exists():
        raise FileNotFoundError(f"Input path does not exist: {in_root}")

    out_root.mkdir(parents=True, exist_ok=True)
    logging.info(f"Converting DICOMs under {in_root} -> {out_root} (min_files={min_files})")
    walk_and_convert(in_root, out_root, overwrite=overwrite, min_files=min_files)


def parse_args():
    parser = argparse.ArgumentParser(description='Convert DICOM repo to PNG repository')
    parser.add_argument('input', type=str, nargs='?', help='Path to input repository (root)')
    parser.add_argument('output', type=str, nargs='?', help='Path to output repository (root)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing PNG files')
    parser.add_argument('--nogui', action='store_true', help='Disable GUI prompts when running in an IDE')
    parser.add_argument('--min-files', type=int, default=DEFAULT_MIN_FILES, help='Minimum .dcm files required in a folder to process (default 20)')
    return parser.parse_args()


def run_with_gui_selection(overwrite_default: bool = False, min_files_default: int = DEFAULT_MIN_FILES) -> None:
    """Open folder selection dialogs (useful when running inside an IDE with no CLI).

    Requires tkinter; if not available, prints instructions instead.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, simpledialog

        root = tk.Tk()
        root.withdraw()

        in_dir = filedialog.askdirectory(title='Select input DICOM repository')
        if not in_dir:
            print('No input selected, exiting.')
            return
        out_dir = filedialog.askdirectory(title='Select output directory (will be created if needed)')
        if not out_dir:
            print('No output selected, exiting.')
            return

        overwrite = overwrite_default
        try:
            if messagebox.askyesno('Overwrite', 'Overwrite existing PNG files if they exist?'):
                overwrite = True
        except Exception:
            pass

        # Ask for min_files with a simple dialog (pre-filled with default)
        try:
            val = simpledialog.askinteger('Min files', 'Minimum number of DICOM files per folder to process:', initialvalue=min_files_default, minvalue=1)
            if val is None:
                min_files = min_files_default
            else:
                min_files = int(val)
        except Exception:
            min_files = min_files_default

        convert_repository(in_dir, out_dir, overwrite=overwrite, min_files=min_files)

    except Exception as e:
        print('tkinter not available or an error occurred while opening file dialogs:', e)
        print('You can run from terminal like: python dicom_to_png_converter.py /path/to/input /path/to/output --min-files 20')


if __name__ == '__main__':
    # Behavior when invoked directly:
    # - If called with two arguments, use them
    # - If called with no args and IDE constants are set, use the constants
    # - If called with no args and constants empty, open GUI folder dialogs
    if len(sys.argv) == 1:
        if IDE_INPUT_PATH and IDE_OUTPUT_PATH:
            # Use the hardcoded IDE paths (convenient when running inside an IDE)
            try:
                convert_repository(IDE_INPUT_PATH, IDE_OUTPUT_PATH, overwrite=False, min_files=DEFAULT_MIN_FILES)
            except Exception as e:
                logging.error(f"Error during conversion: {e}")
        else:
            # No args and no IDE constants: open GUI selection
            run_with_gui_selection(overwrite_default=False, min_files_default=DEFAULT_MIN_FILES)
    else:
        args = parse_args()

        # If user provided explicit paths, use them; otherwise fallback to GUI unless --nogui
        if args.input and args.output:
            try:
                convert_repository(args.input, args.output, overwrite=args.overwrite, min_files=args.min_files)
            except Exception as e:
                logging.error(f"Error during conversion: {e}")
                sys.exit(1)
        elif not args.nogui:
            run_with_gui_selection(overwrite_default=args.overwrite, min_files_default=args.min_files)
        else:
            print('Input and output paths are required when --nogui is set.')
