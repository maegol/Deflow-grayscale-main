# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 15:35:58 2025

@author: Maego
"""

"""
dicom_to_tif_converter.py

Same functionality as the PNG converter but writes TIFF files instead.

Features:
 - Mirrors the input <patient>/<date>/<study> structure into the output repo
 - Skips folders with fewer than `min_files` DICOMs (default 20)
 - Creates output repository if missing
 - Supports single-frame and multi-frame DICOMs
 - Optionally writes multi-frame DICOMs to a single multi-page TIFF with --multipage
 - GUI folder selection when run without arguments (IDE-friendly)

Usage:
    python dicom_to_tif_converter.py /path/to/dicom_repo /path/to/output_repo --min-files 20 --multipage

"""

import argparse
import sys
from pathlib import Path
import logging
from collections import defaultdict

try:
    import numpy as np
    from PIL import Image
    import pydicom
    from pydicom import dcmread
    from pydicom.errors import InvalidDicomError
except Exception:
    print('Missing required libraries. Install: pydicom pillow numpy')
    raise

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **kw: x

# === IDE convenience constants ===
IDE_INPUT_PATH = ""  # e.g. "/home/user/dicom_repo"
IDE_OUTPUT_PATH = ""
# ================================

DEFAULT_MIN_FILES = 20

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def is_dicom_file(path: Path) -> bool:
    try:
        dcmread(str(path), stop_before_pixels=True)
        return True
    except InvalidDicomError:
        return False
    except Exception:
        return False


def apply_windowing(pixel_array: np.ndarray, ds: pydicom.dataset.FileDataset) -> np.ndarray:
    arr = pixel_array.astype(np.float32)
    wc = getattr(ds, 'WindowCenter', None)
    ww = getattr(ds, 'WindowWidth', None)
    if wc is None or ww is None:
        return arr
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
    lower = wc - 0.5 - (ww - 1) / 2.0
    upper = wc - 0.5 + (ww - 1) / 2.0
    arr = np.clip(arr, lower, upper)
    arr = (arr - lower) / (upper - lower)
    return arr


def dicom_to_arrays(ds: pydicom.dataset.FileDataset) -> list:
    try:
        pixels = ds.pixel_array
    except Exception as e:
        raise RuntimeError(f'Cannot get pixel data: {e}')

    intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
    slope = float(getattr(ds, 'RescaleSlope', 1.0))
    arr = pixels.astype(np.float32) * slope + intercept
    photometric = getattr(ds, 'PhotometricInterpretation', '').upper()

    # RGB handling
    if photometric.startswith('RGB') or (arr.ndim == 3 and arr.shape[-1] == 3):
        if arr.ndim == 3:
            arr_list = [arr]
        elif arr.ndim == 4:
            arr_list = [arr[i] for i in range(arr.shape[0])]
        out = []
        for a in arr_list:
            a = a - np.min(a)
            if np.max(a) != 0:
                a = a / np.max(a)
            out.append((a * 255.0).astype(np.uint8))
        return out

    invert = photometric.startswith('MONOCHROME1')
    if arr.ndim == 3:
        frames = [arr[i] for i in range(arr.shape[0])]
    elif arr.ndim == 2:
        frames = [arr]
    else:
        frames = [arr.reshape(arr.shape[0], -1)]

    out = []
    for f in frames:
        f_win = apply_windowing(f, ds)
        if np.max(f_win) <= 1.0 and np.min(f_win) >= 0.0:
            norm = f_win
        else:
            norm = f - np.min(f)
            if np.max(norm) != 0:
                norm = norm / np.max(norm)
        if invert:
            norm = 1.0 - norm
        out.append((norm * 255.0).astype(np.uint8))
    return out


def save_arrays_as_tif(arrays, out_path: Path, multipage: bool = False) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if multipage and len(arrays) > 1:
        pil_images = []
        for a in arrays:
            if a.ndim == 2:
                pil_images.append(Image.fromarray(a, mode='L'))
            elif a.ndim == 3 and a.shape[2] == 3:
                pil_images.append(Image.fromarray(a))
            else:
                pil_images.append(Image.fromarray(np.squeeze(a).astype('uint8')))
        first, rest = pil_images[0], pil_images[1:]
        first.save(str(out_path), format='TIFF', save_all=True, append_images=rest)
    else:
        # Save frames separately
        for i, a in enumerate(arrays):
            if len(arrays) == 1:
                name = out_path.with_suffix('.tif')
            else:
                name = out_path.with_suffix(f'_frame{i:03d}.tif')
            if a.ndim == 2:
                img = Image.fromarray(a, mode='L')
            elif a.ndim == 3 and a.shape[2] == 3:
                img = Image.fromarray(a)
            else:
                img = Image.fromarray(np.squeeze(a).astype('uint8'))
            img.save(str(name), format='TIFF')


def process_file(in_path: Path, in_root: Path, out_root: Path, overwrite: bool = False, multipage: bool = False) -> int:
    try:
        if not is_dicom_file(in_path):
            return 0
        ds = dcmread(str(in_path))
        arrays = dicom_to_arrays(ds)
        rel = in_path.relative_to(in_root)
        stem = in_path.stem
        out_base = out_root.joinpath(rel.parent).with_suffix('') / stem
        written = 0
        if multipage and len(arrays) > 1:
            out_file = out_base.with_suffix('.tif')
            if out_file.exists() and not overwrite:
                return 0
            save_arrays_as_tif(arrays, out_base, multipage=True)
            written = 1
        else:
            for i, a in enumerate(arrays):
                if len(arrays) == 1:
                    out_file = out_base.with_suffix('.tif')
                else:
                    out_file = out_base.with_suffix(f'_frame{i:03d}.tif')
                if out_file.exists() and not overwrite:
                    continue
                save_arrays_as_tif([a], out_base, multipage=False)
                written += 1
        return written
    except Exception as e:
        logging.warning(f'Failed to convert {in_path}: {e}')
        return 0


def walk_and_convert(in_root: Path, out_root: Path, overwrite: bool = False, min_files: int = DEFAULT_MIN_FILES, multipage: bool = False) -> None:
    all_files = [p for p in in_root.rglob('*') if p.is_file()]
    parents = defaultdict(list)
    for p in all_files:
        parents[p.parent].append(p)

    total = 0
    written_total = 0
    folders_processed = 0
    folders_skipped = 0

    for parent, files in tqdm(list(parents.items()), desc='Folders'):
        ext_count = sum(1 for f in files if f.suffix.lower() in ('.dcm', '.dicom'))
        dicom_count = ext_count
        if dicom_count < min_files:
            for f in files:
                if dicom_count >= min_files:
                    break
                if f.suffix.lower() not in ('.dcm', '.dicom'):
                    try:
                        if is_dicom_file(f):
                            dicom_count += 1
                    except Exception:
                        pass
        if dicom_count < min_files:
            folders_skipped += 1
            continue
        folders_processed += 1
        for p in tqdm(files, desc=f'Processing {parent.name}', leave=False):
            written = process_file(p, in_root, out_root, overwrite=overwrite, multipage=multipage)
            written_total += written
            total += 1

    logging.info(f'Scanned {total} files across {len(parents)} folders.')
    logging.info(f'Processed {folders_processed} folders, skipped {folders_skipped} folders (min_files={min_files}).')
    logging.info(f'Wrote {written_total} TIFF files to {out_root}')


def convert_repository(input_path, output_path, overwrite: bool = False, min_files: int = DEFAULT_MIN_FILES, multipage: bool = False) -> None:
    in_root = Path(input_path).expanduser().resolve()
    out_root = Path(output_path).expanduser().resolve()
    if not in_root.exists():
        raise FileNotFoundError(f'Input path does not exist: {in_root}')
    out_root.mkdir(parents=True, exist_ok=True)
    logging.info(f'Converting DICOMs under {in_root} -> {out_root} (min_files={min_files}, multipage={multipage})')
    walk_and_convert(in_root, out_root, overwrite=overwrite, min_files=min_files, multipage=multipage)


def parse_args():
    parser = argparse.ArgumentParser(description='Convert DICOM repo to multi-page or single-page TIFF repository')
    parser.add_argument('input', type=str, nargs='?', help='Path to input repository (root)')
    parser.add_argument('output', type=str, nargs='?', help='Path to output repository (root)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing TIFF files')
    parser.add_argument('--nogui', action='store_true', help='Disable GUI prompts when running in an IDE')
    parser.add_argument('--min-files', type=int, default=DEFAULT_MIN_FILES, help='Minimum .dcm files required in a folder to process (default 20)')
    parser.add_argument('--multipage', action='store_true', help='Save multi-frame DICOMs as single multi-page TIFF when possible')
    return parser.parse_args()


def run_with_gui_selection(overwrite_default: bool = False, min_files_default: int = DEFAULT_MIN_FILES, multipage_default: bool = False) -> None:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, simpledialog
        root = tk.Tk(); root.withdraw()
        in_dir = filedialog.askdirectory(title='Select input DICOM repository')
        if not in_dir:
            print('No input selected, exiting.'); return
        out_dir = filedialog.askdirectory(title='Select output directory (will be created if needed)')
        if not out_dir:
            print('No output selected, exiting.'); return
        overwrite = overwrite_default
        try:
            if messagebox.askyesno('Overwrite', 'Overwrite existing TIFF files if they exist?'):
                overwrite = True
        except Exception:
            pass
        try:
            val = simpledialog.askinteger('Min files', 'Minimum number of DICOM files per folder to process:', initialvalue=min_files_default, minvalue=1)
            min_files = int(val) if val is not None else min_files_default
        except Exception:
            min_files = min_files_default
        try:
            multipage = messagebox.askyesno('Multipage', 'Save multi-frame DICOMs as single multi-page TIFF?')
        except Exception:
            multipage = multipage_default
        convert_repository(in_dir, out_dir, overwrite=overwrite, min_files=min_files, multipage=multipage)
    except Exception as e:
        print('tkinter not available or an error occurred while opening file dialogs:', e)
        print('You can run from terminal like: python dicom_to_tif_converter.py /path/to/input /path/to/output --min-files 20 --multipage')


if __name__ == '__main__':
    if len(sys.argv) == 1:
        if IDE_INPUT_PATH and IDE_OUTPUT_PATH:
            try:
                convert_repository(IDE_INPUT_PATH, IDE_OUTPUT_PATH, overwrite=False, min_files=DEFAULT_MIN_FILES)
            except Exception as e:
                logging.error(f'Error during conversion: {e}')
        else:
            run_with_gui_selection(overwrite_default=False, min_files_default=DEFAULT_MIN_FILES, multipage_default=False)
    else:
        args = parse_args()
        if args.input and args.output:
            try:
                convert_repository(args.input, args.output, overwrite=args.overwrite, min_files=args.min_files, multipage=args.multipage)
            except Exception as e:
                logging.error(f'Error during conversion: {e}'); sys.exit(1)
        elif not args.nogui:
            run_with_gui_selection(overwrite_default=args.overwrite, min_files_default=args.min_files, multipage_default=args.multipage)
        else:
            print('Input and output paths are required when --nogui is set.')
