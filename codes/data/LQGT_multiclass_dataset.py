import numpy as np
import torch
import torch.utils.data as data
import data.util as util
from data import random_crop, center_crop, random_flip, random_rotation, imread
import matlablike_resize
import os
import cv2


def getEnv(name): import os; return True if name in os.environ.keys() else False

class LQGTMulticlassDataset(data.Dataset):
    '''
    Read LQ (Low Quality, here is LR) and GT image pairs.
    If only GT image is provided, generate LQ image on-the-fly.
    The pair is ensured by 'sorted' function, so please check the name convention.
    '''

    def __init__(self, opt):
        super(LQGTMulticlassDataset, self).__init__()
        self.opt = opt
        self.crop_size = opt.get("GT_size", None)
        self.alignment = opt.get("alignment", None) # aligns random crops
        self.target_scale = opt.get("target_scale", None)
        self.scale = None
        self.random_scale_list = [1]

        if 'normalize' in opt:
            self.normalize = True
            self.mean_clean_hr = torch.tensor(opt['normalize']['mean_clean_hr']).reshape(3,1,1)/255
            self.mean_noisy_hr = torch.tensor(opt['normalize']['mean_noisy_hr']).reshape(3,1,1)/255
            self.std_clean_hr = torch.tensor(opt['normalize']['std_clean_hr']).reshape(3,1,1)/255
            self.std_noisy_hr = torch.tensor(opt['normalize']['std_noisy_hr']).reshape(3,1,1)/255

            self.mean_clean_lr = torch.tensor(opt['normalize']['mean_clean_lr']).reshape(3,1,1)/255
            self.mean_noisy_lr = torch.tensor(opt['normalize']['mean_noisy_lr']).reshape(3,1,1)/255
            self.std_clean_lr = torch.tensor(opt['normalize']['std_clean_lr']).reshape(3,1,1)/255
            self.std_noisy_lr = torch.tensor(opt['normalize']['std_noisy_lr']).reshape(3,1,1)/255

                    # If dataset images are grayscale, reduce mean/std that were provided for 3 channels -> 1 channel
            if self.normalize:
                def _to_1ch(t):
                    # input: torch tensor shaped (3,1,1) or (1,1,1)
                    try:
                        if t.shape[0] == 3:
                            # average the 3 channel scalars into one (keeps consistent brightness)
                            m = t.mean().reshape(1,1,1)
                            return m
                    except Exception:
                        pass
                    return t
                self.mean_clean_hr = _to_1ch(self.mean_clean_hr)
                self.mean_noisy_hr = _to_1ch(self.mean_noisy_hr)
                self.std_clean_hr = _to_1ch(self.std_clean_hr)
                self.std_noisy_hr = _to_1ch(self.std_noisy_hr)
                self.mean_clean_lr = _to_1ch(self.mean_clean_lr)
                self.mean_noisy_lr = _to_1ch(self.mean_noisy_lr)
                self.std_clean_lr = _to_1ch(self.std_clean_lr)
                self.std_noisy_lr = _to_1ch(self.std_noisy_lr)

        else:
            self.normalize = False

        self.data_type = self.opt['data_type']
        self.paths_LQ, self.paths_GT = None, None
        self.sizes_LQ, self.sizes_GT = None, None
        self.LQ_env, self.GT_env = None, None  # environment for lmdb

        dataroot_GT = opt['dataroot_GT']
        dataroot_LQ = opt['dataroot_LQ']

        assert type(dataroot_GT) == list
        assert type(dataroot_LQ) == list

        gpu = True
        augment = True

        self.use_flip = opt["use_flip"] if "use_flip" in opt.keys() else False
        self.use_rot = opt["use_rot"] if "use_rot" in opt.keys() else False
        self.use_crop = opt["use_crop"] if "use_crop" in opt.keys() else False
        self.center_crop_hr_size = opt.get("center_crop_hr_size", None)
        # y_labels_file_path  = opt['dataroot_y_labels']

        self.dataroot_GT = dataroot_GT
        self.dataroot_LQ = dataroot_LQ

        self.paths_GT, self.sizes_GT = [], []
        self.paths_LQ, self.sizes_LQ = [], []

        self.y_labels = []

        for class_i, (root_GT, root_LQ) in enumerate(zip(dataroot_GT, dataroot_LQ)):
            paths_GT, sizes_GT = util.get_image_paths(self.data_type, root_GT)
            paths_LQ, sizes_LQ = util.get_image_paths(self.data_type, root_LQ)
            
            assert paths_GT, 'Error: GT path is empty.'
            assert paths_LQ, 'Error: LQ path is empty on the fly downsampling not yet supported.'
            # print(len(paths_GT), self.paths_GT[:10])
            if len(paths_LQ) != len(paths_GT):
                min_len = min(len(paths_LQ), len(paths_GT))
                # Truncate both lists to the minimum length
                paths_LQ = paths_LQ[:min_len]
                paths_GT = paths_GT[:min_len]
               
                print(f"[Dataset] Warning: mismatched dataset sizes. Truncated to {min_len} pairs (LQ:{len(paths_LQ)}, GT:{len(paths_GT)}).")
            
            if paths_LQ and paths_GT:
                assert len(paths_LQ) == len(paths_GT), 'GT and LQ datasets have different number of images - LQ: {}, GT: {}'.format(len(paths_LQ), len(paths_GT))
    

            # limit to n_max images per class if specified
            n_max = opt.get('n_max', None)
            if n_max is not None:
                if n_max > 0:
                    paths_GT = paths_GT[:n_max]
                    paths_LQ = paths_LQ[:n_max]
                elif n_max < 0:
                    paths_GT = paths_GT[n_max:]
                    paths_LQ = paths_LQ[n_max:]
                else:
                    raise RuntimeError("Not implemented")

            self.paths_GT += paths_GT
            if sizes_GT is not None:
                self.sizes_GT += sizes_GT
            self.paths_LQ += paths_LQ
            if sizes_LQ is not None:
                self.sizes_LQ += sizes_LQ

            self.y_labels += [class_i]*len(paths_GT)
        
        if not self.sizes_GT:
            self.sizes_GT = None
        if not self.sizes_LQ:
            self.sizes_LQ = None

        assert self.paths_GT, 'Error: GT path is empty.'
        # print(len(self.paths_GT), self.paths_GT[:10])
        if self.paths_LQ and self.paths_GT:
            assert len(self.paths_LQ) == len(
                self.paths_GT
            ), 'GT and LQ datasets have different number of images - {}, {}.'.format(
                len(self.paths_LQ), len(self.paths_GT))

        # preload data to RAM
        if opt.get('preload', True):
            print('preloading data...')
            self.preloaded = True
            self.GT_imgs = []
            
            # get GT image
            for GT_path in self.paths_GT:
                self.GT_imgs.append(imread(GT_path))
            
            if self.paths_LQ:
                self.LQ_imgs = []
                for LQ_path in self.paths_LQ:
                    self.LQ_imgs.append(imread(LQ_path))
        else: 
            self.preloaded = False

        self.gpu = gpu
        self.augment = augment

        self.measures = None

        self.label_hist = {}

    def __getitem__(self, index):
        if self.data_type == 'lmdb':
            if (self.GT_env is None) or (self.LQ_env is None):
                self._init_lmdb()

        # get GT image path & image (hr) and LQ image (lr)
        GT_path = self.paths_GT[index]
        if self.preloaded:  # use preloaded images
            hr = self.GT_imgs[index]
        else:
            hr = imread(GT_path)

        if self.paths_LQ:
            LQ_path = self.paths_LQ[index]
            if self.preloaded:
                lr = self.LQ_imgs[index]
            else:
                lr = imread(LQ_path)
        else:
            LQ_path = None
            raise RuntimeError("on-the-fly downsampling not implemented")

        # --- normalize image layout to CHW (C,H,W) early on ---
        # handle grayscale HxW -> 1xHxW, HWC -> CHW, CHW keep as-is
        def ensure_chw(img):
            if isinstance(img, np.ndarray):
                if img.ndim == 2:
                    # H x W -> 1 x H x W
                    return img[np.newaxis, :, :]
                if img.ndim == 3:
                    # detect HWC if last dim is channel count 1 or 3
                    if img.shape[2] in (1, 3):
                        return np.transpose(img, (2, 0, 1))
                    # otherwise assume it's already CHW
                    return img
            # if it's not a numpy array, just return it (imread should return np.ndarray)
            return img

        try:
            hr = ensure_chw(hr)
            lr = ensure_chw(lr)
        except Exception as e:
            raise RuntimeError(f"[Dataset] Failed to standardize image layout for index {index}, GT_path={GT_path}, LQ_path={LQ_path}: {e}")

        # --- ENSURE single-channel (grayscale) in CHW layout --------------------------------
        # Convert any 3-channel images to 1-channel using luminance, keeping dtype.
        def _to_grayscale_chw(arr):
            """
            Input: numpy array in CHW or HWC or HxW forms (but here we pass CHW).
            Return: CHW numpy array with 1 channel (shape (1,H,W)).
            """
            # assume arr is CHW already (as ensure_chw returned)
            if not isinstance(arr, np.ndarray):
                return arr
            if arr.ndim == 2:
                # H x W -> 1 x H x W
                return arr[np.newaxis, ...]
            if arr.ndim == 3:
                # if arr is CHW and first dim is channels
                C = arr.shape[0]
                if C == 1:
                    return arr
                if C == 3:
                    # arr is CHW with 3 channels -> convert to luminance
                    r = arr[0].astype(np.float32)
                    g = arr[1].astype(np.float32)
                    b = arr[2].astype(np.float32)
                    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
                    return gray[np.newaxis, ...]
            # fallback: return as is
            return arr

        try:
            hr = _to_grayscale_chw(hr)
            lr = _to_grayscale_chw(lr)
        except Exception as e:
            raise RuntimeError(f"[Dataset] Failed to convert to grayscale for index {index}, GT_path={GT_path}, LQ_path={LQ_path}: {e}")
        # -----------------------------------------------------------------------------------

        # Get GT_size (safe fallbacks)
        GT_size = None
        if hasattr(self, 'dataset_opt') and isinstance(self.dataset_opt, dict):
            GT_size = self.dataset_opt.get('GT_size', None)

        if GT_size is None and hasattr(self, 'opt') and isinstance(self.opt, dict):
            phase = getattr(self, 'phase', None)
            if phase is None:
                if hasattr(self, 'dataset_opt') and isinstance(self.dataset_opt, dict):
                    phase = self.dataset_opt.get('phase', 'train')
                else:
                    phase = 'train'
            GT_size = self.opt.get('datasets', {}).get(phase, {}).get('GT_size', None)

        if GT_size is None and hasattr(self, 'opt') and isinstance(self.opt, dict):
            GT_size = self.opt.get('GT_size', None)

        scale = int(self.opt.get('scale', 1))

        # If user requested resizing, do it safely (convert to HWC for cv2)
        if GT_size is not None:
            try:
                # helper conversions
                def chw_to_hwc(arr):
                    if isinstance(arr, np.ndarray) and arr.ndim == 3:
                        return np.transpose(arr, (1, 2, 0))
                    return arr

                def hwc_to_chw(arr):
                    if isinstance(arr, np.ndarray) and arr.ndim == 3:
                        return np.transpose(arr, (2, 0, 1))
                    return arr

                hr_hwc = chw_to_hwc(hr)
                lr_hwc = chw_to_hwc(lr)

                # cv2.resize expects (width, height)
                hr_hwc = cv2.resize(hr_hwc, (GT_size, GT_size), interpolation=cv2.INTER_CUBIC)
                lq_size = max(1, GT_size // scale)
                lr_hwc = cv2.resize(lr_hwc, (lq_size, lq_size), interpolation=cv2.INTER_AREA)

                # back to CHW
                hr = hwc_to_chw(hr_hwc)
                lr = hwc_to_chw(lr_hwc)
            except Exception as e2:
                # Provide full debug info in the raised error so you can identify the bad sample quickly
                raise RuntimeError(f"[Dataset] Fatal: couldn't normalize sample sizes for index {index} (GT_path={GT_path}, LQ_path={LQ_path}). "
                                   f"hr shape before/after: {getattr(self, 'last_hr_shape', None)} -> {getattr(hr, 'shape', None)}; "
                                   f"lr shape: {getattr(lr, 'shape', None)}; underlying error: {e2}")

        # ensure integer shapes and infer scale if needed
        if self.scale is None:
            # safe check to avoid zero-division or dimension errors
            try:
                self.scale = int(hr.shape[1] // lr.shape[1])
            except Exception as e:
                raise RuntimeError(f"[Dataset] Couldn't infer scale for index {index} (GT_path={GT_path}, LQ_path={LQ_path}): hr.shape={getattr(hr,'shape',None)}, lr.shape={getattr(lr,'shape',None)}; {e}")

        assert hr.shape[1] == self.scale * lr.shape[1], ('non-fractional ratio or incorrect scale', lr.shape, hr.shape)

        if self.alignment is None:
            self.alignment = self.target_scale if self.target_scale else self.scale

        # -------------------- ENSURE CHW numpy arrays before spatial ops --------------------
        def _ensure_chw_array(x):
            """Return numpy array with shape (C, H, W). Accepts torch.Tensor, HxW, HxWxC, CxHxW."""
            # convert torch -> numpy
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
            if not isinstance(x, np.ndarray):
                raise RuntimeError(f"Unsupported image type: {type(x)}")

            # H x W -> 1 x H x W
            if x.ndim == 2:
                return x[np.newaxis, ...]
            # H x W x C -> C x H x W when last dim is channels
            if x.ndim == 3:
                # if the first dim looks like channels (1 or 3), assume CHW already
                if x.shape[0] in (1, 3):
                    return x
                # otherwise if last dim is 1 or 3, assume HWC -> transpose
                if x.shape[2] in (1, 3):
                    return np.transpose(x, (2, 0, 1))
                # otherwise, try to guess: if middle dim very small, it might be CHW transposed
                # fallback: treat as CHW if plausible
                if x.shape[0] not in (1, 3) and x.shape[2] not in (1, 3):
                    # prefer keeping as-is and let next checks fail clearly
                    return x
            raise RuntimeError(f"Image has unsupported shape: {x.shape}")

        # enforce shapes and provide debug info if wrong
        try:
            hr = _ensure_chw_array(hr)
            lr = _ensure_chw_array(lr)
        except Exception as e:
            # helpful debug print and re-raise so you can find the problematic sample
            print(f"[Dataset][ERROR] index={index} GT_path={GT_path} LQ_path={LQ_path} -- shape/type problem: hr_type={type(hr)}, hr_shape={getattr(hr,'shape',None)}; lr_type={type(lr)}, lr_shape={getattr(lr,'shape',None)}; error={e}")
            raise

        # Now shapes are (C,H,W). Verify dims are present for random_crop
        if hr.ndim != 3 or lr.ndim != 3:
            print(f"[Dataset][ERROR] index={index} GT_path={GT_path} LQ_path={LQ_path} -- unexpected dims after ensure: hr.ndim={getattr(hr,'ndim',None)}, lr.ndim={getattr(lr,'ndim',None)}")
            raise RuntimeError(f"[Dataset] Unexpected image dims (need CHW): hr.ndim={getattr(hr,'ndim',None)}, lr.ndim={getattr(lr,'ndim',None)}")

        # Proceed with crops / flips / rotations (same API as before)
        if self.use_crop:
            hr, lr = random_crop(hr, lr, self.crop_size, self.scale, self.use_crop, alignment=self.alignment)

        if self.center_crop_hr_size:
            hr = center_crop(hr, self.center_crop_hr_size, alignment=self.alignment)
            lr = center_crop(lr, self.center_crop_hr_size // self.scale, alignment=self.alignment // self.scale)

        if self.use_flip:
            hr, lr = random_flip(hr, lr)

        if self.use_rot:
            hr, lr = random_rotation(hr, lr)
        # -------------------------------------------------------------------------------


        if self.target_scale is not None and self.scale != self.target_scale:
            rescale = self.target_scale // self.scale
            lr = lr[:, :(lr.shape[1] // rescale) * rescale, :(lr.shape[2] // rescale) * rescale]
            hr = hr[:, :(hr.shape[1] // self.target_scale) * self.target_scale, :(hr.shape[2] // self.target_scale) * self.target_scale]

            lr = np.transpose(lr, [1, 2, 0])
            lr = matlablike_resize.imresize(lr, 1 / rescale)
            lr = np.transpose(lr, [2, 0, 1])

            assert hr.shape[1] == self.target_scale * lr.shape[1], ('non-fractional ratio', lr.shape, hr.shape)

        # scale to [0,1] and convert to torch
        hr = hr.astype(np.float32) / 255.0
        lr = lr.astype(np.float32) / 255.0

        hr = torch.Tensor(hr)
        lr = torch.Tensor(lr)

        if self.y_labels is not None:
            y_label = self.y_labels[index]

            if self.normalize and y_label == 1:
                hr = torch.clamp(((hr - self.mean_noisy_hr) / self.std_noisy_hr) * self.std_clean_hr + self.mean_clean_hr, 0, 1)
                lr = torch.clamp(((lr - self.mean_noisy_lr) / self.std_noisy_lr) * self.std_clean_lr + self.mean_clean_lr, 0, 1)

            return {'LQ': lr,
                    'GT': hr,
                    'y_label': y_label,
                    'LQ_path': LQ_path,
                    'GT_path': GT_path}

        return {'LQ': lr, 'GT': hr, 'LQ_path': LQ_path, 'GT_path': GT_path}


    def __len__(self):
        return len(self.paths_GT)

