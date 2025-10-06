

# deflow_reference.py
"""
Reference PyTorch implementation of DeFlow's conditional flow model
(adapted to single-channel CT slices). Use as a starting point and
adapt to your data and compute environment.

Notes:
- This code prioritizes readability & fidelity to the paper over raw performance.
- For multi-GPU it uses DistributedDataParallel (launch with torch.distributed.run).
- Expected Python packages: torch, torchvision, numpy, pillow
"""

import os
import math
import argparse
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from PIL import Image
import numpy as np

# -----------------------
# Utils / Conditioning h
# -----------------------
def domain_invariant_h(img: torch.Tensor, downsample_factor: int = 8, sigma_noise: float = 0.01):
    """
    Compute h(x) for domain invariant conditioning.
    img: tensor shape (B, C=1, H, W), float in [0,1] or normalized
    For CT we perform bicubic downsample (via interpolate) and add small gaussian noise.
    """
    B, C, H, W = img.shape
    newH, newW = H // downsample_factor, W // downsample_factor
    # bicubic downsample implemented via interpolate
    h = F.interpolate(img, size=(newH, newW), mode='bicubic', align_corners=False)
    if sigma_noise > 0:
        h = h + torch.randn_like(h) * sigma_noise
    return h

# -----------------------
# Data placeholders
# -----------------------
class SingleSliceCTDataset(Dataset):
    """
    Placeholder dataset loading single-slice CT images from a folder.
    Expects images (png, jpg) or .npy arrays. Returns single-channel float32 [0,1].
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.files = [os.path.join(root_dir, f) for f in sorted(os.listdir(root_dir))
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.npy', '.npz'))]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def _load_file(self, path):
        if path.endswith('.npy') or path.endswith('.npz'):
            arr = np.load(path)
            if 'arr_0' in arr:
                arr = arr['arr_0']
            # If multi-slice, choose middle or first channel
            if arr.ndim == 3:
                arr = arr[arr.shape[0] // 2]
            img = Image.fromarray(arr.astype(np.float32))
        else:
            img = Image.open(path).convert('L')
        return img

    def __getitem__(self, idx):
        path = self.files[idx]
        img = self._load_file(path)
        # convert to float tensor
        arr = np.array(img).astype(np.float32)
        # normalize to [0,1]
        if arr.max() > 0:
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        else:
            arr = arr
        t = torch.from_numpy(arr).unsqueeze(0)  # (1,H,W)
        if self.transform:
            t = self.transform(t)
        return t

# -----------------------
# Flow building blocks
# -----------------------
class ActNorm(nn.Module):
    """Per-channel affine normalization (Glow)"""
    def __init__(self, num_channels):
        super().__init__()
        self.initialized = False
        self.loc = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_channels, 1, 1))

    def initialize(self, x):
        # data-based init: set loc/scale s.t. outputs have zero mean and unit var per channel
        with torch.no_grad():
            mean = x.mean(dim=[0, 2, 3], keepdim=True)
            std = x.std(dim=[0, 2, 3], keepdim=True)
            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1.0 / (std + 1e-6))
            self.initialized = True

    def forward(self, x, logdet=None, reverse=False):
        if not self.initialized and not reverse:
            self.initialize(x)
        if reverse:
            x = (x - self.loc) / (self.scale + 1e-6)
            if logdet is not None:
                # reverse: subtract logdet
                B, C, H, W = x.shape
                ld = -torch.sum(torch.log(torch.abs(self.scale))) * H * W
                return x, logdet + ld
            return x
        else:
            x = self.scale * x + self.loc
            if logdet is not None:
                B, C, H, W = x.shape
                ld = torch.sum(torch.log(torch.abs(self.scale))) * H * W
                return x, logdet + ld
            return x

class Invertible1x1Conv(nn.Module):
    """Invertible 1x1 conv with LU decomposition for speed/stability"""
    def __init__(self, num_channels):
        super().__init__()
        w = torch.qr(torch.randn(num_channels, num_channels))[0]  # random orthogonal init
        P, L, U = torch.lu_unpack(*w.lu())
        # We will param as P, L (mask lower), U (masked upper with diag)
        self.register_buffer('P', P)
        self.L = nn.Parameter(L)
        self.U = nn.Parameter(U)
        # store sign and log of diagonal of U for stable determinant
        s = torch.diag(self.U)
        self.log_s = nn.Parameter(torch.log(torch.abs(s)))
        self.sign_s = nn.Parameter(torch.sign(s))
        # zero diagonal in U (we'll incorporate diag separately)
        with torch.no_grad():
            self.U.fill_diagonal_(0.)

    def get_weight(self):
        # reconstruct W = P @ L @ (diag(sign*exp(log_s)) + U)
        L = torch.tril(self.L, -1) + torch.eye(self.L.size(0), device=self.L.device)
        U = torch.triu(self.U, 1) + torch.diag(self.sign_s * torch.exp(self.log_s))
        W = self.P @ (L @ U)
        return W

    def forward(self, x, logdet=None, reverse=False):
        B, C, H, W = x.shape
        weight = self.get_weight().view(C, C, 1, 1)
        if reverse:
            inv_weight = torch.inverse(self.get_weight()).view(C, C, 1, 1)
            x = F.conv2d(x, inv_weight)
            if logdet is not None:
                ld = -H * W * torch.sum(self.log_s)
                return x, logdet + ld
            return x
        else:
            x = F.conv2d(x, weight)
            if logdet is not None:
                ld = H * W * torch.sum(self.log_s)
                return x, logdet + ld
            return x

class AffineCoupling(nn.Module):
    """
    Conditional affine coupling block.
    Splits channels into a1 (conditioning) and a2 (transformed).
    Uses a small conv-net to predict scale and shift for a2 from a1 and condition h.
    """
    def __init__(self, in_channels, hidden_channels=128, cond_channels=16):
        super().__init__()
        assert in_channels % 2 == 0, "in_channels must be divisible by 2 for splitting"
        self.in_channels = in_channels
        self.split = in_channels // 2
        # a simple network to predict scale and shift
        inp_ch = self.split + cond_channels
        self.net = nn.Sequential(
            nn.Conv2d(inp_ch, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, self.split * 2, kernel_size=3, padding=1)
        )
        # initialize last conv to zero to start as identity
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, h_cond, logdet=None, reverse=False):
        # x: (B, C, H, W)
        a1, a2 = x[:, :self.split], x[:, self.split:]
        # broadcast/resize h_cond if needed
        if h_cond is not None:
            # h_cond expected (B, Cc, h, w) — we upsample to match a1 spatial
            if h_cond.shape[2:] != a1.shape[2:]:
                h_resized = F.interpolate(h_cond, size=a1.shape[2:], mode='bilinear', align_corners=False)
            else:
                h_resized = h_cond
            net_in = torch.cat([a1, h_resized], dim=1)
        else:
            net_in = a1
        st = self.net(net_in)
        s, t = st.chunk(2, dim=1)
        # scale parametrization: make scale small initially
        s = torch.tanh(s) * 0.9  # ensures stability
        if reverse:
            a2 = (a2 - t) * torch.exp(-s)
            x = torch.cat([a1, a2], dim=1)
            if logdet is not None:
                B, C2, H, W = a2.shape
                ld = -torch.sum(s, dim=[1,2,3])
                return x, logdet + torch.sum(ld)
            return x
        else:
            a2 = a2 * torch.exp(s) + t
            x = torch.cat([a1, a2], dim=1)
            if logdet is not None:
                B, C2, H, W = a2.shape
                ld = torch.sum(s, dim=[1,2,3])
                return x, logdet + torch.sum(ld)
            return x

class AffineInjector(nn.Module):
    """
    Applies an affine transform whose parameters are computed from h(x).
    This influences all channels directly.
    """
    def __init__(self, in_channels, cond_channels=16):
        super().__init__()
        # cond -> 2*in_channels (s,b)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # pool cond spatially
            nn.Conv2d(cond_channels, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, in_channels * 2, kernel_size=1)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, h_cond, logdet=None, reverse=False):
        # h_cond expected (B,Cc,h,w) ; we pool to produce per-sample parameters
        if h_cond is None:
            # identity
            return (x, logdet) if logdet is not None else x
        params = self.net(h_cond)  # (B, 2*in_channels, 1, 1)
        s, b = params.chunk(2, dim=1)
        s = torch.tanh(s) * 0.9
        if reverse:
            x = (x - b) * torch.exp(-s)
            if logdet is not None:
                B, C, H, W = x.shape
                ld = -torch.sum(s) * H * W
                return x, logdet + ld
            return x
        else:
            x = x * torch.exp(s) + b
            if logdet is not None:
                B, C, H, W = x.shape
                ld = torch.sum(s) * H * W
                return x, logdet + ld
            return x

# -----------------------
# Flow step and full flow
# -----------------------
class FlowStep(nn.Module):
    def __init__(self, channels, cond_channels):
        super().__init__()
        self.actnorm = ActNorm(channels)
        self.invconv = Invertible1x1Conv(channels)
        self.coupling = AffineCoupling(channels, hidden_channels=max(128, channels*4), cond_channels=cond_channels)
        self.injector = AffineInjector(channels, cond_channels=cond_channels)

    def forward(self, x, h_cond, logdet=None, reverse=False):
        if reverse:
            x, logdet = self.injector(x, h_cond, logdet=logdet, reverse=True)
            x, logdet = self.coupling(x, h_cond, logdet=logdet, reverse=True)
            x, logdet = self.invconv(x, logdet=logdet, reverse=True)
            x, logdet = self.actnorm(x, logdet=logdet, reverse=True)
            return x, logdet
        else:
            x, logdet = self.actnorm(x, logdet=logdet, reverse=False)
            x, logdet = self.invconv(x, logdet=logdet, reverse=False)
            x, logdet = self.coupling(x, h_cond, logdet=logdet, reverse=False)
            x, logdet = self.injector(x, h_cond, logdet=logdet, reverse=False)
            return x, logdet

def squeeze2x(x):
    # Pixel-shuffle like squeeze: (B,C,H,W) -> (B,4C,H/2,W/2)
    B, C, H, W = x.shape
    assert H % 2 == 0 and W % 2 == 0
    x = x.view(B, C, H//2, 2, W//2, 2)
    x = x.permute(0,1,3,5,2,4).contiguous()
    x = x.view(B, C*4, H//2, W//2)
    return x

def unsqueeze2x(x):
    B, C, H, W = x.shape
    assert C % 4 == 0
    x = x.view(B, C//4, 2, 2, H, W)
    x = x.permute(0,1,4,2,5,3).contiguous()
    x = x.view(B, C//4, H*2, W*2)
    return x

class DeFlowNet(nn.Module):
    def __init__(self, in_channels=1, levels=3, K=8, cond_channels=16):
        super().__init__()
        self.levels = levels
        self.K = K
        self.cond_channels = cond_channels
        self.in_channels = in_channels

        # optional small encoder to produce cond_channels from h(x)
        # a simple conv stack that turns small h(x) into cond feature maps
        self.cond_net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, cond_channels, kernel_size=3, padding=1)
        )

        # build levels
        self.levels_modules = nn.ModuleList()
        current_channels = in_channels
        for lvl in range(levels):
            # each level: squeeze -> K flow steps -> split (we implement split inside forward)
            steps = nn.ModuleList([FlowStep(current_channels*4 if lvl> -1 else current_channels, cond_channels)
                                   for _ in range(K)])  # we'll handle squeeze first
            self.levels_modules.append(steps)
            # after squeeze channels multiply by 4
            current_channels = current_channels * 4
            # simulate split: we'll remove half channels at each level (common choice)
            current_channels = current_channels // 2

        # latent shift params mu_u (dimension D) and M (D x D) where D is channels of final latent
        self.latent_dim = current_channels  # after last split, this is the per-channel latent dim
        # We'll store MU and M for the *channel* dimension (spatial invariant)
        self.mu_u = nn.Parameter(torch.zeros(self.latent_dim))
        # for M parameterize lower triangular matrix for stability
        tril_indices = torch.tril_indices(self.latent_dim, self.latent_dim)
        self.register_buffer('tril_idx', tril_indices)
        # store vector for lower-tri elements (unconstrained)
        self.M_params = nn.Parameter(torch.randn(tril_indices.size(1)) * 1e-2)

    def build_M_matrix(self):
        D = self.latent_dim
        M = torch.zeros(D, D, device=self.M_params.device)
        M[self.tril_idx[0], self.tril_idx[1]] = self.M_params
        return M

    def encode_cond(self, h):
        # h is low-res; cond_net will produce small feature map to plug into each flow step
        return self.cond_net(h)

    def forward(self, x, h, reverse=False, zs_collect=None):
        """
        If reverse=False: encodes x -> returns z and sum logdet
        If reverse=True: decodes z -> returns x
        zs_collect: when decoding, list of latent tensors to be inserted at splits
        """
        if not reverse:
            # encoding path: produce h cond features and run levels
            cond = self.encode_cond(h) if h is not None else None
            logdet = torch.zeros(x.size(0), device=x.device)
            z_list = []
            a = x
            for lvl_idx, steps in enumerate(self.levels_modules):
                # squeeze
                a = squeeze2x(a)  # channels x4
                # run K steps
                for step in steps:
                    a, logdet = step(a, cond, logdet=logdet, reverse=False)
                # split: factor out half channels as latent
                B, C, H, W = a.shape
                C1 = C // 2
                z = a[:, :C1].contiguous()
                a = a[:, C1:].contiguous()
                z_list.append(z)
            # final 'a' is the remaining latent too
            z_list.append(a)
            # combine all z's by storing -> we'll return stacked (do not merge spatial dims)
            # For likelihood computation we will evaluate per-latent base distributions
            return z_list, logdet
        else:
            # reverse decoding: zs_collect must be provided as list of tensors in same order as z_list
            assert zs_collect is not None, "When reverse=True you must pass zs_collect (list of latents)."
            cond = self.encode_cond(h) if h is not None else None
            logdet = torch.zeros(zs_collect[0].size(0), device=zs_collect[0].device)
            # start from last latent
            a = zs_collect[-1]
            for lvl_idx in reversed(range(len(self.levels_modules))):
                # merge in the split latent for this level
                z_split = zs_collect[lvl_idx]
                a = torch.cat([z_split, a], dim=1)  # put split back in front
                # run flow steps in reverse order
                steps = self.levels_modules[lvl_idx]
                for step in reversed(steps):
                    a, logdet = step(a, cond, logdet=logdet, reverse=True)
                # unsqueeze
                a = unsqueeze2x(a)
            x_recon = a
            return x_recon, logdet

    # helper to flatten channel vectors for normal multivariate computations
    @staticmethod
    def flatten_channel_vectors(z):
        # z: list of latents each (B,C,H,W) -> we treat each spatial location as a sample of dimension C
        # Return concatenated (B * H * W, C_total) and sizes to reconstruct
        flat_list = []
        meta = []
        for t in z:
            B, C, H, W = t.shape
            flat = t.permute(0,2,3,1).reshape(-1, C)  # (B*H*W, C)
            flat_list.append(flat)
            meta.append((B, C, H, W))
        if len(flat_list) == 1:
            return flat_list[0], meta
        else:
            concat = torch.cat(flat_list, dim=1)  # concat on channel dim
            return concat, meta

    @staticmethod
    def unflatten_channel_vectors(flat, meta):
        # inverse of flatten: split flat by meta channel dims and reshape to list of tensors
        outs = []
        idx = 0
        for (B, C, H, W) in meta:
            chunk = flat[:, idx: idx + C]
            idx += C
            t = chunk.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            outs.append(t)
        return outs

# -----------------------
# Likelihood / loss functions
# -----------------------
def standard_normal_logprob(z):
    # z: (N, D)
    # log p = -0.5 * (D*log(2pi) + sum(z^2))
    D = z.shape[1]
    log_z2 = -0.5 * (z * z).sum(dim=1)
    const = -0.5 * D * math.log(2 * math.pi)
    return const + log_z2

def multivariate_normal_logpdf(z, mu, cov_cholesky):
    """
    z: (N, D)
    mu: (D,)
    cov_cholesky: lower-triangular L such that cov = L @ L^T. (D,D)
    Returns: (N,) logpdf under N(mu, cov)
    """
    # Solve (z - mu) under cov via cholesky solve
    # Use torch.linalg.solve_triangular if available
    centered = z - mu.unsqueeze(0)
    # solve L y = centered^T
    L = cov_cholesky
    # first solve L y = centered^T
    y = torch.triangular_solve(centered.t(), L, upper=False)[0]  # (D, N)
    quad = -0.5 * torch.sum(y * y, dim=0)  # (N,)
    # logdet = sum log diag(L)^2 = 2 sum log diag(L)
    logdet = 2.0 * torch.sum(torch.log(torch.diagonal(L)))
    const = -0.5 * z.shape[1] * math.log(2 * math.pi)
    return const - 0.5 * 0.0 + quad - 0.5 * logdet  # rearranged

# -----------------------
# Training loop & sampling
# -----------------------
def train(args):
    # distributed init
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)

    dist.init_process_group(backend='nccl', init_method='env://') if world_size > 1 else None

    # Datasets
    ds_clean = SingleSliceCTDataset(args.clean_dir)
    ds_degraded = SingleSliceCTDataset(args.degraded_dir)

    sampler_clean = DistributedSampler(ds_clean) if world_size > 1 else None
    sampler_degraded = DistributedSampler(ds_degraded) if world_size > 1 else None

    dl_clean = DataLoader(ds_clean, batch_size=args.batch_size, sampler=sampler_clean, shuffle=(sampler_clean is None), num_workers=4, pin_memory=True)
    dl_degraded = DataLoader(ds_degraded, batch_size=args.batch_size, sampler=sampler_degraded, shuffle=(sampler_degraded is None), num_workers=4, pin_memory=True)

    # model
    model = DeFlowNet(in_channels=1, levels=args.levels, K=args.K, cond_channels=args.cond_channels).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # training loop
    iters = 0
    iter_clean = iter(dl_clean)
    iter_degraded = iter(dl_degraded)
    while iters < args.total_iters:
        model.train()
        try:
            x = next(iter_clean)
        except StopIteration:
            iter_clean = iter(dl_clean)
            x = next(iter_clean)
        try:
            y = next(iter_degraded)
        except StopIteration:
            iter_degraded = iter(dl_degraded)
            y = next(iter_degraded)

        x = x.to(device)
        y = y.to(device)
        # build h(x), h(y)
        h_x = domain_invariant_h(x, downsample_factor=args.downsample_h, sigma_noise=args.h_noise).to(device)
        h_y = domain_invariant_h(y, downsample_factor=args.downsample_h, sigma_noise=args.h_noise).to(device)

        # Encode marginals: get z lists and logdet contributions
        z_x_list, logdet_x = model(x, h_x, reverse=False)  # list of tensors
        z_y_list, logdet_y = model(y, h_y, reverse=False)

        # For computing base logprobs: flatten channel vectors
        z_x_flat, meta = model.module.flatten_channel_vectors(z_x_list) if isinstance(model, DDP) else model.flatten_channel_vectors(z_x_list)
        z_y_flat, meta_y = model.module.flatten_channel_vectors(z_y_list) if isinstance(model, DDP) else model.flatten_channel_vectors(z_y_list)

        # p(x): base is standard normal (0,I)
        logp_x_base = standard_normal_logprob(z_x_flat)  # (N_samples,)
        # we need avg over spatial locations per batch: recall flatten was (B*H*W, D_concat)
        # but logdet_x was per-batch (B,)
        # To keep consistent, compute average per-image base logprob:
        # each image contributed (H*W) rows; group by image: splitted per-sample
        # We can sum logp_x_base per image and then divide by H*W to get per-image base.
        # Simpler: reshape and sum per image:
        B = x.size(0)
        # recover H*W product for first meta entry
        H_all = 1
        W_all = 1
        for (b, Cc, Hc, Wc) in meta:
            H_all *= 1  # placeholder - not used
        # We know flatten concatenated all spatial positions: so block_size = (z_x_flat.size(0) / B)
        block_size = z_x_flat.size(0) // B
        logp_x_base_per_image = logp_x_base.view(B, block_size).sum(dim=1)  # sum over spatial positions
        # combine: log p(x) = base + logdet
        logpx = logp_x_base_per_image + logdet_x  # (B,)

        # For y: base is N(mu_u, I + Sigma_u)
        # Build mu_u and cov cholesky
        if isinstance(model, DDP):
            mu_u = model.module.mu_u
            M = model.module.build_M_matrix()
        else:
            mu_u = model.mu_u
            M = model.build_M_matrix()
        # cov = I + Sigma_u where Sigma_u = M @ M^T
        Sigma = M @ M.t()
        cov = Sigma + torch.eye(Sigma.size(0), device=Sigma.device)
        # cholesky
        L = torch.linalg.cholesky(cov)  # (D,D)
        # compute logpdf
        logp_y_base = multivariate_normal_logpdf(z_y_flat, mu_u.to(z_y_flat.device), L)
        # group per-image
        block_size_y = z_y_flat.size(0) // B
        logp_y_base_per_image = logp_y_base.view(B, block_size_y).sum(dim=1)
        logpy = logp_y_base_per_image + logdet_y

        # negative log-likelihood loss (we minimize - (sum logpx + sum logpy) / (2*B))
        loss = - (logpx.mean() + logpy.mean()) * 0.5

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iters % args.log_interval == 0 and (not world_size or rank == 0):
            print(f"Iter {iters}: loss={loss.item():.4f}, -logpx={-logpx.mean().item():.4f}, -logpy={-logpy.mean().item():.4f}")

        if iters % args.save_interval == 0 and (not world_size or rank == 0):
            torch.save({
                'iter': iters,
                'model_state': model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
                'optimizer_state': optimizer.state_dict()
            }, os.path.join(args.checkpoint_dir, f"deflow_iter_{iters}.pt"))
            # sample a few outputs: sample y|x
            sample_and_save(model if not isinstance(model, DDP) else model.module, x[:2], h_x[:2], args.sample_dir, iters, device)

        iters += 1

def sample_and_save(model, x_batch, h_x_batch, out_dir, iter_idx, device):
    """Sample y ~ p(y|x) using z_x + u and decode"""
    model.eval()
    with torch.no_grad():
        z_x_list, _ = model(x_batch.to(device), h_x_batch.to(device), reverse=False)
        # flatten, then sample u_tilde, compute u = M u_tilde + mu_u, split back into z_list shapes, and decode
        flat_zx, meta = model.flatten_channel_vectors(z_x_list)
        B = x_batch.size(0)
        D = flat_zx.shape[1]
        # sample u_tilde per spatial location (B*H*W, D)
        u_tilde = torch.randn_like(flat_zx)
        M = model.build_M_matrix().to(flat_zx.device)
        mu_u = model.mu_u.to(flat_zx.device)
        u = (u_tilde @ M.t()) + mu_u.unsqueeze(0)
        z_y_flat = flat_zx + u
        # unflatten to z_y_list
        z_y_list = model.unflatten_channel_vectors(z_y_flat, meta)
        # decode z_y_list -> y
        y_recon, _ = model(None, None, reverse=True, zs_collect=z_y_list)  # NOTE: our forward API expects proper args; we call module directly
        # x_batch & y_recon are in [0,1] range (approximately). Save as pngs
        os.makedirs(out_dir, exist_ok=True)
        for i in range(B):
            arr = (y_recon[i,0].cpu().clamp(0,1).numpy() * 255).astype(np.uint8)
            Image.fromarray(arr).save(os.path.join(out_dir, f"sample_iter{iter_idx}_{i}.png"))

# -----------------------
# Argparse & main
# -----------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_dir', type=str, required=True)
    parser.add_argument('--degraded_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--total_iters', type=int, default=100000)
    parser.add_argument('--levels', type=int, default=3)
    parser.add_argument('--K', type=int, default=16)
    parser.add_argument('--cond_channels', type=int, default=16)
    parser.add_argument('--downsample_h', type=int, default=8)
    parser.add_argument('--h_noise', type=float, default=0.03)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--save_interval', type=int, default=2000)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--sample_dir', type=str, default='./samples')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)
    train(args)
