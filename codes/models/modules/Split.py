import torch
from torch import nn as nn

from models.modules import thops
from models.modules.FlowStep import FlowStep
from models.modules.flow import Conv2dZeros, GaussianDiag
from models.modules import flow
from utils.util import opt_get

import numpy as np


class Split2d(nn.Module):
    def __init__(self, num_channels, logs_eps=0, cond_channels=0, position=None, consume_ratio=0.5, opt=None):
        super().__init__()

        self.num_channels_consume = int(round(num_channels * consume_ratio))
        self.num_channels_pass = num_channels - self.num_channels_consume

        self.conv = Conv2dZeros(in_channels=self.num_channels_pass + cond_channels,
                                out_channels=self.num_channels_consume * 2)
        self.logs_eps = opt_get(opt, ['network_G', 'flow', 'split', 'eps'],  logs_eps)
        self.position = position
        self.opt = opt

        C = self.num_channels_consume

        # parameters to model the domain shift
        # domain X is N(0,1)
        # domain Y is N(mean_shift, cov_shift @ cov_shift^T)
        self.I = torch.nn.Parameter(torch.eye(C, requires_grad=False), requires_grad=False)
        self.mean_shift = torch.nn.Parameter(torch.zeros(C,requires_grad=True), requires_grad=True) 
        std_init_shift = opt_get(self.opt, ['network_G', 'flow', 'shift', 'std_init_shift'], 1.0)
        self.cov_shift = torch.nn.Parameter(torch.eye(C,requires_grad=True) * std_init_shift, 
                                            requires_grad=True)

        self.register_parameter(f"mean_shift", self.mean_shift)
        self.register_parameter(f"cov_shift", self.cov_shift)


    def split2d_prior(self, z, ft=None):
        """
        Robust split-prior computation:
        - compute runtime split (z1, z2) from input z
        - build prior input from z1 (+ ft if applicable)
        - ensure self.conv has in_channels == prior_in.shape[1] and
          out_channels == z2.shape[1] * 2; rebuild if necessary
        - compute h = self.conv(prior_in) and return mean, logs (each with channels == z2.shape[1])
        """
        # 1) compute z1, z2 using runtime-safe split logic (mirrors split_ratio)
        in_ch = int(z.shape[1])
        nominal_pass = getattr(self, 'num_channels_pass', None)
        if nominal_pass is None:
            nominal_pass = in_ch // 2

        if nominal_pass >= in_ch:
            fallback_pass = in_ch // 2
            if fallback_pass == 0:
                raise RuntimeError(f"Split2d: cannot split tensor with {in_ch} channel(s).")
            num_pass = fallback_pass
            print(f"[WARN] Split2d.split2d_prior: nominal num_channels_pass={nominal_pass} >= in_ch={in_ch}. "
                f"Falling back to num_pass={num_pass}.")
        else:
            num_pass = nominal_pass
            if num_pass == in_ch:
                num_pass = in_ch // 2
                print(f"[WARN] Split2d.split2d_prior: adjusted num_pass to {num_pass} to avoid empty z2.")

        z1 = z[:, :num_pass]
        z2 = z[:, num_pass:]

        # 2) build prior input: usually z1 concatenated with ft (if ft and layer expects it)
        if ft is not None and getattr(self, 'position', None) is not None:
            # ensure ft spatial size matches z1
            if ft.shape[2:] != z1.shape[2:]:
                ft = torch.nn.functional.interpolate(ft, size=z1.shape[2:], mode='bilinear', align_corners=False)
            prior_in = torch.cat([z1, ft], dim=1)
        else:
            prior_in = z1

        # 3) desired conv output must produce channels == z2.channels * 2 (for mean+logs via split)
        desired_out_ch = int(z2.shape[1]) * 2
        runtime_in_ch = int(prior_in.shape[1])

        # 4) check current conv shapes (best-effort)
        try:
            conv_in_ch = getattr(self.conv, 'in_channels', None)
            if conv_in_ch is None:
                conv_in_ch = self.conv.weight.shape[1]
        except Exception:
            conv_in_ch = None

        try:
            conv_out_ch = getattr(self.conv, 'out_channels', None)
            if conv_out_ch is None:
                conv_out_ch = self.conv.weight.shape[0]
        except Exception:
            conv_out_ch = None

        # 5) If conv shapes don't match runtime needs, rebuild it to match runtime sizes
        if conv_in_ch != runtime_in_ch or conv_out_ch != desired_out_ch:
            # try to reuse original conv class if present
            ConvCls = self.conv.__class__ if hasattr(self, 'conv') and self.conv is not None else None
            device = prior_in.device
            dtype = prior_in.dtype

            # Try to infer kernel size from existing conv if possible
            kH = kW = 3
            try:
                w = self.conv.weight
                kH, kW = w.shape[2], w.shape[3]
            except Exception:
                pass

            new_conv = None
            if ConvCls is not None:
                # try to instantiate with the common repo signature
                try:
                    new_conv = ConvCls(runtime_in_ch, desired_out_ch, kernel_size=[kH, kW])
                except Exception:
                    try:
                        # fallback to torch.nn.Conv2d
                        new_conv = torch.nn.Conv2d(runtime_in_ch, desired_out_ch, kernel_size=(kH, kW),
                                               padding=(kH // 2, kW // 2))
                    except Exception:
                        new_conv = None
            else:
                try:
                    new_conv = torch.nn.Conv2d(runtime_in_ch, desired_out_ch, kernel_size=(kH, kW),
                                           padding=(kH // 2, kW // 2))
                except Exception:
                    new_conv = None

            if new_conv is None:
                raise RuntimeError(f"Split2d: could not rebuild conv for prior (in_ch={runtime_in_ch}, out_ch={desired_out_ch}).")

            # move to correct device/dtype
            try:
                new_conv.to(device=device, dtype=dtype)
            except Exception:
                new_conv.to(device)

            # replace and register
            self.conv = new_conv
            print(f"[WARN] Split2d: rebuilt prior conv -> in_ch={runtime_in_ch}, out_ch={desired_out_ch}, kernel=({kH},{kW}), device={device}")

        # 6) compute prior and split into mean/logs
        h = self.conv(prior_in)
        mean, logs = thops.split_feature(h, "split")

        # --- defensive spatial alignment ---
        if mean.shape[2:] != z2.shape[2:]:
            target_size = (z2.shape[2], z2.shape[3])
            mean = torch.nn.functional.interpolate(mean, size=target_size, mode='bilinear', align_corners=False)
            logs = torch.nn.functional.interpolate(logs, size=target_size, mode='bilinear', align_corners=False)
            print(f"[WARN] Split2d: interpolated prior mean/logs from spatial {h.shape[2:]} -> {target_size} to match z2.")

        # final sanity check
        if mean.shape[1] != z2.shape[1] or logs.shape[1] != z2.shape[1]:
            raise AssertionError(
                f"Split2d: prior produced mean/logs channels ({mean.shape[1]},{logs.shape[1]}) "
                f"but z2 channels = {z2.shape[1]}"
            )

        return mean, logs




    def exp_eps(self, logs):
        if opt_get(self.opt, ['network_G', 'flow', 'split', 'tanh'], False):
            return torch.exp(torch.tanh(logs)) + self.logs_eps

        return torch.exp(logs) + self.logs_eps

    def forward(self, input, logdet=0., reverse=False, eps_std=None, eps=None, ft=None, y_onehot=None):
        domX = y_onehot == 0
        domY = y_onehot == 1

        if type(y_onehot) != torch.Tensor:
            y_onehot = torch.Tensor(y_onehot)

        assert len(y_onehot.shape) == 1, 'labels must be one dimensional'

        if not reverse:
            self.input = input

            z1, z2 = self.split_ratio(input)

            self.z1 = z1
            self.z2 = z2

            mean, logs = self.split2d_prior(z1, ft)
            
            eps = (z2 - mean) / (self.exp_eps(logs) + 1e-6)

            assert eps.shape[0] == y_onehot.shape[0], 'need one class label per datapoint'

            cov_shifted = self.I + torch.matmul(self.cov_shift, self.cov_shift.T)
            mean_shifted = self.mean_shift

            ll = torch.zeros(eps.shape[0], device=eps.device)
            ll[domX] = flow.Gaussian.logp(None, None, eps[domX])
            ll[domY] = flow.Gaussian.logp(mean=mean_shifted, cov=cov_shifted, x=eps[domY])

            logdet = logdet + ll - logs.sum(dim=[1,2,3])

            return z1, logdet, eps
        else:
            z1 = input
            mean, logs = self.split2d_prior(z1, ft)

            if eps is None: # sample eps
                eps = GaussianDiag.sample_eps(mean.shape, eps_std)
                eps = eps.to(mean.device)
                
                shape = mean.shape

                if domY.any(): # sample and add u
                    z_noise = flow.GaussianDiag.sample_eps(shape, eps_std)[domY].to(mean.device)
                    eps[domY] = (eps[domY] + self.mean_shift.reshape(1,self.mean_shift.shape[0],1,1) 
                                   + torch.matmul(self.cov_shift, z_noise.T).T)
                    
                
            else:
                eps = eps.to(mean.device)

            assert eps.shape[0] == y_onehot.shape[0], 'need one class label per datapoint'

            cov_shifted = self.I + torch.matmul(self.cov_shift, self.cov_shift.T)
            mean_shifted = self.mean_shift

            ll = torch.zeros(eps.shape[0], device=eps.device)

            ll[domX] = flow.Gaussian.logp(None, None, eps[domX])
            ll[domY] = flow.Gaussian.logp(mean=mean_shifted, cov=cov_shifted, x=eps[domY])

            z2 = mean + self.exp_eps(logs) * eps

            z = thops.cat_feature(z1, z2)

            logdet = logdet - ll + logs.sum(dim=[1,2,3])

            return z, logdet

    def split_ratio(self, input):
        """
        Robust splitting by runtime channel count. Uses `self.num_channels_pass` if set,
        otherwise falls back to a half-split. Ensures z2 is not empty and prints warnings
        when a fallback/adjustment happens.
        """
        # runtime input channels
        in_ch = int(input.shape[1])

        # nominal intended number of channels to pass (may be set at init)
        nominal_pass = getattr(self, 'num_channels_pass', None)

        # default to half if not provided
        if nominal_pass is None:
            nominal_pass = in_ch // 2

        # clamp/adjust to avoid empty slices
        if nominal_pass >= in_ch:
            fallback_pass = in_ch // 2
            if fallback_pass == 0:
                # cannot reasonably split a single-channel tensor in this design
                raise RuntimeError(
                    f"Split2d: cannot split tensor with {in_ch} channel(s). "
                    "Upstream produced too few channels for splitting."
                )
            print(f"[WARN] Split2d: nominal num_channels_pass={nominal_pass} >= runtime in_ch={in_ch}. "
                f"Falling back to half-split num_pass={fallback_pass}.")
            num_pass = fallback_pass
        else:
            num_pass = nominal_pass
            if num_pass == in_ch:
                # avoid producing empty z2
                num_pass = in_ch // 2
                print(f"[WARN] Split2d: adjusted num_pass to {num_pass} to avoid empty z2 (runtime in_ch={in_ch}).")

        # perform split using the resolved num_pass
        z1 = input[:, :num_pass]
        z2 = input[:, num_pass:]

        # sanity check
        if z1.shape[1] + z2.shape[1] != in_ch:
            raise RuntimeError(
                f"Split2d split failed: in_ch={in_ch}, z1={z1.shape[1]}, z2={z2.shape[1]}"
            )

        return z1, z2
