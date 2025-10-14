import torch
from torch import nn as nn

from models.modules import thops


class _ActNorm(nn.Module):
    """
    Activation Normalization
    Initialize the bias and scale with a given minibatch,
    so that the output per-channel have zero mean and unit variance for that.

    After initialization, `bias` and `logs` will be trained as parameters.
    """

    def __init__(self, num_features, scale=1.):
        super().__init__()
        # register mean and scale
        size = [1, num_features, 1, 1]
        self.register_parameter("bias", nn.Parameter(torch.zeros(*size)))
        self.register_parameter("logs", nn.Parameter(torch.zeros(*size)))
        self.num_features = num_features
        self.scale = float(scale)
        self.inited = False

    def _check_input_dim(self, input):
        return NotImplemented

    def initialize_parameters(self, input):
        # Attempt to adapt stored params if model was built for RGB but input is grayscale.
        input_channels = input.size(1)

        # If shapes differ, try adaptation before asserting in _check_input_dim
        if hasattr(self, 'num_features') and self.num_features != input_channels:
            # Case: weights were created for RGB triplets (num_features = 3 * base),
            # but runtime input is grayscale (base). Collapse triplets -> single channel by averaging.
            if self.num_features == input_channels * 3:
                try:
                    with torch.no_grad():
                        dev = None
                        dtype = None
                        if hasattr(self, 'bias') and self.bias is not None:
                            try:
                                dev = self.bias.device
                                dtype = self.bias.dtype
                            except Exception:
                                dev = input.device
                                dtype = input.dtype
                        else:
                            dev = input.device
                            dtype = input.dtype

                        def _collapse_param(p):
                            """Collapse a per-channel parameter (1 x N x 1 x 1 or N) into (1 x base x 1 x 1)."""
                            if p is None:
                                return None
                            arr = p.data
                            # flatten to 1D
                            arr_flat = arr.view(-1).to(device=dev, dtype=dtype)
                            # reshape to (base, 3) and average
                            base = input_channels
                            try:
                                collapsed = arr_flat.view(base, 3).mean(dim=1)  # shape (base,)
                            except Exception:
                                # attempt alternative reshape if arr had extra dims
                                collapsed = arr_flat.contiguous().view(base, 3).mean(dim=1)
                            # return shaped as (1, base, 1, 1)
                            return collapsed.view(1, base, 1, 1).to(device=dev, dtype=dtype)

                        # Collapse common per-channel parameters if present
                        if hasattr(self, 'bias') and self.bias is not None:
                            self.bias = torch.nn.Parameter(_collapse_param(self.bias))
                        if hasattr(self, 'logs') and self.logs is not None:
                            self.logs = torch.nn.Parameter(_collapse_param(self.logs))
                        # If running_mean / running_var exist as buffers, collapse them too
                        if hasattr(self, 'running_mean') and getattr(self, 'running_mean') is not None:
                            try:
                                self.running_mean = _collapse_param(self.running_mean)
                            except Exception:
                                self.running_mean = self.running_mean  # leave as-is if we can't adapt
                        if hasattr(self, 'running_var') and getattr(self, 'running_var') is not None:
                            try:
                                self.running_var = _collapse_param(self.running_var)
                            except Exception:
                                self.running_var = self.running_var

                        old_nf = self.num_features
                        self.num_features = input_channels
                        try:
                            print(f"[ActNorm] Adapted parameters from {old_nf} -> {self.num_features} channels (RGB->grayscale collapse).")
                        except Exception:
                            pass
                except Exception as e:
                    raise AssertionError(f"[ActNorm] Parameter adaptation failed: {e}")
        # else: not an RBC->grayscale case; keep original behavior and let _check_input_dim assert

        # Now call the original dimension check (will assert if still mismatched)
        self._check_input_dim(input)

        # Original initialization logic follows
        if not self.training:
            return
        # If bias already non-zero, assume inited
        try:
        # works when bias shape is (1,C,1,1)
            if (self.bias != 0).any():
                self.inited = True
                return
        except Exception:
        # conservative fallback: if any issue checking bias, proceed with init
         pass

        assert input.device == self.bias.device, (input.device, self.bias.device)
        with torch.no_grad():
            bias = thops.mean(input.clone(), dim=[0, 2, 3], keepdim=True) * -1.0
            vars = thops.mean((input.clone() + bias) ** 2, dim=[0, 2, 3], keepdim=True)
            logs = torch.log(self.scale / (torch.sqrt(vars) + 1e-6))
            # copy into parameters (shapes should match now)
            self.bias.data.copy_(bias.data)
            self.logs.data.copy_(logs.data)
            self.inited = True


    def _center(self, input, reverse=False, offset=None):
        bias = self.bias

        if offset is not None:
            bias = bias + offset

        if not reverse:
            return input + bias
        else:
            return input - bias

    def _scale(self, input, logdet=None, reverse=False, offset=None):
        logs = self.logs

        if offset is not None:
            logs = logs + offset

        if not reverse:
            input = input * torch.exp(logs) # should have shape batchsize, n_channels, 1, 1
            # input = input * torch.exp(logs+logs_offset)
        else:
            input = input * torch.exp(-logs)
        if logdet is not None:
            """
            logs is log_std of `mean of channels`
            so we need to multiply pixels
            """
            dlogdet = thops.sum(logs) * thops.pixels(input)
            if reverse:
                dlogdet *= -1
            logdet = logdet + dlogdet
        return input, logdet

    def forward(self, input, logdet=None, reverse=False, offset_mask=None, logs_offset=None, bias_offset=None):
        if not self.inited:
            self.initialize_parameters(input)
        self._check_input_dim(input)

        if offset_mask is not None:
            logs_offset *= offset_mask
            bias_offset *= offset_mask
        # no need to permute dims as old version
        if not reverse:
            # center and scale

            # self.input = input
            input = self._center(input, reverse, bias_offset)
            input, logdet = self._scale(input, logdet, reverse, logs_offset)
        else:
            # scale and center
            input, logdet = self._scale(input, logdet, reverse, logs_offset)
            input = self._center(input, reverse, bias_offset)
        return input, logdet


class ActNorm2d(_ActNorm):
    def __init__(self, num_features, scale=1.):
        super().__init__(num_features, scale)

    def _check_input_dim(self, input):
        assert len(input.size()) == 4
        assert input.size(1) == self.num_features, (
            "[ActNorm]: input should be in shape as `BCHW`,"
            " channels should be {} rather than {}".format(
                self.num_features, input.size()))


class MaskedActNorm2d(ActNorm2d):
    def __init__(self, num_features, scale=1.):
        super().__init__(num_features, scale)

    def forward(self, input, mask, logdet=None, reverse=False):

        assert mask.dtype == torch.bool
        output, logdet_out = super().forward(input, logdet, reverse)

        input[mask] = output[mask]
        logdet[mask] = logdet_out[mask]

        return input, logdet



class ConditionalActNorm2d(nn.Module):
    """
    Conditional Activation Normalization
    Initialize the bias and scale with a given minibatch,
    so that the output per-channel have zero mean and unit variance for that.

    After initialization, `bias` and `logs` will be trained as parameters.
    """

    def _center(self, input, bias, reverse=False):
        if not reverse:
            return input + bias
        else:
            return input - bias

    def _scale(self, input, scale_logs, logdet=None, reverse=False):
        if not reverse:
            input = input * torch.exp(scale_logs)
        else:
            input = input * torch.exp(-scale_logs)
        if logdet is not None:
            """
            logs is log_std of `mean of channels`
            so we need to multiply pixels
            """
            dlogdet = thops.sum(scale_logs, dim=(1, 2, 3))
            if scale_logs.shape[2:] == (1, 1):
                dlogdet = dlogdet * thops.pixels(input)
            if reverse:
                dlogdet = -dlogdet
            logdet = logdet + dlogdet
        return input, logdet

    def forward(self, input, weight, logdet=None, reverse=False):
        if weight.dim() == 2:
            weight = weight.view(*weight.shape, 1, 1)
        assert input.shape[0] == weight.shape[0]
        assert input.shape[1] * 2 == weight.shape[1]
        assert input.shape[2:] == weight.shape[2:] or weight.shape[2:] == (1, 1)

        split_ind = weight.shape[1] // 2
        if not reverse:
            # center and scale
            input = self._center(input, weight[:, :split_ind, ...], reverse)
            input, logdet = self._scale(input, weight[:, split_ind:, ...], logdet, reverse)
        else:
            # scale and center
            input, logdet = self._scale(input, weight[:, split_ind:, ...], logdet, reverse)
            input = self._center(input, weight[:, :split_ind, ...], reverse)
        return input, logdet


class ConditionalActNormImageInjector(nn.Module):
    def __init__(self, in_channels, out_channels, scale=1, mode='scalar'):
        super().__init__()
        self.NN = self.f_lin(in_channels, out_channels * 2)
        self.scale = scale
        self.mode = mode
        self.conditionalActNorm2d = ConditionalActNorm2d()

    def forward(self, input, img_ft, logdet=None, reverse=False):
        if not reverse:
            return self.normal_flow(input, img_ft, logdet=logdet)
        else:
            return self.reverse_flow(input, img_ft, logdet=logdet)

    def normal_flow(self, z, ft, logdet):
        weights = self.get_weights(ft)
        if self.mode == 'scalar': assert weights.shape[2] == 1
        if self.mode == 'scalar': assert weights.shape[3] == 1
        z, logdet = self.conditionalActNorm2d(z, weights, logdet, reverse=False)
        return z, logdet

    def reverse_flow(self, z, ft, logdet):
        weights = self.get_weights(ft)
        z, logdet = self.conditionalActNorm2d(z, weights, reverse=True)
        return z, logdet

    def get_weights(self, img_ft):
        if self.mode == 'scalar':
            return self.NN(img_ft).mean([2, 3], keepdim=True)
        elif self.mode == 'fullRes':
            return self.NN(img_ft)
        else:
            raise RuntimeError("Mode not found")

    @staticmethod
    def f_lin(in_channels, out_channels):
        return nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                            kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True))