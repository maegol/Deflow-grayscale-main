import torch
from torch import nn as nn

from models.modules import thops
from models.modules.flow import Conv2d, Conv2dZeros
from utils.util import opt_get


def affine(x, scale, shift, reverse):
    if not reverse:
        x_shift = x + shift
        x_aff = x_shift * scale
        return x_aff
    else:
        x_aff = x
        x_shift = x_aff / scale
        x_reconst = x_shift - shift
        return x_reconst


class CondAffineCatAblation(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.need_features = True
        self.in_channels = in_channels
        self.in_channels_rrdb = 318
        self.kernel_hidden = 1
        self.affine_eps = 0.0001
        self.n_hidden_layers = 1
        self.hidden_channels = 64

        self.channels_for_nn = self.in_channels // 2
        self.channels_for_co = self.in_channels - self.channels_for_nn

        if self.channels_for_nn is None:
            self.channels_for_nn = self.in_channels // 2

        self.f = self.F(in_channels=self.channels_for_nn + self.in_channels_rrdb,
                        out_channels=self.channels_for_co * 2,
                        hidden_channels=self.hidden_channels,
                        kernel_hidden=self.kernel_hidden,
                        n_hidden_layers=self.n_hidden_layers)

    def forward(self, input: torch.Tensor, logdet=None, reverse=False, ft=None):
        if not reverse:
            z = input
            assert z.shape[1] == self.in_channels, (z.shape[1], self.in_channels)
            z1, z2 = self.split(z)
            scale, shift = self.feature_extract(z1, ft)

            self.asserts(scale, shift, z1, z2)

            z2 = z2 + shift
            z2 = z2 * scale

            logdet = self.get_logdet(logdet, scale)
            z = thops.cat_feature(z1, z2)
            output = z
        else:
            z = input
            z1, z2 = self.split(z)
            scale, shift = self.feature_extract(z1, ft)

            z2 = z2 / scale
            z2 = z2 - shift

            logdet = -self.get_logdet(logdet, scale)
            z = thops.cat_feature(z1, z2)
            output = z
        return output, logdet

    def asserts(self, scale, shift, z1, z2):
        assert z1.shape[1] == self.channels_for_nn, (z1.shape[1], self.channels_for_nn)
        assert z2.shape[1] == self.channels_for_co, (z2.shape[1], self.channels_for_co)
        assert scale.shape[1] == shift.shape[1], (scale.shape[1], shift.shape[1])
        assert scale.shape[1] == z2.shape[1], (scale.shape[1], z1.shape[1], z2.shape[1])

    def get_logdet(self, logdet, scale):
        logdet = thops.sum(torch.log(scale), dim=[1, 2, 3]) + logdet
        return logdet

    def feature_extract(self, z1, ft):
        z = torch.cat([z1, ft], dim=1)
        h = self.f(z)
        shift, scale = thops.split_feature(h, "cross")
        scale = (torch.sigmoid(scale + 2.) + self.affine_eps)
        return scale, shift

    def split(self, z):
        z1 = z[:, :self.channels_for_nn]
        z2 = z[:, self.channels_for_nn:]
        assert z1.shape[1] + z2.shape[1] == z.shape[1], (z1.shape[1], z2.shape[1], z.shape[1])
        return z1, z2

    def F(self, in_channels, out_channels, hidden_channels, kernel_hidden=1, n_hidden_layers=1):
        layers = [Conv2d(in_channels, hidden_channels), nn.ReLU(inplace=False)]

        for _ in range(n_hidden_layers):
            layers.append(Conv2d(hidden_channels, hidden_channels, kernel_size=[kernel_hidden, kernel_hidden]))
            layers.append(nn.ReLU(inplace=False))
        layers.append(Conv2dZeros(hidden_channels, out_channels))

        return nn.Sequential(*layers)


class CondAffineSeparated(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.need_features = True
        self.in_channels = in_channels
        self.in_channels_rrdb = 318
        self.kernel_hidden = 1
        self.affine_eps = 0.0001
        self.n_hidden_layers = 1
        self.hidden_channels = 64

        self.channels_for_nn = self.in_channels // 2
        self.channels_for_co = self.in_channels - self.channels_for_nn

        if self.channels_for_nn is None:
            self.channels_for_nn = self.in_channels // 2

        self.fAffine = self.F(in_channels=self.channels_for_nn,
                              out_channels=self.channels_for_co * 2,
                              hidden_channels=self.hidden_channels,
                              kernel_hidden=self.kernel_hidden,
                              n_hidden_layers=self.n_hidden_layers)

        self.fFeatures = self.F(in_channels=self.in_channels_rrdb,
                                out_channels=self.in_channels * 2,
                                hidden_channels=self.hidden_channels,
                                kernel_hidden=self.kernel_hidden,
                                n_hidden_layers=self.n_hidden_layers)

    def forward(self, input: torch.Tensor, logdet=None, reverse=False, ft=None):
        if not reverse:
            z = input
            assert z.shape[1] == self.in_channels, (z.shape[1], self.in_channels)

            # Feature Conditional
            scaleFt, shiftFt = self.feature_extract(ft, self.fFeatures)
            z = z + shiftFt
            z = z * scaleFt
            logdet = logdet + self.get_logdet(scaleFt)

            # Self Conditional
            z1, z2 = self.split(z)
            scale, shift = self.feature_extract(z1, self.fAffine)
            self.asserts(scale, shift, z1, z2)
            z2 = z2 + shift
            z2 = z2 * scale

            logdet = logdet + self.get_logdet(scale)
            z = thops.cat_feature(z1, z2)
            output = z
        else:
            z = input

            # Self Conditional
            z1, z2 = self.split(z)
            scale, shift = self.feature_extract(z1, self.fAffine)
            self.asserts(scale, shift, z1, z2)
            z2 = z2 / scale
            z2 = z2 - shift
            z = thops.cat_feature(z1, z2)
            logdet = logdet - self.get_logdet(scale)

            # Feature Conditional
            scaleFt, shiftFt = self.feature_extract(ft, self.fFeatures)
            z = z / scaleFt
            z = z - shiftFt
            logdet = logdet - self.get_logdet(scaleFt)

            output = z
        return output, logdet

    def asserts(self, scale, shift, z1, z2):
        assert z1.shape[1] == self.channels_for_nn, (z1.shape[1], self.channels_for_nn)
        assert z2.shape[1] == self.channels_for_co, (z2.shape[1], self.channels_for_co)
        assert scale.shape[1] == shift.shape[1], (scale.shape[1], shift.shape[1])
        assert scale.shape[1] == z2.shape[1], (scale.shape[1], z1.shape[1], z2.shape[1])

    def get_logdet(self, scale):
        return thops.sum(torch.log(scale), dim=[1, 2, 3])

    def feature_extract(self, z, f):
        h = f(z)
        shift, scale = thops.split_feature(h, "cross")
        scale = (torch.sigmoid(scale + 2.) + self.affine_eps)
        return scale, shift

    def split(self, z):
        z1 = z[:, :self.channels_for_nn]
        z2 = z[:, self.channels_for_nn:]
        assert z1.shape[1] + z2.shape[1] == z.shape[1], (z1.shape[1], z2.shape[1], z.shape[1])
        return z1, z2

    def F(self, in_channels, out_channels, hidden_channels, kernel_hidden=1, n_hidden_layers=1):
        layers = [Conv2d(in_channels, hidden_channels), nn.ReLU(inplace=False)]

        for _ in range(n_hidden_layers):
            layers.append(Conv2d(hidden_channels, hidden_channels, kernel_size=[kernel_hidden, kernel_hidden]))
            layers.append(nn.ReLU(inplace=False))
        layers.append(Conv2dZeros(hidden_channels, out_channels))

        return nn.Sequential(*layers)


class CondAffineCatSymmetric(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.need_features = True
        self.in_channels = in_channels
        self.in_channels_rrdb = 318
        self.kernel_hidden = 1
        self.affine_eps = 0.0001
        self.n_hidden_layers = 1
        self.hidden_channels = 64

        self.channels_for_nn = self.in_channels // 2
        self.channels_for_co = self.in_channels - self.channels_for_nn

        if self.channels_for_nn is None:
            self.channels_for_nn = self.in_channels // 2

        self.f1 = self.F(in_channels=self.channels_for_nn + self.in_channels_rrdb,
                         out_channels=self.channels_for_co * 2,
                         hidden_channels=self.hidden_channels,
                         kernel_hidden=self.kernel_hidden,
                         n_hidden_layers=self.n_hidden_layers)

        self.f2 = self.F(in_channels=self.channels_for_nn + self.in_channels_rrdb,
                         out_channels=self.channels_for_co * 2,
                         hidden_channels=self.hidden_channels,
                         kernel_hidden=self.kernel_hidden,
                         n_hidden_layers=self.n_hidden_layers)

    def forward(self, input: torch.Tensor, logdet=None, reverse=False, ft=None):
        if not reverse:
            z = input
            assert z.shape[1] == self.in_channels, (z.shape[1], self.in_channels)

            z1bypass, z1trans = self.split(z)
            scale, shift = self.feature_extract(z1bypass, ft, self.f1)
            self.asserts(scale, shift, z1bypass, z1trans)
            z1trans = z1trans + shift
            z1trans = z1trans * scale
            logdet = logdet + self.get_logdet(scale)
            z = thops.cat_feature(z1bypass, z1trans)

            z2trans, z2bypass = self.split(z)  # different order
            scale2, shift2 = self.feature_extract(z2bypass, ft, self.f2)
            self.asserts(scale2, shift2, z2bypass, z2trans)
            z2trans = z2trans + shift2
            z2trans = z2trans * scale2
            logdet = logdet + self.get_logdet(scale2)
            z = thops.cat_feature(z2trans, z2bypass)

            output = z
        else:
            z = input
            logdet = 0

            z2trans, z2bypass = self.split(z)  # different order
            scale2, shift2 = self.feature_extract(z2bypass, ft, self.f2)
            self.asserts(scale2, shift2, z2bypass, z2trans)
            z2trans = z2trans / scale2
            z2trans = z2trans - shift2
            z = thops.cat_feature(z2trans, z2bypass)
            logdet = logdet - self.get_logdet(scale2)

            z1bypass, z1trans = self.split(z)
            scale, shift = self.feature_extract(z1bypass, ft, self.f1)
            self.asserts(scale, shift, z1bypass, z1trans)
            z1trans = z1trans / scale
            z1trans = z1trans - shift
            z = thops.cat_feature(z1bypass, z1trans)
            logdet = logdet - self.get_logdet(scale)

            output = z
        return output, logdet

    def asserts(self, scale, shift, z1, z2):
        assert z1.shape[1] == self.channels_for_nn, (z1.shape[1], self.channels_for_nn)
        assert z2.shape[1] == self.channels_for_co, (z2.shape[1], self.channels_for_co)
        assert scale.shape[1] == shift.shape[1], (scale.shape[1], shift.shape[1])
        assert scale.shape[1] == z2.shape[1], (scale.shape[1], z1.shape[1], z2.shape[1])

    def get_logdet(self, scale):
        return thops.sum(torch.log(scale), dim=[1, 2, 3])

    def feature_extract(self, z1, ft, f):
        z = torch.cat([z1, ft], dim=1)
        h = f(z)
        shift, scale = thops.split_feature(h, "cross")
        scale = (torch.sigmoid(scale + 2.) + self.affine_eps)
        return scale, shift

    def split(self, z):
        z1 = z[:, :self.channels_for_nn]
        z2 = z[:, self.channels_for_nn:]
        assert z1.shape[1] + z2.shape[1] == z.shape[1], (z1.shape[1], z2.shape[1], z.shape[1])
        return z1, z2

    def F(self, in_channels, out_channels, hidden_channels, kernel_hidden=1, n_hidden_layers=1):
        layers = [Conv2d(in_channels, hidden_channels), nn.ReLU(inplace=False)]

        for _ in range(n_hidden_layers):
            layers.append(Conv2d(hidden_channels, hidden_channels, kernel_size=[kernel_hidden, kernel_hidden]))
            layers.append(nn.ReLU(inplace=False))
        layers.append(Conv2dZeros(hidden_channels, out_channels))

        return nn.Sequential(*layers)


class CondAffineCatSymmetricBaseline(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.need_features = True
        self.in_channels = in_channels
        self.in_channels_rrdb = 318
        self.kernel_hidden = 1
        self.affine_eps = 0.0001
        self.n_hidden_layers = 1
        self.hidden_channels = 64

        self.channels_for_nn = self.in_channels // 2
        self.channels_for_co = self.in_channels - self.channels_for_nn

        if self.channels_for_nn is None:
            self.channels_for_nn = self.in_channels // 2

        self.f1 = self.F(in_channels=self.channels_for_nn + self.in_channels_rrdb,
                         out_channels=self.channels_for_co * 2,
                         hidden_channels=self.hidden_channels,
                         kernel_hidden=self.kernel_hidden,
                         n_hidden_layers=self.n_hidden_layers)

        self.f2 = self.F(in_channels=self.channels_for_nn + self.in_channels_rrdb,
                         out_channels=self.channels_for_co * 2,
                         hidden_channels=self.hidden_channels,
                         kernel_hidden=self.kernel_hidden,
                         n_hidden_layers=self.n_hidden_layers)

    def forward(self, input: torch.Tensor, logdet=None, reverse=False, ft=None):
        if not reverse:
            z = input
            assert z.shape[1] == self.in_channels, (z.shape[1], self.in_channels)

            z1bypass, z1trans = self.split(z)
            scale, shift = self.feature_extract(z1bypass, ft, self.f1)
            self.asserts(scale, shift, z1bypass, z1trans)
            z1trans = z1trans + shift
            z1trans = z1trans * scale
            logdet = logdet + self.get_logdet(scale)
            z = thops.cat_feature(z1bypass, z1trans)

            z2bypass, z2trans = self.split(z)  # different order
            scale2, shift2 = self.feature_extract(z2bypass, ft, self.f2)
            self.asserts(scale2, shift2, z2bypass, z2trans)
            z2trans = z2trans + shift2
            z2trans = z2trans * scale2
            logdet = logdet + self.get_logdet(scale2)
            z = thops.cat_feature(z2bypass, z2trans)

            output = z
        else:
            z = input

            z2bypass, z2trans = self.split(z)  # different order
            scale2, shift2 = self.feature_extract(z2bypass, ft, self.f2)
            self.asserts(scale2, shift2, z2bypass, z2trans)
            z2trans = z2trans / scale2
            z2trans = z2trans - shift2
            z = thops.cat_feature(z2bypass, z2trans)
            logdet = logdet - self.get_logdet(scale2)

            z1bypass, z1trans = self.split(z)
            scale, shift = self.feature_extract(z1bypass, ft, self.f1)
            self.asserts(scale, shift, z1bypass, z1trans)
            z1trans = z1trans / scale
            z1trans = z1trans - shift
            z = thops.cat_feature(z1bypass, z1trans)
            logdet = logdet - self.get_logdet(scale)

            output = z
        return output, logdet

    def asserts(self, scale, shift, z1, z2):
        assert z1.shape[1] == self.channels_for_nn, (z1.shape[1], self.channels_for_nn)
        assert z2.shape[1] == self.channels_for_co, (z2.shape[1], self.channels_for_co)
        assert scale.shape[1] == shift.shape[1], (scale.shape[1], shift.shape[1])
        assert scale.shape[1] == z2.shape[1], (scale.shape[1], z1.shape[1], z2.shape[1])

    def get_logdet(self, scale):
        return thops.sum(torch.log(scale), dim=[1, 2, 3])

    def feature_extract(self, z1, ft, f):
        z = torch.cat([z1, ft], dim=1)
        h = f(z)
        shift, scale = thops.split_feature(h, "cross")
        scale = (torch.sigmoid(scale + 2.) + self.affine_eps)
        return scale, shift

    def split(self, z):
        z1 = z[:, :self.channels_for_nn]
        z2 = z[:, self.channels_for_nn:]
        assert z1.shape[1] + z2.shape[1] == z.shape[1], (z1.shape[1], z2.shape[1], z.shape[1])
        return z1, z2

    def F(self, in_channels, out_channels, hidden_channels, kernel_hidden=1, n_hidden_layers=1):
        layers = [Conv2d(in_channels, hidden_channels), nn.ReLU(inplace=False)]

        for _ in range(n_hidden_layers):
            layers.append(Conv2d(hidden_channels, hidden_channels, kernel_size=[kernel_hidden, kernel_hidden]))
            layers.append(nn.ReLU(inplace=False))
        layers.append(Conv2dZeros(hidden_channels, out_channels))

        return nn.Sequential(*layers)


class CondAffineCondZTrCmid18Ablation(nn.Module):
    # 2019_12_04_ConditionalAffine.pdf
    def __init__(self, in_channels):
        super().__init__()
        self.need_features = True
        self.in_channels = in_channels
        self.in_channels_rrdb = 318
        self.channels_middle = 18
        self.kernel_hidden = 1
        self.affine_eps = 0.0001
        self.n_hidden_layers = 1
        self.hidden_channels = 64

        self.n_bypass = self.in_channels // 2
        self.n_transform = self.in_channels - self.n_bypass

        self.f_fpre = Conv2d(in_channels=self.in_channels_rrdb, out_channels=self.channels_middle, kernel_size=[1, 1])

        self.f_z = self.F(in_channels=self.n_bypass,
                          out_channels=self.channels_middle * 2,
                          hidden_channels=self.hidden_channels,
                          kernel_hidden=self.kernel_hidden,
                          n_hidden_layers=self.n_hidden_layers)

        self.f_f = self.F(in_channels=self.channels_middle,
                          out_channels=self.n_transform * 2,
                          hidden_channels=self.hidden_channels,
                          kernel_hidden=self.kernel_hidden,
                          n_hidden_layers=self.n_hidden_layers)

    def forward(self, input: torch.Tensor, logdet=None, reverse=False, ft=None):
        z = input
        if not reverse:
            z_bypass, z_trans = self.split(z)
            ft_prep = self.f_fpre(ft)

            scaleFt, shiftFt = self.feature_extract(z_bypass, self.f_z)
            ft_trans = affine(ft_prep, scale=scaleFt, shift=shiftFt, reverse=False)

            scale, shift = self.feature_extract(ft_trans, self.f_f)
            z_trans = affine(z_trans, scale=scale, shift=shift, reverse=False)

            logdet = logdet + self.get_logdet(scale)
            z = thops.cat_feature(z_bypass, z_trans)
            output = z
        else:
            z_bypass, z_trans = self.split(z)
            ft_prep = self.f_fpre(ft)

            scaleFt, shiftFt = self.feature_extract(z_bypass, self.f_z)
            ft_trans = affine(ft_prep, scale=scaleFt, shift=shiftFt, reverse=False)

            scale, shift = self.feature_extract(ft_trans, self.f_f)
            z_trans = affine(z_trans, scale=scale, shift=shift, reverse=True)

            logdet = logdet - self.get_logdet(scale)

            z = thops.cat_feature(z_bypass, z_trans)
            output = z
        return output, logdet

    def asserts(self, scale, shift, z_bypass, z_trans):
        assert z_bypass.shape[1] == self.n_bypass, (z_bypass.shape[1], self.n_bypass)
        assert z_trans.shape[1] == self.n_transform, (z_trans.shape[1], self.n_transform)
        assert scale.shape[1] == shift.shape[1], (scale.shape[1], shift.shape[1])
        assert scale.shape[1] == z_trans.shape[1], (scale.shape[1], z_bypass.shape[1], z_trans.shape[1])

    def get_logdet(self, scale):
        return thops.sum(torch.log(scale), dim=[1, 2, 3])

    def feature_extract(self, z, f):
        h = f(z)
        shift, scale = thops.split_feature(h, "cross")
        scale = (torch.sigmoid(scale + 2.) + self.affine_eps)
        return scale, shift

    def split(self, z):
        z_bypass = z[:, :self.n_bypass]
        z_trans = z[:, self.n_bypass:]
        assert z_bypass.shape[1] + z_trans.shape[1] == z.shape[1], (z_bypass.shape[1], z_trans.shape[1], z.shape[1])
        return z_bypass, z_trans

    def F(self, in_channels, out_channels, hidden_channels, kernel_hidden=1, n_hidden_layers=1):
        layers = [Conv2d(in_channels, hidden_channels), nn.ReLU(inplace=False)]

        for _ in range(n_hidden_layers):
            layers.append(Conv2d(hidden_channels, hidden_channels, kernel_size=[kernel_hidden, kernel_hidden]))
            layers.append(nn.ReLU(inplace=False))
        layers.append(Conv2dZeros(hidden_channels, out_channels))

        return nn.Sequential(*layers)


class CondAffineCondFtAblation(nn.Module):
    # 2019_12_04_ConditionalAffine.pdf
    def __init__(self, in_channels):
        super().__init__()
        self.need_features = True
        self.in_channels = in_channels
        self.in_channels_rrdb = 318
        self.channels_middle = 18
        self.kernel_hidden = 1
        self.affine_eps = 0.0001
        self.n_hidden_layers = 1
        self.hidden_channels = 64

        self.n_bypass = self.in_channels // 2
        self.n_transform = self.in_channels - self.n_bypass

        self.f_f = self.F(in_channels=self.in_channels_rrdb,
                          out_channels=self.n_bypass * 2,
                          hidden_channels=self.hidden_channels,
                          kernel_hidden=self.kernel_hidden,
                          n_hidden_layers=self.n_hidden_layers)

        self.f_z = self.F(in_channels=self.n_bypass,
                          out_channels=self.n_transform * 2,
                          hidden_channels=self.hidden_channels,
                          kernel_hidden=self.kernel_hidden,
                          n_hidden_layers=self.n_hidden_layers)

    def forward(self, input: torch.Tensor, logdet=None, reverse=False, ft=None):
        z = input
        if not reverse:
            z_bypass, z_trans = self.split(z)

            scaleFt, shiftFt = self.feature_extract(ft, self.f_f)
            z_bypass_prep = affine(z_bypass, scale=scaleFt, shift=shiftFt, reverse=False)

            scale, shift = self.feature_extract(z_bypass_prep, self.f_z)
            z_trans = affine(z_trans, scale=scale, shift=shift, reverse=False)

            logdet = logdet + self.get_logdet(scale)
            z = thops.cat_feature(z_bypass, z_trans)
            output = z
        else:
            z_bypass, z_trans = self.split(z)

            scaleFt, shiftFt = self.feature_extract(ft, self.f_f)
            z_bypass_prep = affine(z_bypass, scale=scaleFt, shift=shiftFt, reverse=False)

            scale, shift = self.feature_extract(z_bypass_prep, self.f_z)
            z_trans = affine(z_trans, scale=scale, shift=shift, reverse=True)

            logdet = -self.get_logdet(scale)
            z = thops.cat_feature(z_bypass, z_trans)

            output = z
        return output, logdet

    def asserts(self, scale, shift, z_bypass, z_trans):
        assert z_bypass.shape[1] == self.n_bypass, (z_bypass.shape[1], self.n_bypass)
        assert z_trans.shape[1] == self.n_transform, (z_trans.shape[1], self.n_transform)
        assert scale.shape[1] == shift.shape[1], (scale.shape[1], shift.shape[1])
        assert scale.shape[1] == z_trans.shape[1], (scale.shape[1], z_bypass.shape[1], z_trans.shape[1])

    def get_logdet(self, scale):
        return thops.sum(torch.log(scale), dim=[1, 2, 3])

    def feature_extract(self, z, f):
        h = f(z)
        shift, scale = thops.split_feature(h, "cross")
        scale = (torch.sigmoid(scale + 2.) + self.affine_eps)
        return scale, shift

    def split(self, z):
        z_bypass = z[:, :self.n_bypass]
        z_trans = z[:, self.n_bypass:]
        assert z_bypass.shape[1] + z_trans.shape[1] == z.shape[1], (z_bypass.shape[1], z_trans.shape[1], z.shape[1])
        return z_bypass, z_trans

    def F(self, in_channels, out_channels, hidden_channels, kernel_hidden=1, n_hidden_layers=1):
        layers = [Conv2d(in_channels, hidden_channels), nn.ReLU(inplace=False)]

        for _ in range(n_hidden_layers):
            layers.append(Conv2d(hidden_channels, hidden_channels, kernel_size=[kernel_hidden, kernel_hidden]))
            layers.append(nn.ReLU(inplace=False))
        layers.append(Conv2dZeros(hidden_channels, out_channels))

        return nn.Sequential(*layers)


class CondAffineCatSymmetricAndSep(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.need_features = True
        self.in_channels = in_channels
        self.in_channels_rrdb = 318
        self.kernel_hidden = 1
        self.affine_eps = 0.0001
        self.n_hidden_layers = 1
        self.hidden_channels = 64

        self.channels_for_nn = self.in_channels // 2
        self.channels_for_co = self.in_channels - self.channels_for_nn

        if self.channels_for_nn is None:
            self.channels_for_nn = self.in_channels // 2

        self.f1 = self.F(in_channels=self.channels_for_nn + self.in_channels_rrdb,
                         out_channels=self.channels_for_co * 2,
                         hidden_channels=self.hidden_channels,
                         kernel_hidden=self.kernel_hidden,
                         n_hidden_layers=self.n_hidden_layers)

        self.f2 = self.F(in_channels=self.channels_for_nn + self.in_channels_rrdb,
                         out_channels=self.channels_for_co * 2,
                         hidden_channels=self.hidden_channels,
                         kernel_hidden=self.kernel_hidden,
                         n_hidden_layers=self.n_hidden_layers)

        self.fFeatures = self.F(in_channels=self.in_channels_rrdb,
                                out_channels=self.in_channels * 2,
                                hidden_channels=self.hidden_channels,
                                kernel_hidden=self.kernel_hidden,
                                n_hidden_layers=self.n_hidden_layers)

    def forward(self, input: torch.Tensor, logdet=None, reverse=False, ft=None):
        if not reverse:
            z = input
            assert z.shape[1] == self.in_channels, (z.shape[1], self.in_channels)

            # Feature Conditional
            scaleFt, shiftFt = self.feature_extract_ft(ft, self.fFeatures)
            z = z + shiftFt
            z = z * scaleFt
            logdet = logdet + self.get_logdet(scaleFt)

            z1bypass, z1trans = self.split(z)
            scale, shift = self.feature_extract(z1bypass, ft, self.f1)
            self.asserts(scale, shift, z1bypass, z1trans)
            z1trans = z1trans + shift
            z1trans = z1trans * scale
            logdet = logdet + self.get_logdet(scale)
            z = thops.cat_feature(z1bypass, z1trans)

            z2trans, z2bypass = self.split(z)  # different order
            scale2, shift2 = self.feature_extract(z2bypass, ft, self.f2)
            self.asserts(scale2, shift2, z2bypass, z2trans)
            z2trans = z2trans + shift2
            z2trans = z2trans * scale2
            logdet = logdet + self.get_logdet(scale2)
            z = thops.cat_feature(z2trans, z2bypass)

            output = z
        else:
            z = input

            z2trans, z2bypass = self.split(z)  # different order
            scale2, shift2 = self.feature_extract(z2bypass, ft, self.f2)
            self.asserts(scale2, shift2, z2bypass, z2trans)
            z2trans = z2trans / scale2
            z2trans = z2trans - shift2
            z = thops.cat_feature(z2trans, z2bypass)
            logdet = logdet - self.get_logdet(scale2)

            z1bypass, z1trans = self.split(z)
            scale, shift = self.feature_extract(z1bypass, ft, self.f1)
            self.asserts(scale, shift, z1bypass, z1trans)
            z1trans = z1trans / scale
            z1trans = z1trans - shift
            z = thops.cat_feature(z1bypass, z1trans)
            logdet = logdet - self.get_logdet(scale)

            # Feature Conditional
            scaleFt, shiftFt = self.feature_extract_ft(ft, self.fFeatures)
            z = z / scaleFt
            z = z - shiftFt
            logdet = logdet - self.get_logdet(scaleFt)

            output = z
        return output, logdet

    def asserts(self, scale, shift, z1, z2):
        assert z1.shape[1] == self.channels_for_nn, (z1.shape[1], self.channels_for_nn)
        assert z2.shape[1] == self.channels_for_co, (z2.shape[1], self.channels_for_co)
        assert scale.shape[1] == shift.shape[1], (scale.shape[1], shift.shape[1])
        assert scale.shape[1] == z2.shape[1], (scale.shape[1], z1.shape[1], z2.shape[1])

    def feature_extract_ft(self, z, f):
        h = f(z)
        shift, scale = thops.split_feature(h, "cross")
        scale = (torch.sigmoid(scale + 2.) + self.affine_eps)
        return scale, shift

    def get_logdet(self, scale):
        return thops.sum(torch.log(scale), dim=[1, 2, 3])

    def feature_extract(self, z1, ft, f):
        z = torch.cat([z1, ft], dim=1)
        h = f(z)
        shift, scale = thops.split_feature(h, "cross")
        scale = (torch.sigmoid(scale + 2.) + self.affine_eps)
        return scale, shift

    def split(self, z):
        z1 = z[:, :self.channels_for_nn]
        z2 = z[:, self.channels_for_nn:]
        assert z1.shape[1] + z2.shape[1] == z.shape[1], (z1.shape[1], z2.shape[1], z.shape[1])
        return z1, z2

    def F(self, in_channels, out_channels, hidden_channels, kernel_hidden=1, n_hidden_layers=1):
        layers = [Conv2d(in_channels, hidden_channels), nn.ReLU(inplace=False)]

        for _ in range(n_hidden_layers):
            layers.append(Conv2d(hidden_channels, hidden_channels, kernel_size=[kernel_hidden, kernel_hidden]))
            layers.append(nn.ReLU(inplace=False))
        layers.append(Conv2dZeros(hidden_channels, out_channels))

        return nn.Sequential(*layers)

'''
class CondAffineSeparatedAndCond(nn.Module):
    def __init__(self, in_channels, opt):
        super().__init__()
        self.need_features = True
        self.in_channels = in_channels
        self.in_channels_rrdb = 318
        self.kernel_hidden = 1
        self.affine_eps = 0.0001
        self.n_hidden_layers = 1
        hidden_channels = opt_get(opt, ['network_G', 'flow', 'CondAffineSeparatedAndCond', 'hidden_channels'])
        self.hidden_channels = 64 if hidden_channels is None else hidden_channels

        self.affine_eps = opt_get(opt, ['network_G', 'flow', 'CondAffineSeparatedAndCond', 'eps'],  0.0001)

        self.mult_reverse = opt_get(opt, ['network_G', 'flow', 'CondAffineSeparatedAndCond', 'multReverse'],  False)
        self.use_tanh = opt_get(opt, ['network_G', 'flow', 'CondAffineSeparatedAndCond', 'tanh'],  False)
        self.skip = opt_get(opt, ['network_G', 'flow', 'CondAffineSeparatedAndCond', 'skip'],  False)

        self.channels_for_nn = self.in_channels // 2
        self.channels_for_co = self.in_channels - self.channels_for_nn

        if self.channels_for_nn is None:
            self.channels_for_nn = self.in_channels // 2

        self.fAffine = self.F(in_channels=self.channels_for_nn + self.in_channels_rrdb,
                              out_channels=self.channels_for_co * 2,
                              hidden_channels=self.hidden_channels,
                              kernel_hidden=self.kernel_hidden,
                              n_hidden_layers=self.n_hidden_layers)

        self.fFeatures = self.F(in_channels=self.in_channels_rrdb,
                                out_channels=self.in_channels * 2,
                                hidden_channels=self.hidden_channels,
                                kernel_hidden=self.kernel_hidden,
                                n_hidden_layers=self.n_hidden_layers)

    def forward(self, input: torch.Tensor, logdet=None, reverse=False, ft=None):

        if self.skip: # quick way to deactivate conditional
            return input, logdet 

        if self.mult_reverse:
            reverse = not reverse

        if not reverse:
            z = input
            assert z.shape[1] == self.in_channels, (z.shape[1], self.in_channels)

            # Feature Conditional
            scaleFt, shiftFt = self.feature_extract(ft, self.fFeatures)
            z = z + shiftFt
            z = z * scaleFt
            logdet = logdet + self.get_logdet(scaleFt)

            # Self Conditional
            z1, z2 = self.split(z)
            scale, shift = self.feature_extract_aff(z1, ft, self.fAffine)
            self.asserts(scale, shift, z1, z2)
            z2 = z2 + shift
            z2 = z2 * scale

            logdet = logdet + self.get_logdet(scale)
            z = thops.cat_feature(z1, z2)
            output = z
        else:
            z = input

            # Self Conditional
            z1, z2 = self.split(z)
            scale, shift = self.feature_extract_aff(z1, ft, self.fAffine)
            self.asserts(scale, shift, z1, z2)
            z2 = z2 / scale
            z2 = z2 - shift
            z = thops.cat_feature(z1, z2)
            logdet = logdet - self.get_logdet(scale)
            # print("CondAffineSeparatedAndCond Reverse Self Conditional:", z.abs().max().item(), logdet.abs().max().item())#, scale , shift)

            # Feature Conditional
            scaleFt, shiftFt = self.feature_extract(ft, self.fFeatures)
            z = z / scaleFt
            z = z - shiftFt
            logdet = logdet - self.get_logdet(scaleFt)
            # print("CondAffineSeparatedAndCond Reverse Feature Conditional:", z.abs().max().item(), logdet.abs().max().item())#, scaleFt, shiftFt)

            output = z
        return output, logdet

    def asserts(self, scale, shift, z1, z2):
        expected_nn = self.channels_for_nn
        expected_co = self.channels_for_co

        if z1.shape[1] != expected_nn or z2.shape[1] != expected_co:
            # Mensaje informativo para debug — quita/transforma en logging en producción
            msg = (
            "Channel mismatch in CondAffineSeparatedAndCond asserts:\n"
            f"  z.shape = {z1.shape[0]},{z1.shape[1]},{z1.shape[2]},{z1.shape[3]}\n"
            f"  z1.shape[1] = {z1.shape[1]} (expected {expected_nn})\n"
            f"  z2.shape[1] = {z2.shape[1]} (expected {expected_co})\n"
            f"  in_channels = {self.in_channels}\n"
            f"  channels_for_nn = {self.channels_for_nn}\n"
            f"  channels_for_co = {self.channels_for_co}\n"
            f"  in_channels_rrdb = {self.in_channels_rrdb}\n"
            )
            raise AssertionError(msg)

        if scale.shape[1] != shift.shape[1] or scale.shape[1] != z2.shape[1]:
            raise AssertionError((scale.shape[1], shift.shape[1], z2.shape[1]))


    def get_logdet(self, scale):
        return thops.sum(torch.log(scale), dim=[1, 2, 3])

    def feature_extract(self, z, f):
        h = f(z)
        shift, scale = thops.split_feature(h, "cross")
        
        if self.use_tanh:
            scale = torch.exp(torch.tanh(scale)) + self.affine_eps
        else:
            scale = torch.sigmoid(scale + 2.) + self.affine_eps

        return scale, shift

    def feature_extract_aff(self, z1, ft, f):
        z = torch.cat([z1, ft], dim=1)
        h = f(z)
        shift, scale = thops.split_feature(h, "cross")

        if self.use_tanh:
            scale = torch.exp(torch.tanh(scale)) + self.affine_eps
        else:
            scale = torch.sigmoid(scale + 2.) + self.affine_eps

        return scale, shift

    def split(self, z):
        z1 = z[:, :self.channels_for_nn]
        z2 = z[:, self.channels_for_nn:]
        assert z1.shape[1] + z2.shape[1] == z.shape[1], (z1.shape[1], z2.shape[1], z.shape[1])
        return z1, z2

    def F(self, in_channels, out_channels, hidden_channels, kernel_hidden=1, n_hidden_layers=1):
        layers = [Conv2d(in_channels, hidden_channels), nn.ReLU(inplace=False)]

        for _ in range(n_hidden_layers):
            layers.append(Conv2d(hidden_channels, hidden_channels, kernel_size=[kernel_hidden, kernel_hidden]))
            layers.append(nn.ReLU(inplace=False))
        layers.append(Conv2dZeros(hidden_channels, out_channels))

        return nn.Sequential(*layers)
'''

# FlowAffineCouplingsAblation.py
class CondAffineSeparatedAndCond(nn.Module):
    def __init__(self, in_channels, opt):
        super().__init__()
        self.need_features = True

        # keep the provided in_channels as a "nominal" value, but we will
        # actually derive channels_for_nn / channels_for_co at first forward
        self.nominal_in_channels = in_channels

        # Read rrdb channels from config if provided, fallback to 0
        self.in_channels_rrdb = opt_get(opt, ['network_G', 'flow', 'CondAffineSeparatedAndCond', 'in_channels_rrdb'], 0)

        self.kernel_hidden = 1
        self.affine_eps = opt_get(opt, ['network_G', 'flow', 'CondAffineSeparatedAndCond', 'eps'], 0.0001)
        self.n_hidden_layers = 1
        hidden_channels = opt_get(opt, ['network_G', 'flow', 'CondAffineSeparatedAndCond', 'hidden_channels'])
        self.hidden_channels = 64 if hidden_channels is None else hidden_channels

        self.mult_reverse = opt_get(opt, ['network_G', 'flow', 'CondAffineSeparatedAndCond', 'multReverse'], False)
        self.use_tanh = opt_get(opt, ['network_G', 'flow', 'CondAffineSeparatedAndCond', 'tanh'], False)
        self.skip = opt_get(opt, ['network_G', 'flow', 'CondAffineSeparatedAndCond', 'skip'], False)

        # We will lazily initialise the two networks below (so they match runtime channel counts)
        self.fAffine = None
        self.fFeatures = None

        # store opt for lazy init
        self._opt = opt

        # placeholders for runtime-derived channel counts
        self.in_channels = None
        self.channels_for_nn = None
        self.channels_for_co = None

    # ------------------------------------------------------------------
    # Helper factory and feature-extractors (add these inside the class)
    # ------------------------------------------------------------------
    def F(self, in_channels, out_channels, hidden_channels, kernel_hidden=1, n_hidden_layers=1):
        """
        Same factory used elsewhere in this file: conv -> ReLU -> (hidden convs) -> Conv2dZeros.
        Returns nn.Sequential mapping (N,in_channels,H,W)->(N,out_channels,H,W).
        """
        layers = [Conv2d(in_channels, hidden_channels), nn.ReLU(inplace=False)]

        for _ in range(n_hidden_layers):
            layers.append(Conv2d(hidden_channels, hidden_channels, kernel_size=[kernel_hidden, kernel_hidden]))
            layers.append(nn.ReLU(inplace=False))
        layers.append(Conv2dZeros(hidden_channels, out_channels))

        return nn.Sequential(*layers)

    def feature_extract(self, z, f):
        """
        Generic feature extractor used for fFeatures (ft-only) in other classes.
        If f is None, return identity-like scale=1 and shift=0 (no-op).
        """
        if f is None:
            # create scale=ones and shift=zeros to be shape-compatible with z
            B, C, H, W = z.shape
            scale = torch.ones((B, C, H, W), device=z.device, dtype=z.dtype)
            shift = torch.zeros((B, C, H, W), device=z.device, dtype=z.dtype)
            return scale, shift

        h = f(z)
        shift, scale = thops.split_feature(h, "cross")
        scale = (torch.sigmoid(scale + 2.) + self.affine_eps)
        return scale, shift

    def feature_extract_aff(self, z1, ft, f):
        """
        fAffine in this class expects concatenated [z1, ft] as input (matches original design).
        If ft is None but f expects it, we still call f with z1 (best-effort).
        """
        if f is None:
            # no affine net: return neutral scale and zero shift sized like z1
            B, C, H, W = z1.shape
            scale = torch.ones((B, C, H, W), device=z1.device, dtype=z1.dtype)
            shift = torch.zeros((B, C, H, W), device=z1.device, dtype=z1.dtype)
            return scale, shift

        if ft is None:
            # If ft not provided, fall back to applying f on z1 only (some variants use this)
            h = f(z1)
        else:
            # concatenate along channel dim (same pattern used in original code)
            # make sure spatial sizes match
            if ft.shape[2:] != z1.shape[2:]:
                # interpolate ft to z1 spatial size (safe fallback)
                ft = torch.nn.functional.interpolate(ft, size=z1.shape[2:], mode='bilinear', align_corners=False)
            z = torch.cat([z1, ft], dim=1)
            h = f(z)

        shift, scale = thops.split_feature(h, "cross")
        scale = (torch.sigmoid(scale + 2.) + self.affine_eps)
        return scale, shift

    def get_logdet(self, scale):
        """
        Consistent logdet helper used in other classes.
        """
        return thops.sum(torch.log(scale), dim=[1, 2, 3])

    # ------------------------------------------------------------------
    def _lazy_init_with(self, z, ft):
        """
        Initialize fAffine and fFeatures based on actual incoming tensor shapes.
        Called on the first forward pass (or whenever shapes change).
        """
        # actual input channels
        actual_in_ch = int(z.shape[1])
        self.in_channels = actual_in_ch

        # derive split: default to half/half (same behavior as original)
        self.channels_for_nn = self.in_channels // 2
        self.channels_for_co = self.in_channels - self.channels_for_nn

        # validate rrdb feature channels from ft if provided
        if ft is not None:
            actual_ft_ch = int(ft.shape[1])  # assume ft is tensor with shape [B,C,H,W]
            # if config gives in_channels_rrdb but it doesn't match ft, warn and prefer ft shape
            if self.in_channels_rrdb and self.in_channels_rrdb != actual_ft_ch:
                print(f"[WARN] CondAffineSeparatedAndCond: opt in_channels_rrdb={self.in_channels_rrdb} "
                      f"!= runtime ft channels={actual_ft_ch}. Using runtime ft channels.")
            self.in_channels_rrdb = actual_ft_ch
        else:
            # no ft provided: keep whatever config had (could be 0)
            if self.in_channels_rrdb == 0:
                # not ideal — but allow it: fFeatures will be created with in_channels=0
                pass

        # create fAffine and fFeatures matching the derived sizes
        # Note: self.F should be set in the class scope usage (original code used self.F)
        # Keep same hidden / kernel settings as original
        self.fAffine = self.F(
            in_channels=self.channels_for_nn + self.in_channels_rrdb,
            out_channels=self.channels_for_co * 2,
            hidden_channels=self.hidden_channels,
            kernel_hidden=self.kernel_hidden,
            n_hidden_layers=self.n_hidden_layers
        )

        self.fFeatures = self.F(
            in_channels=self.in_channels_rrdb,
            out_channels=self.in_channels * 2,
            hidden_channels=self.hidden_channels,
            kernel_hidden=self.kernel_hidden,
            n_hidden_layers=self.n_hidden_layers
        )

    def forward(self, input: torch.Tensor, logdet=None, reverse=False, ft=None):

        if self.skip:  # quick way to deactivate conditional
            return input, logdet

        if self.mult_reverse:
            reverse = not reverse

        # lazy init if needed (or if input channel count changed)
        if self.fAffine is None or self.in_channels is None or self.in_channels != input.shape[1]:
            # init based on runtime tensors
            self._lazy_init_with(input, ft)

        z = input

        # Validate z channels equal to self.in_channels
        if z.shape[1] != self.in_channels:
            raise AssertionError(
                f"Input channel mismatch: runtime z has {z.shape[1]} channels but module expects {self.in_channels}."
                " This means upstream channel bookkeeping is inconsistent. "
                "Check FlowUpsamplerNet.self.C and any bypass/squeeze layers."
            )

        if not reverse:
            # Feature Conditional
            scaleFt, shiftFt = self.feature_extract(ft, self.fFeatures)
            z = z + shiftFt
            z = z * scaleFt
            logdet = logdet + self.get_logdet(scaleFt)

            # Self Conditional
            z1, z2 = self.split(z)
            scale, shift = self.feature_extract_aff(z1, ft, self.fAffine)

            # asserts will confirm sizes are ok
            self.asserts(scale, shift, z1, z2)

            z2 = z2 + shift
            z2 = z2 * scale
            logdet = logdet + self.get_logdet(scale)
            z = thops.cat_feature(z1, z2)
            output = z
        else:
            # reverse path
            z1, z2 = self.split(z)
            scale, shift = self.feature_extract_aff(z1, ft, self.fAffine)
            self.asserts(scale, shift, z1, z2)
            z2 = z2 / scale
            z2 = z2 - shift
            z = thops.cat_feature(z1, z2)
            logdet = logdet - self.get_logdet(scale)

            # Feature Conditional reverse
            scaleFt, shiftFt = self.feature_extract(ft, self.fFeatures)
            z = z / scaleFt
            z = z - shiftFt
            logdet = logdet - self.get_logdet(scaleFt)
            output = z

        return output, logdet

    def asserts(self, scale, shift, z1, z2):
        """
        Informative checks using runtime-derived channels_for_nn and channels_for_co.
        """
        expected_nn = self.channels_for_nn
        expected_co = self.channels_for_co

        if z1.shape[1] != expected_nn or z2.shape[1] != expected_co:
            msg = (
                "Channel mismatch in CondAffineSeparatedAndCond asserts:\n"
                f"  runtime z1.shape = {tuple(z1.shape)}\n"
                f"  runtime z2.shape = {tuple(z2.shape)}\n"
                f"  expected channels_for_nn = {expected_nn}\n"
                f"  expected channels_for_co = {expected_co}\n"
                f"  in_channels (module) = {self.in_channels}\n"
                f"  in_channels_rrdb (module) = {self.in_channels_rrdb}\n"
            )
            raise AssertionError(msg)

        if scale.shape[1] != shift.shape[1] or scale.shape[1] != z2.shape[1]:
            raise AssertionError(
                f"Scale/shift/z2 channel mismatch: scale={scale.shape}, shift={shift.shape}, z2={z2.shape}"
            )
