import torch
import torch.nn as nn
from torch.distributions.uniform import Uniform
import numpy as np


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):

        return self.fn(x) + x


class RREC(nn.Module):
    def __init__(self, dim=1024, depth=7, k=7):
        super(RREC, self).__init__()
        self.block = nn.Sequential(
            *[Residual(nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=(k, k), groups=dim, padding=(k // 2, k // 2)),
                nn.GELU(),
                nn.BatchNorm2d(dim),
                nn.Conv2d(dim, dim, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(dim)
            )) for i in range(depth)]
        )

    def forward(self, x):
        x = self.block(x)

        return x


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.GELU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.GELU()
        )

    def forward(self, x):
        x = self.conv(x)

        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.up(x)

        return x


class FeatureNoise(nn.Module):
    def __init__(self, uniform_range=0.3):
        super(FeatureNoise, self).__init__()
        self.uni_dist = Uniform(-uniform_range, uniform_range)

    def feature_based_noise(self, x):
        noise_vector = self.uni_dist.sample(x.shape[1:]).to(x.device).unsqueeze(0)
        x_noise = x.mul(noise_vector) + x

        return x_noise

    def forward(self, x):
        x = self.feature_based_noise(x)

        return x


def Dropout(x, p=0.3):
    x = torch.nn.functional.dropout(x, p)

    return x


def FeatureDropout(x):
    attention = torch.mean(x, dim=1, keepdim=True)
    max_val, _ = torch.max(attention.view(x.size(0), -1), dim=1, keepdim=True)
    threshold = max_val * np.random.uniform(0.7, 0.9)
    threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
    drop_mask = (attention < threshold).float()
    x = x.mul(drop_mask)

    return x


class Decoder(nn.Module):
    def __init__(self, dim_mult=4):
        super(Decoder, self).__init__()
        self.Up5 = up_conv(ch_in=256 * dim_mult, ch_out=128 * dim_mult)
        self.Up_conv5 = conv_block(ch_in=128 * 2 * dim_mult, ch_out=128 * dim_mult)
        self.Up4 = up_conv(ch_in=128 * dim_mult, ch_out=64 * dim_mult)
        self.Up_conv4 = conv_block(ch_in=64 * 2 * dim_mult, ch_out=64 * dim_mult)
        self.Up3 = up_conv(ch_in=64 * dim_mult, ch_out=32 * dim_mult)
        self.Up_conv3 = conv_block(ch_in=32 * 2 * dim_mult, ch_out=32 * dim_mult)
        self.Up2 = up_conv(ch_in=32 * dim_mult, ch_out=16 * dim_mult)
        self.Up_conv2 = conv_block(ch_in=16 * 2 * dim_mult, ch_out=16 * dim_mult)
        self.Conv_1x1 = nn.Conv2d(16 * dim_mult, 1, kernel_size=1, stride=1, padding=0)


    def forward(self, feature):
        x1, x2, x3, x4, x5 = feature

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1


class DEMS(nn.Module):
    def __init__(self, img_ch=3, length=(3, 3, 3), k=7, dim_mult=4):
        super(DEMS, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = conv_block(ch_in=img_ch, ch_out=16 * dim_mult)
        self.Conv2 = conv_block(ch_in=16 * dim_mult, ch_out=32 * dim_mult)
        self.Conv3 = conv_block(ch_in=32 * dim_mult, ch_out=64 * dim_mult)
        self.Conv4 = conv_block(ch_in=64 * dim_mult, ch_out=128 * dim_mult)
        self.Conv5 = conv_block(ch_in=128 * dim_mult, ch_out=256 * dim_mult)
        self.RREC1 = RREC(dim=256 * dim_mult, depth=length[0], k=k)
        self.RREC2 = RREC(dim=256 * dim_mult, depth=length[1], k=k)
        self.RREC3 = RREC(dim=256 * dim_mult, depth=length[2], k=k)
        self.main_decoder = Decoder(dim_mult=dim_mult)
        self.aux_decoder1 = Decoder(dim_mult=dim_mult)
        self.aux_decoder2 = Decoder(dim_mult=dim_mult)
        self.aux_decoder3 = Decoder(dim_mult=dim_mult)

    def forward(self, x):
        x1 = self.Conv1(x)
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        if not self.training:
            x5 = self.RREC1(x5)
            x5 = self.RREC2(x5)
            x5 = self.RREC3(x5)
            feature = [x1, x2, x3, x4, x5]
            main_seg = self.main_decoder(feature)

            return main_seg

        feature = [x1, x2, x3, x4, x5]
        aux1_feature = [FeatureDropout(i) for i in feature]
        aux_seg1 = self.aux_decoder1(aux1_feature)

        x5 = self.RREC1(x5)
        feature = [x1, x2, x3, x4, x5]
        aux2_feature = [Dropout(i) for i in feature]
        aux_seg2 = self.aux_decoder2(aux2_feature)

        x5 = self.RREC2(x5)
        feature = [x1, x2, x3, x4, x5]
        aux3_feature = [FeatureNoise()(i) for i in feature]
        aux_seg3 = self.aux_decoder3(aux3_feature)

        x5 = self.RREC3(x5)
        feature = [x1, x2, x3, x4, x5]
        main_seg = self.main_decoder(feature)

        return main_seg, aux_seg1, aux_seg2, aux_seg3
