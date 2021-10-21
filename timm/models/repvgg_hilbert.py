import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Dict, Any, cast

import numpy as np
from copy import deepcopy
import math
from hilbert import decode, encode
from pyzorder import ZOrderIndexer

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .helpers import build_model_with_cfg
from .layers import ClassifierHead, ConvBnAct
from .registry import register_model



def _mcfg(**kwargs):
    cfg = dict(encoder_mode='scan', kernel_size=(3, 3), padding=(1, 1, 1, 1), alter=1, pooling=False)
    cfg.update(**kwargs)
    return cfg

a = 1
num_blocks_1 = [4 * a, 6 * a, 16 * a, 1]
a = 1.5
num_blocks_1_5 = [4 * a, 6 * a, 16 * a, 1]
num_blocks_1_5 = [math.floor(i) for i in num_blocks_1_5]
a = 2
num_blocks_2 = [4 * a, 6 * a, 16 * a, 1]
a = 3
num_blocks_3 = [4 * a, 6 * a, 16 * a, 1]

# kernel size: (h, w)
# padding: (left, right, top, bottom)
model_cfgs = dict(
    repvgg_a0=_mcfg(num_blocks=[2, 4, 14, 1], width_multiplier=[0.75, 0.75, 0.75, 2.5], kernel_size=(3, 3), padding=(1, 1)),
    repvgg_b1=_mcfg(num_blocks=num_blocks_1, width_multiplier=[2, 2, 2, 4], kernel_size=(3, 3), padding=(1, 1, 1, 1)),

    repvgg_b1_3x1=_mcfg(num_blocks=num_blocks_1, width_multiplier=[2, 2, 2, 4], kernel_size=(3, 1), padding=(0, 0, 1, 1)),
    repvgg_b1_1x3=_mcfg(num_blocks=num_blocks_1, width_multiplier=[2, 2, 2, 4], kernel_size=(1, 3), padding=(1, 1, 0, 0)),
    repvgg_b1_3x1_1x3=_mcfg(num_blocks=num_blocks_1, width_multiplier=[2, 2, 2, 4], kernel_size=(3, 1), padding=(0, 0, 1, 1), alter=2),
    repvgg_b1_2x2=_mcfg(num_blocks=num_blocks_1, width_multiplier=[2, 2, 2, 4], kernel_size=(2, 2), padding=(0, 1, 0, 1)),
    repvgg_b1_3x1_pooling=_mcfg(num_blocks=num_blocks_1, width_multiplier=[2, 2, 2, 4], kernel_size=(3, 1), padding=(0, 0, 1, 1), pooling=True),
    repvgg_b1_3x1_1x3_pooling=_mcfg(num_blocks=num_blocks_1, width_multiplier=[2, 2, 2, 4], kernel_size=(3, 1), padding=(0, 0, 1, 1), alter=2, pooling=True),

    repvgg_b1_3x1_scan=_mcfg(num_blocks=num_blocks_1, width_multiplier=[2, 2, 2, 4], encoder_mode='scan', kernel_size=(3), padding=(1)),
    repvgg_b1_3x1_zigzag=_mcfg(num_blocks=num_blocks_1, width_multiplier=[2, 2, 2, 4], encoder_mode='zigzag', kernel_size=(3), padding=(1)),
    repvgg_b1_3x1_zorder=_mcfg(num_blocks=num_blocks_1, width_multiplier=[2, 2, 2, 4], encoder_mode='zorder', kernel_size=(3), padding=(1)),
    repvgg_b1_3x1_hilbert=_mcfg(num_blocks=num_blocks_1, width_multiplier=[2, 2, 2, 4], encoder_mode='hilbert', kernel_size=(3), padding=(1)),
)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 256, 256), 'pool_size': (1, 1),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'features.0', 'classifier': 'head.fc',
        **kwargs
    }


default_cfgs = {
    'repvgg_a0': _cfg(url=''),
    'repvgg_b1': _cfg(url=''),

    'repvgg_b1_3x1': _cfg(url=''),
    'repvgg_b1_1x3': _cfg(url=''),
    'repvgg_b1_3x1_1x3': _cfg(url=''),
    'repvgg_b1_2x2': _cfg(url=''),
    'repvgg_b1_3x1_pooling': _cfg(url=''),
    'repvgg_b1_3x1_1x3_pooling': _cfg(url=''),

    'repvgg_b1_3x1_scan': _cfg(url=''),
    'repvgg_b1_3x1_zigzag': _cfg(url=''),
    'repvgg_b1_3x1_zorder': _cfg(url=''),
    'repvgg_b1_3x1_hilbert': _cfg(url=''),
}


class ConvNormActi(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='reflect',
                 norm=None, acti=None):
        super(ConvNormActi, self).__init__()
        block = [nn.ZeroPad2d(padding=padding), nn.Conv2d(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias, padding_mode)]
        if norm:
            block.append(nn.BatchNorm2d(out_channels))
        if acti:
            block.append(nn.ReLU())
        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class SkipConvNormActi(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='reflect'):
        super(SkipConvNormActi, self).__init__()
        self.conv = ConvNormActi(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, norm=True)
        self.skip = nn.BatchNorm2d(out_channels) if out_channels == in_channels and stride == 1 else None
        self.acti = nn.ReLU()

    def forward(self, x):
        x_conv = self.conv(x)
        x_skip = self.skip(x) if self.skip else 0
        out = self.acti((x_conv + x_skip) / math.sqrt(2))
        return out


class Skip2ConvNormActi(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='reflect'):
        super(Skip2ConvNormActi, self).__init__()
        self.conv = nn.Sequential(ConvNormActi(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, norm=True, acti=True),
                                  ConvNormActi(out_channels, out_channels, kernel_size, 1, padding, dilation, groups, bias, padding_mode, norm=True))
        self.skip = nn.BatchNorm2d(out_channels) if out_channels == in_channels and stride == 1 else None
        self.acti = nn.ReLU()

    def forward(self, x):
        x_conv = self.conv(x)
        x_skip = self.skip(x) if self.skip else 0
        out = self.acti((x_conv + x_skip) / math.sqrt(2))
        return out


class RepVGG_2D(nn.Module):

    def __init__(self, cfg, img_size=(224,), num_classes=1000, in_chans=3, drop_rate=0.0):
        super(RepVGG_2D, self).__init__()
        self.num_classes = num_classes

        num_blocks = cfg['num_blocks']
        width_multiplier = cfg['width_multiplier']
        assert len(width_multiplier) == 4
        self.kernel_size = cfg['kernel_size']
        self.padding = cfg['padding']
        self.pooling = cfg['pooling']
        self.kernel_size_alter = tuple(reversed(self.kernel_size))
        self.padding_alter = tuple(reversed(self.padding))

        self.alter = cfg['alter']

        self.feature_info = []

        self.in_planes = min(64, int(64 * width_multiplier[0]))
        self.stage0 = SkipConvNormActi(in_channels=in_chans, out_channels=self.in_planes, kernel_size=self.kernel_size, stride=2, padding=self.padding)
        self.layer_idx = 0
        self.stage1 = self._make_stage(int(64 * width_multiplier[0]), num_blocks[0], stride=2)
        self.stage2 = self._make_stage(int(128 * width_multiplier[1]), num_blocks[1], stride=2)
        self.stage3 = self._make_stage(int(256 * width_multiplier[2]), num_blocks[2], stride=2)
        self.stage4 = self._make_stage(int(512 * width_multiplier[3]), num_blocks[3], stride=2)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.dropout = nn.Dropout(p=drop_rate)
        self.linear = nn.Linear(int(512 * width_multiplier[3]), num_classes)

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        blocks = []
        for stride in strides:
            if self.layer_idx % self.alter == 0:
                kernel_size = self.kernel_size
                padding = self.padding
            else:
                kernel_size = self.kernel_size_alter
                padding = self.padding_alter
            padding = (0, 0, 0, 0) if stride == 2 and self.kernel_size == (2, 2) else padding
            if not self.pooling:
                blocks.append(SkipConvNormActi(in_channels=self.in_planes, out_channels=planes, kernel_size=kernel_size, stride=stride, padding=padding))
            else:
                blocks.append(SkipConvNormActi(in_channels=self.in_planes, out_channels=planes, kernel_size=kernel_size, stride=1, padding=padding))
                if stride == 2:
                    blocks.append(nn.AvgPool2d(kernel_size=stride))
            self.in_planes = planes
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = self.stage0(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.linear(out)
        return out


class ConvNormActi_1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='reflect',
                 norm=None, acti=None):
        super(ConvNormActi_1d, self).__init__()
        block = [nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)]
        if norm:
            block.append(nn.BatchNorm1d(out_channels))
        if acti:
            block.append(nn.ReLU())
        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class SkipConvNormActi_1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='reflect'):
        super(SkipConvNormActi_1d, self).__init__()
        self.conv = ConvNormActi_1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, norm=True)
        self.skip = nn.BatchNorm1d(out_channels) if out_channels == in_channels and stride == 1 else None
        self.acti = nn.ReLU()

    def forward(self, x):
        x_conv = self.conv(x)
        x_skip = self.skip(x) if self.skip else 0
        out = self.acti((x_conv + x_skip) / math.sqrt(2))
        return out


class Skip2ConvNormActi_1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='reflect'):
        super(Skip2ConvNormActi_1d, self).__init__()
        self.conv = nn.Sequential(ConvNormActi_1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, norm=True, acti=True),
                                  ConvNormActi_1d(out_channels, out_channels, kernel_size, 1, padding, dilation, groups, bias, padding_mode, norm=True))
        self.skip = nn.BatchNorm1d(out_channels) if out_channels == in_channels and stride == 1 else None
        self.acti = nn.ReLU()

    def forward(self, x):
        x_conv = self.conv(x)
        x_skip = self.skip(x) if self.skip else 0
        out = self.acti((x_conv + x_skip) / math.sqrt(2))
        return out


class LocED(nn.Module):

    def __init__(self, bit=4, dim=2, loc_encoder='hilbert'):
        super().__init__()
        size = 2 ** bit
        max_num = size ** dim
        indexes = np.arange(max_num)
        if 'scan' in loc_encoder:  # ['scan', 'zigzag'', 'zorder', 'hilbert']
            locs_flat = indexes
        elif 'zigzag' in loc_encoder:
            indexes = indexes.reshape(size, size)
            for i in np.arange(1, size, step=2):
                indexes[i, :] = indexes[i, :][::-1]
            locs_flat = indexes.reshape(-1)
        elif 'zorder' in loc_encoder:
            zi = ZOrderIndexer((0, size - 1), (0, size - 1))
            locs_flat = []
            for z in indexes:
                r, c = zi.rc(int(z))
                locs_flat.append(c * size + r)
            locs_flat = np.array(locs_flat)
        elif 'hilbert' in loc_encoder:
            locs = decode(indexes, dim, bit)
            locs_flat = self.flat_locs_hilbert(locs, dim, bit)
            locs_unflat = self.unflat_locs_hilbert(locs_flat, dim, bit)
            self.index_unflat = torch.LongTensor(locs_unflat.astype(np.int64)).unsqueeze(0).unsqueeze(2)
        else:
            raise Exception('invalid encoder mode')
        locs_flat_inv = np.argsort(locs_flat)
        index_flat = torch.LongTensor(locs_flat.astype(np.int64)).unsqueeze(0).unsqueeze(2)
        index_flat_inv = torch.LongTensor(locs_flat_inv.astype(np.int64)).unsqueeze(0).unsqueeze(2)
        self.index_flat = nn.Parameter(index_flat, requires_grad=False)
        self.index_flat_inv = nn.Parameter(index_flat_inv, requires_grad=False)


    def flat_locs_hilbert(self, locs, num_dim, num_bit):
        ret = []
        l = 2 ** num_bit
        for i in range(len(locs)):
            loc = locs[i]
            loc_flat = 0
            for j in range(num_dim):
                loc_flat += loc[j] * (l ** j)
            ret.append(loc_flat)
        return np.array(ret).astype(np.uint64)

    def unflat_locs_hilbert(self, locs, num_dim, num_bit):
        ret = []
        l = 2 ** num_bit
        for i in range(len(locs)):
            loc = locs[i]
            loc_unflat = []
            for j in range(num_dim):
                loc_unflat.insert(0, loc // (l ** (num_dim - 1 - j)))
                loc = loc % (l ** (num_dim - 1 - j))
            ret.append(loc_unflat)
        return np.array(ret).astype(np.uint64)

    def __call__(self, img):
        img_encode = self.encode(img)
        return img_encode

    def encode(self, img):
        img_encode = torch.zeros(img.shape, dtype=img.dtype, device=img.device).scatter_(1, self.index_flat_inv.expand(img.shape), img)
        return img_encode

    def decode(self, img):
        img_decode = torch.zeros(img.shape, dtype=img.dtype, device=img.device).scatter_(1, self.index_flat.expand(img.shape), img)
        return img_decode


class RepVGG_1D(nn.Module):

    def __init__(self, cfg, img_size=(224,), num_classes=1000, in_chans=3, drop_rate=0.0):
        super(RepVGG_1D, self).__init__()
        self.num_classes = num_classes

        num_blocks = cfg['num_blocks']
        width_multiplier = cfg['width_multiplier']
        assert len(width_multiplier) == 4
        self.kernel_size = cfg['kernel_size']
        self.padding = cfg['padding']
        self.encoder_mode = cfg['encoder_mode']

        bit = int(math.log2(img_size[0]))
        self.loc_encoder = LocED(bit=bit, dim=2, loc_encoder=self.encoder_mode)

        self.feature_info = []

        self.in_planes = min(64, int(64 * width_multiplier[0]))
        self.stage0 = SkipConvNormActi_1d(in_channels=in_chans, out_channels=self.in_planes, kernel_size=self.kernel_size, stride=4, padding=self.padding)
        self.layer_idx = 0
        self.stage1 = self._make_stage(int(64 * width_multiplier[0]), num_blocks[0], stride=4)
        self.stage2 = self._make_stage(int(128 * width_multiplier[1]), num_blocks[1], stride=4)
        self.stage3 = self._make_stage(int(256 * width_multiplier[2]), num_blocks[2], stride=4)
        self.stage4 = self._make_stage(int(512 * width_multiplier[3]), num_blocks[3], stride=4)
        self.gap = nn.AdaptiveAvgPool1d(output_size=1)
        self.dropout = nn.Dropout(p=drop_rate)
        self.linear = nn.Linear(int(512 * width_multiplier[3]), num_classes)

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        blocks = []
        for stride in strides:
            blocks.append(SkipConvNormActi_1d(in_channels=self.in_planes, out_channels=planes, kernel_size=self.kernel_size, stride=stride, padding=self.padding))
            self.in_planes = planes
        return nn.Sequential(*blocks)

    def forward(self, x):
        B, C, H, W = x.shape
        if self.encoder_mode.find('_t') > -1:
            x = x.permute(0, 3, 2, 1).reshape(B, W * H, C)
        else:
            x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x = self.loc_encoder.encode(x)
        x = x.permute(0, 2, 1)
        out = self.stage0(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.linear(out)
        return out


def _create_repvgg_2d(variant: str, pretrained: bool, **kwargs: Any) -> RepVGG_2D:
    default_cfg = deepcopy(default_cfgs[variant])
    default_img_size = default_cfg['input_size'][-2:]
    img_size = kwargs.pop('img_size', default_img_size)

    model = build_model_with_cfg(
        RepVGG_2D, variant, pretrained,
        default_cfg=default_cfgs[variant],
        model_cfg=model_cfgs[variant],
        img_size=img_size,
        **kwargs)

    return model


def _create_repvgg_1d(variant: str, pretrained: bool, **kwargs: Any) -> RepVGG_1D:
    default_cfg = deepcopy(default_cfgs[variant])
    default_img_size = default_cfg['input_size'][-2:]
    img_size = kwargs.pop('img_size', default_img_size)

    model = build_model_with_cfg(
        RepVGG_1D, variant, pretrained,
        default_cfg=default_cfgs[variant],
        model_cfg=model_cfgs[variant],
        img_size=img_size,
        **kwargs)

    return model


# 2D
@register_model
def repvgg_a0(pretrained: bool = False, **kwargs: Any) -> RepVGG_2D:
    model_args = dict(**kwargs)
    return _create_repvgg_2d('repvgg_a0', pretrained=pretrained, **model_args)

@register_model
def repvgg_b1(pretrained: bool = False, **kwargs: Any) -> RepVGG_2D:
    model_args = dict(**kwargs)
    return _create_repvgg_2d('repvgg_b1', pretrained=pretrained, **model_args)

@register_model
def repvgg_b1_3x1(pretrained: bool = False, **kwargs: Any) -> RepVGG_2D:
    model_args = dict(**kwargs)
    return _create_repvgg_2d('repvgg_b1_3x1', pretrained=pretrained, **model_args)

@register_model
def repvgg_b1_1x3(pretrained: bool = False, **kwargs: Any) -> RepVGG_2D:
    model_args = dict(**kwargs)
    return _create_repvgg_2d('repvgg_b1_1x3', pretrained=pretrained, **model_args)

@register_model
def repvgg_b1_3x1_1x3(pretrained: bool = False, **kwargs: Any) -> RepVGG_2D:
    model_args = dict(**kwargs)
    return _create_repvgg_2d('repvgg_b1_3x1_1x3', pretrained=pretrained, **model_args)

@register_model
def repvgg_b1_2x2(pretrained: bool = False, **kwargs: Any) -> RepVGG_2D:
    model_args = dict(**kwargs)
    return _create_repvgg_2d('repvgg_b1_2x2', pretrained=pretrained, **model_args)

@register_model
def repvgg_b1_3x1_pooling(pretrained: bool = False, **kwargs: Any) -> RepVGG_2D:
    model_args = dict(**kwargs)
    return _create_repvgg_2d('repvgg_b1_3x1_pooling', pretrained=pretrained, **model_args)

@register_model
def repvgg_b1_3x1_1x3_pooling(pretrained: bool = False, **kwargs: Any) -> RepVGG_2D:
    model_args = dict(**kwargs)
    return _create_repvgg_2d('repvgg_b1_3x1_1x3_pooling', pretrained=pretrained, **model_args)

# 1D
@register_model
def repvgg_b1_3x1_scan(pretrained: bool = False, **kwargs: Any) -> RepVGG_1D:
    model_args = dict(**kwargs)
    return _create_repvgg_1d('repvgg_b1_3x1_scan', pretrained=pretrained, **model_args)


@register_model
def repvgg_b1_3x1_zigzag(pretrained: bool = False, **kwargs: Any) -> RepVGG_1D:
    model_args = dict(**kwargs)
    return _create_repvgg_1d('repvgg_b1_3x1_zigzag', pretrained=pretrained, **model_args)


@register_model
def repvgg_b1_3x1_zorder(pretrained: bool = False, **kwargs: Any) -> RepVGG_1D:
    model_args = dict(**kwargs)
    return _create_repvgg_1d('repvgg_b1_3x1_zorder', pretrained=pretrained, **model_args)


@register_model
def repvgg_b1_3x1_hilbert(pretrained: bool = False, **kwargs: Any) -> RepVGG_1D:
    model_args = dict(**kwargs)
    return _create_repvgg_1d('repvgg_b1_3x1_hilbert', pretrained=pretrained, **model_args)

