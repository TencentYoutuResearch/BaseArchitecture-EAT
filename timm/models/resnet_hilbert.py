"""PyTorch ResNet

This started as a copy of https://github.com/pytorch/vision 'resnet.py' (BSD-3-Clause) with
additional dropout and dynamic global avg/max pool.

ResNeXt, SE-ResNeXt, SENet, and MXNet Gluon stem/downsample variants, tiered stems added by Ross Wightman
Copyright 2020 Ross Wightman
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .helpers import build_model_with_cfg
from .layers import DropBlock2d, DropPath, AvgPool2dSame, BlurPool2d, create_attn, create_classifier
from .registry import register_model

__all__ = ['ResNet', 'BasicBlock', 'Bottleneck']  # model_registry will add each entrypoint fn to this


def _mcfg(**kwargs):
    cfg = dict()
    cfg.update(**kwargs)
    return cfg


# Model FLOPS = three trailing digits * 10^8
model_cfgs = dict(
    resnet18_hil=_mcfg(kernel_size=(3, 3), padding=(1, 1)),
    resnet34_hil=_mcfg(kernel_size=(3, 3), padding=(1, 1)),
    resnet26_hil=_mcfg(kernel_size=(3, 3), padding=(1, 1)),
    resnet50_hil=_mcfg(kernel_size=(3, 3), padding=(1, 1)),
    resnet101_hil=_mcfg(kernel_size=(3, 3), padding=(1, 1)),
    resnet152_hil=_mcfg(kernel_size=(3, 3), padding=(1, 1)),
    resnet200_hil=_mcfg(kernel_size=(3, 3), padding=(1, 1)),
    resnet50_3x1_hil=_mcfg(kernel_size=(3, 1), padding=(1, 0)),
)

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv1', 'classifier': 'fc',
        **kwargs
    }


default_cfgs = {
    # ResNet and Wide ResNet
    'resnet18_hil': _cfg(url=''),
    'resnet34_hil': _cfg(url=''),
    'resnet26_hil': _cfg(url='', interpolation='bicubic'),
    'resnet50_hil': _cfg(url='', interpolation='bicubic'),
    'resnet101_hil': _cfg(url='', interpolation='bicubic'),
    'resnet152_hil': _cfg(url='', interpolation='bicubic'),
    'resnet200_hil': _cfg(url='', interpolation='bicubic'),

    'resnet50_3x1_hil': _cfg(url='', interpolation='bicubic', kernel_size=(3, 1), padding=(1, 0)),
}


def get_padding(kernel_size, stride, dilation=1):
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, cfg, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64,
                 reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
                 attn_layer=None, aa_layer=None, drop_block=None, drop_path=None):
        super(BasicBlock, self).__init__()

        assert cardinality == 1, 'BasicBlock only supports cardinality of 1'
        assert base_width == 64, 'BasicBlock does not support changing base width'
        first_planes = planes // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        self.conv1 = nn.Conv2d(
            inplanes, first_planes, kernel_size=cfg['kernel_size'], stride=1 if use_aa else stride, padding=cfg['padding'],
            dilation=first_dilation, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)
        self.aa = aa_layer(channels=first_planes, stride=stride) if use_aa else None

        self.conv2 = nn.Conv2d(
            first_planes, outplanes, kernel_size=cfg['kernel_size'], padding=cfg['padding'], dilation=dilation, bias=False)
        self.bn2 = norm_layer(outplanes)

        self.se = create_attn(attn_layer, outplanes)

        self.act2 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn2.weight)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)
        if self.aa is not None:
            x = self.aa(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            residual = self.downsample(residual)
        x += residual
        x = self.act2(x)

        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, cfg, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64,
                 reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
                 attn_layer=None, aa_layer=None, drop_block=None, drop_path=None):
        super(Bottleneck, self).__init__()

        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        self.conv1 = nn.Conv2d(inplanes, first_planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(
            first_planes, width, kernel_size=cfg['kernel_size'], stride=1 if use_aa else stride,
            padding=cfg['padding'], dilation=first_dilation, groups=cardinality, bias=False)
        self.bn2 = norm_layer(width)
        self.act2 = act_layer(inplace=True)
        self.aa = aa_layer(channels=width, stride=stride) if use_aa else None

        self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)

        self.se = create_attn(attn_layer, outplanes)

        self.act3 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act2(x)
        if self.aa is not None:
            x = self.aa(x)

        x = self.conv3(x)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            residual = self.downsample(residual)
        x += residual
        x = self.act3(x)

        return x


def downsample_conv(
        in_channels, out_channels, kernel_size, stride=1, dilation=1, first_dilation=None, norm_layer=None):
    norm_layer = norm_layer or nn.BatchNorm2d
    kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
    first_dilation = (first_dilation or dilation) if kernel_size > 1 else 1
    p = get_padding(kernel_size, stride, first_dilation)

    return nn.Sequential(*[
        nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=p, dilation=first_dilation, bias=False),
        norm_layer(out_channels)
    ])


def downsample_avg(
        in_channels, out_channels, kernel_size, stride=1, dilation=1, first_dilation=None, norm_layer=None):
    norm_layer = norm_layer or nn.BatchNorm2d
    avg_stride = stride if dilation == 1 else 1
    if stride == 1 and dilation == 1:
        pool = nn.Identity()
    else:
        avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
        pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)

    return nn.Sequential(*[
        pool,
        nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
        norm_layer(out_channels)
    ])


def drop_blocks(drop_block_rate=0.):
    return [
        None, None,
        DropBlock2d(drop_block_rate, 5, 0.25) if drop_block_rate else None,
        DropBlock2d(drop_block_rate, 3, 1.00) if drop_block_rate else None]


def make_blocks(
        cfg, block_fn, channels, block_repeats, inplanes, reduce_first=1, output_stride=32,
        down_kernel_size=1, avg_down=False, drop_block_rate=0., drop_path_rate=0., **kwargs):
    stages = []
    feature_info = []
    net_num_blocks = sum(block_repeats)
    net_block_idx = 0
    net_stride = 4
    dilation = prev_dilation = 1
    for stage_idx, (planes, num_blocks, db) in enumerate(zip(channels, block_repeats, drop_blocks(drop_block_rate))):
        stage_name = f'layer{stage_idx + 1}'  # never liked this name, but weight compat requires it
        stride = 1 if stage_idx == 0 else 2
        if net_stride >= output_stride:
            dilation *= stride
            stride = 1
        else:
            net_stride *= stride

        downsample = None
        if stride != 1 or inplanes != planes * block_fn.expansion:
            down_kwargs = dict(
                in_channels=inplanes, out_channels=planes * block_fn.expansion, kernel_size=down_kernel_size,
                stride=stride, dilation=dilation, first_dilation=prev_dilation, norm_layer=kwargs.get('norm_layer'))
            downsample = downsample_avg(**down_kwargs) if avg_down else downsample_conv(**down_kwargs)

        block_kwargs = dict(reduce_first=reduce_first, dilation=dilation, drop_block=db, **kwargs)
        blocks = []
        for block_idx in range(num_blocks):
            downsample = downsample if block_idx == 0 else None
            stride = stride if block_idx == 0 else 1
            block_dpr = drop_path_rate * net_block_idx / (net_num_blocks - 1)  # stochastic depth linear decay rule
            blocks.append(block_fn(cfg, inplanes, planes, stride, downsample, first_dilation=prev_dilation,
                                   drop_path=DropPath(block_dpr) if block_dpr > 0. else None, **block_kwargs))
            prev_dilation = dilation
            inplanes = planes * block_fn.expansion
            net_block_idx += 1

        stages.append((stage_name, nn.Sequential(*blocks)))
        feature_info.append(dict(num_chs=inplanes, reduction=net_stride, module=stage_name))

    return stages, feature_info


class ResNet(nn.Module):

    def __init__(self, cfg, block, layers, num_classes=1000, in_chans=3,
                 cardinality=1, base_width=64, stem_width=64, stem_type='',
                 output_stride=32, block_reduce_first=1, down_kernel_size=1, avg_down=False,
                 act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, aa_layer=None, drop_rate=0.0, drop_path_rate=0.,
                 drop_block_rate=0., global_pool='avg', zero_init_last_bn=True, block_args=None):
        block_args = block_args or dict()
        assert output_stride in (8, 16, 32)
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        super(ResNet, self).__init__()

        inplanes = 64
        # self.conv1 = nn.Conv2d(in_chans, inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1 = nn.Sequential(*[nn.Conv2d(in_chans, inplanes, kernel_size=cfg['kernel_size'], stride=2, padding=cfg['padding'], bias=False),
                                     norm_layer(inplanes),
                                     act_layer(inplace=True),
                                     nn.Conv2d(inplanes, inplanes, kernel_size=cfg['kernel_size'], stride=1, padding=cfg['padding'], bias=False),
                                     norm_layer(inplanes),
                                     act_layer(inplace=True),
                                     nn.Conv2d(inplanes, inplanes, kernel_size=cfg['kernel_size'], stride=1, padding=cfg['padding'], bias=False),
                                     ])
        self.bn1 = norm_layer(inplanes)
        self.act1 = act_layer(inplace=True)
        self.feature_info = [dict(num_chs=inplanes, reduction=2, module='act1')]
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Feature Blocks
        channels = [64, 128, 256, 512]
        stage_modules, stage_feature_info = make_blocks(
            cfg, block, channels, layers, inplanes, cardinality=cardinality, base_width=base_width,
            output_stride=output_stride, reduce_first=block_reduce_first, avg_down=avg_down,
            down_kernel_size=down_kernel_size, act_layer=act_layer, norm_layer=norm_layer, aa_layer=aa_layer,
            drop_block_rate=drop_block_rate, drop_path_rate=drop_path_rate, **block_args)
        for stage in stage_modules:
            self.add_module(*stage)  # layer1, layer2, etc
        self.feature_info.extend(stage_feature_info)

        # Head (Pooling and Classifier)
        self.num_features = 512 * block.expansion
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
        if zero_init_last_bn:
            for m in self.modules():
                if hasattr(m, 'zero_init_last_bn'):
                    m.zero_init_last_bn()

    def get_classifier(self):
        return self.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

    def forward_features(self, x):
        x = self.conv1(x)  # torch.Size([2, 3, 224, 224]) -> torch.Size([2, 64, 112, 112])
        x = self.bn1(x)  # torch.Size([2, 64, 112, 112]) -> torch.Size([2, 64, 112, 112])
        x = self.act1(x)  # torch.Size([2, 64, 112, 112]) -> torch.Size([2, 64, 112, 112])
        x = self.maxpool(x)  # torch.Size([2, 64, 112, 112]) -> torch.Size([2, 64, 56, 56])

        x = self.layer1(x)  # torch.Size([2, 64, 56, 56]) -> torch.Size([2, 256, 56, 56])
        x = self.layer2(x)  # torch.Size([2, 256, 56, 56]) -> torch.Size([2, 512, 28, 28])
        x = self.layer3(x)  # torch.Size([2, 512, 28, 28]) -> torch.Size([2, 1024, 14, 14])
        x = self.layer4(x)  # torch.Size([2, 1024, 14, 14]) -> torch.Size([2, 2048, 7, 7])
        return x

    def forward(self, x):
        x = self.forward_features(x)  # torch.Size([2, 2048, 7, 7])
        x = self.global_pool(x)  # torch.Size([2, 2048, 7, 7]) -> torch.Size([2, 2048])
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        x = self.fc(x)
        return x


def _create_resnet(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(ResNet, variant, pretrained, default_cfg=default_cfgs[variant], model_cfg=model_cfgs[variant], **kwargs)


@register_model
def resnet18_hil(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2], **kwargs)
    return _create_resnet('resnet18_hil', pretrained, **model_args)


@register_model
def resnet34_hil(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    """
    model_args = dict(block=BasicBlock, layers=[3, 4, 6, 3], **kwargs)
    return _create_resnet('resnet34_hil', pretrained, **model_args)


@register_model
def resnet2_hil(pretrained=False, **kwargs):
    """Constructs a ResNet-26 model.
    """
    model_args = dict(block=Bottleneck, layers=[2, 2, 2, 2], **kwargs)
    return _create_resnet('resnet26_hil', pretrained, **model_args)


@register_model
def resnet50_hil(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3],  **kwargs)
    return _create_resnet('resnet50_hil', pretrained, **model_args)


@register_model
def resnet101_hil(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], **kwargs)
    return _create_resnet('resnet101_hil', pretrained, **model_args)


@register_model
def resnet152_hil(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 8, 36, 3], **kwargs)
    return _create_resnet('resnet152_hil', pretrained, **model_args)


@register_model
def resnet200_hil(pretrained=False, **kwargs):
    """Constructs a ResNet-200 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 24, 36, 3], **kwargs)
    return _create_resnet('resnet200_hil', pretrained, **model_args)


@register_model
def resnet50_3x1_hil(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3],  **kwargs)
    return _create_resnet('resnet50_3x1_hil', pretrained, **model_args)