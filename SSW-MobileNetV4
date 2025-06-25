from thop import profile
from timm.models import register_model

from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.model_config import MODEL_SPECS
from mmengine.model import BaseModule
from einops import rearrange
from wtconv.wtconv2d  import WTConv2d

import typing as t

#MCF+SCSA+WTC
#3,3,3

def make_divisible(
        value: float,
        divisor: int,
        min_value: Optional[float] = None,
        round_down_protect: bool = True,
) -> int:
   
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if round_down_protect and new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)


def conv2d(in_channels, out_channels, kernel_size=3, stride=1, groups=1, bias=False, norm=True, act=True):
    conv = nn.Sequential()
    padding = (kernel_size - 1) // 2
    conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias, groups=groups))
    if norm:
        conv.append(nn.BatchNorm2d(out_channels))
    if act:
        conv.append(nn.ReLU6())
    return conv

class Conv2D(nn.Module):
    def __init__(self, c2=4, k=3):
        super(Conv2D, self).__init__()
        # 2D卷积层
        self.c = nn.Conv2d(3, c2, k, stride=1, padding=k//2)
        # 2D批归一化层
        self.bn = nn.BatchNorm2d(c2)
        # ReLU激活层
        self.act = nn.ReLU()

    def forward(self, x):
        # 顺序执行卷积，归一化，激活
        return self.act(self.bn(self.c(x)))

class MCF2D(nn.Module):
    def __init__(self, c2=12, ks=(3, 3, 3)):
        super(MCF2D, self).__init__()
        self.m = nn.ModuleList(Conv2D(c2, k) for k in ks)

    def forward(self, x):
        # 通过每个卷积层处理输入，并将结果储存到 fea 中
        fea = [conv(x) for conv in self.m]
        # 通道维度拼接
        fea = torch.cat(fea, dim=1)
        return fea

class SCSA(BaseModule):
    def __init__(
        self,
        dim: int,
        head_num: int,
        window_size: int = 7,
        group_kernel_sizes: t.List[int] = [3, 5, 7, 9],
        qkv_bias: bool = False,
        fuse_bn: bool = False,
        norm_cfg: t.Dict = dict(type='BN'),
        act_cfg: t.Dict = dict(type='ReLU'),
        down_sample_mode: str = 'avg_pool',
        attn_drop_ratio: float = 0.,
        gate_layer: str = 'sigmoid',
        **kwargs
    ):
        super(SCSA, self).__init__()
        self.dim = dim
        self.head_num = head_num
        self.head_dim = dim // head_num
        self.scaler = self.head_dim ** -0.5
        self.group_kernel_sizes = group_kernel_sizes
        self.window_size = window_size
        self.qkv_bias = qkv_bias
        self.fuse_bn = fuse_bn
        self.down_sample_mode = down_sample_mode

        assert self.dim % 4 == 0, 'The dimension of input feature should be divisible by 4.'
        self.group_chans = group_chans = self.dim // 4

        self.local_dwc = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[0],
                                   padding=group_kernel_sizes[0] // 2, groups=group_chans)
        self.global_dwc_s = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[1],
                                      padding=group_kernel_sizes[1] // 2, groups=group_chans)
        self.global_dwc_m = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[2],
                                      padding=group_kernel_sizes[2] // 2, groups=group_chans)
        self.global_dwc_l = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[3],
                                      padding=group_kernel_sizes[3] // 2, groups=group_chans)
        self.sa_gate = nn.Softmax(dim=2) if gate_layer == 'softmax' else nn.Sigmoid()
        self.norm_h = nn.GroupNorm(4, dim)
        self.norm_w = nn.GroupNorm(4, dim)

        self.conv_d = nn.Identity()
        self.norm = nn.GroupNorm(1, dim)
        self.q = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.k = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.v = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.ca_gate = nn.Softmax(dim=1) if gate_layer == 'softmax' else nn.Sigmoid()

        if window_size == -1:
            self.down_func = nn.AdaptiveAvgPool2d((1, 1))
        else:
            if down_sample_mode == 'recombination':
                self.down_func = self.space_to_chans
                # dimensionality reduction
                self.conv_d = nn.Conv2d(in_channels=dim * window_size ** 2, out_channels=dim, kernel_size=1, bias=False)
            elif down_sample_mode == 'avg_pool':
                self.down_func = nn.AvgPool2d(kernel_size=(window_size, window_size), stride=window_size)
            elif down_sample_mode == 'max_pool':
                self.down_func = nn.MaxPool2d(kernel_size=(window_size, window_size), stride=window_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The dim of x is (B, C, H, W)
        """
        # Spatial attention priority calculation

        b, c, h_, w_ = x.size()

        assert c == self.dim, f"Input channels {c} do not match expected dim {self.dim}"
        # (B, C, H)
        x_h = x.mean(dim=3)
        # (B, C, W)
        x_w = x.mean(dim=2)

        # 动态调整 torch.split 的分割数量
        num_groups = c // self.group_chans
        splits = torch.split(x_h, self.group_chans, dim=1)
        if num_groups > 4:
            # 如果分割数量超过 4，将多余的组合并到最后一组
            splits = splits[:3] + (torch.cat(splits[3:], dim=1),)
        l_x_h, g_x_h_s, g_x_h_m, g_x_h_l = splits

        splits = torch.split(x_w, self.group_chans, dim=1)
        if num_groups > 4:
            splits = splits[:3] + (torch.cat(splits[3:], dim=1),)
        l_x_w, g_x_w_s, g_x_w_m, g_x_w_l = splits

        x_h_attn = self.sa_gate(self.norm_h(torch.cat((
            self.local_dwc(l_x_h),
            self.global_dwc_s(g_x_h_s),
            self.global_dwc_m(g_x_h_m),
            self.global_dwc_l(g_x_h_l),
        ), dim=1)))
        x_h_attn = x_h_attn.view(b, c, h_, 1)

        x_w_attn = self.sa_gate(self.norm_w(torch.cat((
            self.local_dwc(l_x_w),
            self.global_dwc_s(g_x_w_s),
            self.global_dwc_m(g_x_w_m),
            self.global_dwc_l(g_x_w_l)
        ), dim=1)))
        x_w_attn = x_w_attn.view(b, c, 1, w_)

        x = x * x_h_attn * x_w_attn

        # Channel attention based on self attention
        # reduce calculations
        y = self.down_func(x)
        y = self.conv_d(y)
        _, _, h_, w_ = y.size()

        # normalization first, then reshape -> (B, H, W, C) -> (B, C, H * W) and generate q, k and v
        y = self.norm(y)
        q = self.q(y)
        k = self.k(y)
        v = self.v(y)
        # (B, C, H, W) -> (B, head_num, head_dim, N)
        q = rearrange(q, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))
        k = rearrange(k, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))
        v = rearrange(v, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))

        # (B, head_num, head_dim, head_dim)
        attn = q @ k.transpose(-2, -1) * self.scaler
        attn = self.attn_drop(attn.softmax(dim=-1))
        # (B, head_num, head_dim, N)
        attn = attn @ v
        # (B, C, H_, W_)
        attn = rearrange(attn, 'b head_num head_dim (h w) -> b (head_num head_dim) h w', h=int(h_), w=int(w_))
        # (B, C, 1, 1)
        attn = attn.mean((2, 3), keepdim=True)
        attn = self.ca_gate(attn)
        return attn * x


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio, act=False, squeeze_exactation=False):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = int(round(in_channels * expand_ratio))
        self.block = nn.Sequential()
        if expand_ratio != 1:
            self.block.add_module("exp_1x1", conv2d(in_channels, hidden_dim, kernel_size=3, stride=stride))
        if squeeze_exactation:
            self.block.add_module("conv_3x3", conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, groups=hidden_dim))
        self.block.add_module("res_1x1", conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, act=act))
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

    def forward(self, x):
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)


class UniversalInvertedBottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, start_dw_kernel_size, middle_dw_kernel_size, middle_dw_downsample,
                 stride, expand_ratio, use_scsa=False, scsa_params=None):
        super(UniversalInvertedBottleneckBlock, self).__init__()

        # starting depthwise conv
        self.start_dw_kernel_size = start_dw_kernel_size
        if self.start_dw_kernel_size:
            stride_ = stride if not middle_dw_downsample else 1
            self._start_dw_ = conv2d(in_channels, in_channels, kernel_size=start_dw_kernel_size, stride=stride_,
                                     groups=in_channels, act=False)

        # expansion with 1x1 convs
        expand_filters = make_divisible(in_channels * expand_ratio, 8)
        self._expand_conv = conv2d(in_channels, expand_filters, kernel_size=1)

        # middle depthwise conv
        self.middle_dw_kernel_size = middle_dw_kernel_size
        if self.middle_dw_kernel_size:
            stride_ = stride if middle_dw_downsample else 1
            self._middle_dw = conv2d(expand_filters, expand_filters, kernel_size=middle_dw_kernel_size, stride=stride_,
                                     groups=expand_filters)

        # 添加 SCSA 注意力层
        self.use_scsa = use_scsa
        if self.use_scsa:
            scsa_params = scsa_params or {}
            scsa_params["dim"] = expand_filters  # 让 SCSA 维度与 expand_filters 匹配
            self.scsa = SCSA(**scsa_params)

        # projection with 1x1 convs (需要保证输入通道数一致)
        self._proj_conv = conv2d(expand_filters, out_channels, kernel_size=1, stride=1, act=False)

    def forward(self, x):
        if self.start_dw_kernel_size:
            x = self._start_dw_(x)
        x = self._expand_conv(x)
        if self.middle_dw_kernel_size:
            x = self._middle_dw(x)
            if self.use_scsa:
                x = self.scsa(x)  # 确保输入 SCSA 的通道数匹配 expand_filters
        x = self._proj_conv(x)  # 确保 SCSA 的输出通道数匹配 _proj_conv 的输入通道数
        return x

class MultiQueryAttentionLayerWithDownSampling(nn.Module):
    def __init__(self, in_channels, num_heads, key_dim, value_dim, query_h_strides, query_w_strides, kv_strides, dw_kernel_size=3, dropout=0.0):
       
        super(MultiQueryAttentionLayerWithDownSampling, self).__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.query_h_strides = query_h_strides
        self.query_w_strides = query_w_strides
        self.kv_strides = kv_strides
        self.dw_kernel_size = dw_kernel_size
        self.dropout = dropout

        self.head_dim = self.key_dim // num_heads

        if self.query_h_strides > 1 or self.query_w_strides > 1:
            self._query_downsampling_norm = nn.BatchNorm2d(in_channels)
        self._query_proj = conv2d(in_channels, self.num_heads * self.key_dim, 1, 1, norm=False, act=False)

        if self.kv_strides > 1:
            self._key_dw_conv = conv2d(in_channels, in_channels, dw_kernel_size, kv_strides, groups=in_channels,
                                       norm=True, act=False)
            self._value_dw_conv = conv2d(in_channels, in_channels, dw_kernel_size, kv_strides, groups=in_channels,
                                         norm=True, act=False)
        self._key_proj = conv2d(in_channels, key_dim, 1, 1, norm=False, act=False)
        self._value_proj = conv2d(in_channels, key_dim, 1, 1, norm=False, act=False)
        self._output_proj = conv2d(num_heads * key_dim, in_channels, 1, 1, norm=False, act=False)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        bs, c,h,w = x.size()
        # print(x.size())

        query_h_strides = min(self.query_h_strides, h)
        query_w_strides = min(self.query_w_strides, w)

        if self.query_h_strides > 1 or self.query_w_strides > 1:
            x_q = F.avg_pool2d(x,kernel_size=(self.query_h_strides, self.query_w_strides))
            q = self._query_downsampling_norm(x_q)
            q = self._query_proj(q)
        else:
            q = self._query_proj(x)
        px = q.size(2)
        q = q.view(bs, self.num_heads, -1, self.key_dim)  # [batch_size, num_heads, seq_len, key_dim]



        if self.kv_strides > 1:
            k = self._key_dw_conv(x)
            k = self._key_proj(k)
            v = self._value_dw_conv(x)
            v = self._value_proj(v)
        else:
            k = self._key_proj(x)
            v = self._value_proj(x)
        k = k.view(bs, 1, self.key_dim, -1)   # [batch_size, 1, key_dim, seq_length]
        v = v.view(bs, 1, -1, self.key_dim)    # [batch_size, 1, seq_length, key_dim]

        # calculate attention score
        # print(q.shape, k.shape, v.shape)
        attn_score = torch.matmul(q, k) / (self.head_dim ** 0.5)
        attn_score = self.dropout(attn_score)
        attn_score = F.softmax(attn_score, dim=-1)

        # context = torch.einsum('bnhm,bmv->bnhv', attn_score, v)
        # print(attn_score.shape, v.shape)
        context = torch.matmul(attn_score, v)
        context = context.view(bs, self.num_heads * self.key_dim, px, px)
        output = self._output_proj(context)
        # print(output.shape)
        return output


class MNV4layerScale(nn.Module):
    def __init__(self, init_value):
        
        super(MNV4layerScale, self).__init__()
        self.init_value = init_value

    def forward(self, x):
        gamma = self.init_value * torch.ones(x.size(-1), dtype=x.dtype, device=x.device)
        return x * gamma


class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(self, in_channels, num_heads, key_dim, value_dim, query_h_strides, query_w_strides,
                 kv_strides, use_layer_scale, use_multi_query, use_residual=True):
        super(MultiHeadSelfAttentionBlock, self).__init__()
        self.query_h_strides = query_h_strides
        self.query_w_strides = query_w_strides
        self.kv_strides = kv_strides
        self.use_layer_scale = use_layer_scale
        self.use_multi_query = use_multi_query
        self.use_residual = use_residual
        self._input_norm = nn.BatchNorm2d(in_channels)

        if self.use_multi_query:
            self.multi_query_attention = MultiQueryAttentionLayerWithDownSampling(
                in_channels, num_heads, key_dim, value_dim, query_h_strides, query_w_strides, kv_strides
            )
        else:
            self.multi_head_attention = nn.MultiheadAttention(in_channels, num_heads, kdim=key_dim)

        if use_layer_scale:
            self.layer_scale_init_value = 1e-5
            self.layer_scale = MNV4layerScale(self.layer_scale_init_value)

    def forward(self, x):
        # Not using CPE, skipped
        # input norm
        shortcut = x
        x = self._input_norm(x)
        # multi query
        if self.use_multi_query:
            # print(x.size())
            x = self.multi_query_attention(x)
            # print(x.size())
        else:
            x = self.multi_head_attention(x, x)
        # layer scale
        if self.use_layer_scale:
            x = self.layer_scale(x)
        # use residual
        if self.use_residual:
            x = x + shortcut
        return x


def build_blocks(layer_spec):
    global msha
    if not layer_spec.get("block_name"):
        return nn.Sequential()
    block_names = layer_spec["block_name"]
    layers = nn.Sequential()
    if block_names == "convbn":
        schema_ = ["in_channels", "out_channels", "kernel_size", "stride", "groups", "bias", "norm", "act"]
        for i in range(layer_spec["num_blocks"]):
            args = dict(zip(schema_, layer_spec["block_specs"][i]))
            if "mcf" in layer_spec and layer_spec["mcf"]:  # 判断是否使用 MCF
                layers.add_module(f"conv0_mcf", MCF2D(c2=args["out_channels"], ks=(3, 3, 3)))
            else:
                layers.add_module(f"convbn_{i}", conv2d(**args))
    elif block_names == "uib":
        schema_ = ["in_channels", "out_channels", "start_dw_kernel_size", "middle_dw_kernel_size", "middle_dw_downsample",
                   "stride", "expand_ratio", "msha", "use_scsa", "scsa_params"]
        for i in range(layer_spec["num_blocks"]):
            args = dict(zip(schema_, layer_spec["block_specs"][i]))
            msha = args.pop("msha") if "msha" in args else 0
            use_scsa = args.pop("use_scsa") if "use_scsa" in args else False
            scsa_params = args.pop("scsa_params") if "scsa_params" in args else None
            layers.add_module(f"uib_{i}", UniversalInvertedBottleneckBlock(**args, use_scsa=use_scsa, scsa_params=scsa_params))
            if msha:
                msha_schema_ = [
                    "in_channels", "num_heads", "key_dim", "value_dim", "query_h_strides", "query_w_strides", "kv_strides",
                    "use_layer_scale", "use_multi_query", "use_residual"
                ]
                args = dict(zip(msha_schema_, [args["out_channels"]] + (msha)))
                layers.add_module(
                    f"msha_{i}", MultiHeadSelfAttentionBlock(**args)
                )
    elif block_names == "fused_ib":
        schema_ = ["in_channels", "out_channels", "stride", "expand_ratio", "act"]
        for i in range(layer_spec["num_blocks"]):
            args = dict(zip(schema_, layer_spec["block_specs"][i]))
            layers.add_module(f"fused_ib_{i}", InvertedResidual(**args))
    else:
        raise NotImplementedError
    return layers


class MobileNetV4(nn.Module):
    def __init__(self, model, num_classes=18, **kwargs):
        # MobileNetV4ConvSmall  MobileNetV4ConvMedium  MobileNetV4ConvLarge
        # MobileNetV4HybridMedium  MobileNetV4HybridLarge
      
        super(MobileNetV4, self).__init__()
        # print(MODEL_SPECS.keys(), model not in MODEL_SPECS.keys())
        assert model in MODEL_SPECS.keys()
        self.model = model
        self.num_classes = num_classes
        self.spec = MODEL_SPECS[self.model]

        # conv0
        self.conv0 = build_blocks(self.spec["conv0"])
        # layer1
        self.layer1 = build_blocks(self.spec["layer1"])
        # layer2
        self.layer2 = build_blocks(self.spec["layer2"])
        # layer3
        self.layer3 = build_blocks(self.spec["layer3"])

        self.wavelet_conv = WTConv2d(in_channels=96, out_channels=96, kernel_size=5, stride=1, wt_levels=1,
                                     wt_type='db1')
        # layer4
        self.layer4 = build_blocks(self.spec["layer4"])
        # layer5
        self.layer5 = build_blocks(self.spec["layer5"])

        # classify [optional]
        self.head = nn.Linear(1280, num_classes)

    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x3_wavelet = self.wavelet_conv(x3)
        x4 = self.layer4(x3_wavelet)

        x5 = self.layer5(x4)
        x5 = F.adaptive_avg_pool2d(x5, 1)
        out = self.head(x5.flatten(1))

        # return [x1, x2, x3, x4, x5]
        return out



@register_model
def mobilenetv4_small(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    model = MobileNetV4('MobileNetV4ConvSmall', **kwargs)
    return model


@register_model
def mobilenetv4_medium(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    model = MobileNetV4('MobileNetV4ConvMedium', **kwargs)
    return model

@register_model
def mobilenetv4_large(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    model = MobileNetV4('MobileNetV4ConvLarge', **kwargs)
    return model

@register_model
def mobilenetv4_hybrid_medium(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    model = MobileNetV4('MobileNetV4HybridMedium', **kwargs)
    return model


@register_model
def mobilenetv4_hybrid_large(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    model = MobileNetV4('MobileNetV4HybridLarge', **kwargs)
    return model


if __name__ == '__main__':
    device = torch.device("cuda")

    model = mobilenetv4_small()
    model = model.to(device)

    print(model)

    model = model.to(device)
    print(model)
    model.eval()

    x = torch.randn((32, 3, 244, 244), device=device)
    y1 = model(x)

    flops, params = profile(model, inputs=(x,))

    print(y1.size())
    print('FLOPs = %.2f G ' % ((flops / 1000 ** 3)))
    print('Params = %.2f M' % (params / 1000 ** 2))
