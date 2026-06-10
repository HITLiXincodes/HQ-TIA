import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers

from einops import rearrange


def modulate_2d(x, shift, scale):
    return x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

########################################################################
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

# Multi-Head C and S Attention Block
class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels


        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
        
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
        
        channel_att_sum /= 2
        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

class MultiHeadChannelAttention(nn.Module):
    def __init__(self, gate_channels, num_heads, reduction_ratio=16, pool_types=['avg', 'max']):
        super(MultiHeadChannelAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = gate_channels // num_heads
        self.gate_channels = gate_channels

        self.heads = nn.ModuleList(
            [ChannelGate(self.head_dim, reduction_ratio, pool_types) for _ in range(num_heads)]
        )

        self.fc = nn.Conv2d(gate_channels, gate_channels, 1, bias=False)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        assert channels == self.gate_channels, "Input channels must match gate_channels"

        x = x.view(batch_size, self.num_heads, self.head_dim, height, width)

        head_outputs = [head(x[:, i, :, :, :]) for i, head in enumerate(self.heads)]
        out = torch.cat(head_outputs, dim=1)

        out = self.fc(out)
        return out

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 3
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, num_heads, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = MultiHeadChannelAttention(gate_channels, num_heads, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

##########################################################################
class AttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        gate_channels,
        num_heads,
        ffn_expansion_factor,
        bias,
        LayerNorm_type,
        time_cond_dim,
        template_cond_dim,
    ):
        super(AttentionBlock, self).__init__()

        self.time_norm1 = LayerNorm(dim, LayerNorm_type)
        self.time_attn = CBAM(gate_channels, num_heads, reduction_ratio=16)
        self.time_norm2 = LayerNorm(dim, LayerNorm_type)
        self.time_ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.time_modulation = (
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_cond_dim, 6 * dim, bias=True)
            )
        )

        self.template_norm1 = LayerNorm(dim, LayerNorm_type)
        self.template_attn = CBAM(gate_channels, num_heads, reduction_ratio=16)
        self.template_norm2 = LayerNorm(dim, LayerNorm_type)
        self.template_ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.template_modulation = (
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(template_cond_dim, 6 * dim, bias=True)
            )
        )

    def _modulation_or_zeros(self, modulation_module, cond):
        return modulation_module(cond).chunk(6, dim=1)

    def forward(self, x, time_cond=None, template_cond=None):
        time_shift_attn, time_scale_attn, time_gate_attn, time_shift_ffn, time_scale_ffn, time_gate_ffn = self._modulation_or_zeros(
            self.time_modulation,
            time_cond
        )
        template_shift_attn, template_scale_attn, template_gate_attn, template_shift_ffn, template_scale_ffn, template_gate_ffn = self._modulation_or_zeros(
            self.template_modulation,
            template_cond
        )

        x = x + time_gate_attn[:, :, None, None] * self.time_attn(
            modulate_2d(self.time_norm1(x), time_shift_attn, time_scale_attn)
        )
        x = x + time_gate_ffn[:, :, None, None] * self.time_ffn(
            modulate_2d(self.time_norm2(x), time_shift_ffn, time_scale_ffn)
        )
        x = x + template_gate_attn[:, :, None, None] * self.template_attn(
            modulate_2d(self.template_norm1(x), template_shift_attn, template_scale_attn)
        )
        x = x + template_gate_ffn[:, :, None, None] * self.template_ffn(
            modulate_2d(self.template_norm2(x), template_shift_ffn, template_scale_ffn)
        )

        return x
