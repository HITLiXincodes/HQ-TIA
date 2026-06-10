import math
from inspect import isfunction
import torch
from torch import nn
from model.ddpm_modules.Attn_block import AttentionBlock
from torchvision.models import vgg16, VGG16_Weights

def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inv_freq = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000) / dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, input):
        shape = input.shape
        sinusoid_in = torch.ger(input.view(-1).float(), self.inv_freq)
        pos_emb = torch.cat([sinusoid_in.sin(), sinusoid_in.cos()], dim=-1)
        pos_emb = pos_emb.view(*shape, self.dim)
        return pos_emb


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim_out, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))

class Downsample(nn.Module):
    def __init__(self, dim, dim_out=None):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = nn.Conv2d(dim, dim_out, 3, padding=1)

    def forward(self, x):
        return self.conv(self.pool(x))


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1),
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, dropout=0, norm_groups=32):
        super().__init__()
        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h + self.res_conv(x)

class ResnetBloc_eca(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        *,
        time_emb_dim=None,
        template_emb_dim=None,
        norm_groups=32,
        dropout=0,
        with_attn=True,
    ):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim,
            dim_out,
            norm_groups=norm_groups,
            dropout=dropout,
        )
        if with_attn:
            self.attn = AttentionBlock(
                dim=dim_out,
                gate_channels=dim_out,
                num_heads=2,
                ffn_expansion_factor=2.66,
                bias=False,
                LayerNorm_type="WithBias",
                time_cond_dim=time_emb_dim,
                template_cond_dim=template_emb_dim,
            )

    def forward(self, x, time_emb=None, template_emb=None):
        x = self.res_block(x)
        if self.with_attn:
            x = self.attn(x, time_cond=time_emb, template_cond=template_emb)
        return x

class ResnetStage(nn.Module):
    def __init__(self, dim, time_emb_dim, template_emb_dim, norm_groups, with_attn, num_blocks):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                ResnetBloc_eca(
                    dim=dim,
                    dim_out=dim,
                    time_emb_dim=time_emb_dim,
                    template_emb_dim=template_emb_dim,
                    norm_groups=norm_groups,
                    with_attn=with_attn,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x, time_emb=None, template_emb=None):
        for block in self.blocks:
            x = block(x, time_emb=time_emb, template_emb=template_emb)
        return x

class Encoder(nn.Module):
    def __init__(
        self,
        in_channel=6,
        inner_channel=32,
        norm_groups=32,
    ):
        super().__init__()

        time_dim = inner_channel
        ch = [32,64,128,256]

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, ch[0], kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.conv2 = Downsample(ch[0], ch[1])
        self.conv3 = Downsample(ch[1], ch[2])
        self.conv4 = Downsample(ch[2], ch[3])

        self.block1 = ResnetStage(
            dim=ch[0],
            time_emb_dim=time_dim,
            template_emb_dim=time_dim,
            norm_groups=norm_groups,
            with_attn=True,
            num_blocks=1,
        )
        self.block2 = ResnetStage(
            dim=ch[1],
            time_emb_dim=time_dim,
            template_emb_dim=time_dim,
            norm_groups=norm_groups,
            with_attn=True,
            num_blocks=2,
        )
        self.block3 = ResnetStage(
            dim=ch[2],
            time_emb_dim=time_dim,
            template_emb_dim=time_dim,
            norm_groups=norm_groups,
            with_attn=True,
            num_blocks=2,
        )
        self.block4 = ResnetStage(
            dim=ch[3],
            time_emb_dim=time_dim,
            template_emb_dim=time_dim,
            norm_groups=norm_groups,
            with_attn=True,
            num_blocks=4,
        )
        self.middle_down = nn.Conv2d(ch[3],64,kernel_size=1,stride=1)
        self.middle = ResnetStage(
            dim=64,
            time_emb_dim=time_dim,
            template_emb_dim=time_dim,
            norm_groups=norm_groups,
            with_attn=True,
            num_blocks=4,
        )
        self.middle_up = nn.Conv2d(64,ch[3],kernel_size=1,stride=1)

        self.conv_up3 = Upsample(ch[3], ch[2])
        self.conv_up2 = Upsample(ch[2], ch[1])
        self.conv_up1 = Upsample(ch[1], ch[0])

        self.conv_cat3 = nn.Conv2d(ch[2]+ch[2],ch[2],kernel_size=1,stride=1)
        self.conv_cat2 = nn.Conv2d(ch[1]+ch[1],ch[1],kernel_size=1,stride=1)
        self.conv_cat1 = nn.Conv2d(ch[0]+ch[0],ch[0],kernel_size=1,stride=1)

        self.decoder_block3 = ResnetStage(
            dim=ch[2],
            time_emb_dim=time_dim,
            template_emb_dim=time_dim,
            norm_groups=norm_groups,
            with_attn=True,
            num_blocks=2,
        )
        self.decoder_block2 = ResnetStage(
            dim=ch[1],
            time_emb_dim=time_dim,
            template_emb_dim=time_dim,
            norm_groups=norm_groups,
            with_attn=True,
            num_blocks=2,
        )
        self.decoder_block1 = ResnetStage(
            dim=ch[0],
            time_emb_dim=time_dim,
            template_emb_dim=time_dim,
            norm_groups=norm_groups,
            with_attn=True,
            num_blocks=1,
        )

    def forward(self, x, time_emb=None, template_emb=None):
        x1 = self.conv1(x)
        x1 = self.block1(x1, time_emb=time_emb, template_emb=template_emb)
        
        x2 = self.conv2(x1)
        x2 = self.block2(x2, time_emb=time_emb, template_emb=template_emb)

        x3 = self.conv3(x2)
        x3 = self.block3(x3, time_emb=time_emb, template_emb=template_emb)

        x4 = self.conv4(x3)
        x4 = self.block4(x4, time_emb=time_emb, template_emb=template_emb)
        
        x_middle = self.middle_down(x4)
        x_middle = self.middle(x_middle, time_emb=time_emb, template_emb=template_emb)
        x_middle = self.middle_up(x_middle)+x4
        
        de_level3 = self.conv_up3(x_middle)
        de_level3 = torch.cat([de_level3, x3], 1)
        de_level3 = self.conv_cat3(de_level3)
        de_level3 = self.decoder_block3(de_level3, time_emb=time_emb, template_emb=template_emb)
        
        de_level2 = self.conv_up2(de_level3)
        de_level2 = torch.cat([de_level2, x2], 1)
        de_level2 = self.conv_cat2(de_level2)
        de_level2 = self.decoder_block2(de_level2, time_emb=time_emb, template_emb=template_emb)
        
        de_level1 = self.conv_up1(de_level2)
        de_level1 = torch.cat([de_level1, x1], 1)
        de_level1 = self.conv_cat1(de_level1)
        de_level1 = self.decoder_block1(de_level1, time_emb=time_emb, template_emb=template_emb)

        return de_level1


class UNet(nn.Module):
    def __init__(
        self,
        in_channel=6,
        out_channel=3,
        inner_channel=512,
        template_dim=512,
        norm_groups=32,
        with_time_emb=True,
    ):
        super().__init__()

        if with_time_emb:
            self.time_mlp = nn.Sequential(
                TimeEmbedding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 2),
                Swish(),
                nn.Linear(inner_channel * 2, inner_channel),
            )
        else:
            self.time_mlp = None
        self.template_mlp = nn.Sequential(
            nn.Linear(template_dim, template_dim * 2),
            Swish(),
            nn.Linear(template_dim * 2, inner_channel),
        )

        self.encoder_VisualCR = Encoder(
            in_channel=in_channel,
            inner_channel=inner_channel,
            norm_groups=norm_groups,
        )
        self.out = nn.Conv2d(32, out_channel, kernel_size=1, stride=1)

    def forward(self, x, time, template):
        if template is None:
            raise ValueError("template must be provided for template-conditioned UNet.")

        if self.time_mlp is not None:
            t_emb = self.time_mlp(time)
            cond_device = t_emb.device
            cond_dtype = t_emb.dtype
        else:
            t_emb = None
            cond_device = x.device
            cond_dtype = x.dtype

        if template.dim() > 2:
            template = template.view(template.shape[0], -1)
        template = template.to(device=cond_device, dtype=cond_dtype)
        template_emb = self.template_mlp(template)

        out = self.encoder_VisualCR(x, time_emb=t_emb, template_emb=template_emb)
        return self.out(out)

class Discriminator(nn.Module):
    def __init__(self, use_enhance=False):
        super().__init__()

        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.eval()

        self.blocks1 = vgg[:4]
        self.blocks2 = vgg[4:9]
        self.blocks3 = vgg[9:16]
        self.blocks4 = vgg[16:23]
        self.blocks5 = vgg[23:30]

        for p in self.parameters():
            p.requires_grad = False

        self.use_enhance = use_enhance

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x):
        x = (x - self.mean) / self.std

        out = []

        x = self.blocks1(x)
        out.append(x + x * torch.sigmoid(x) if self.use_enhance else x)

        x = self.blocks2(x)
        out.append(x + x * torch.sigmoid(x) if self.use_enhance else x)

        x = self.blocks3(x)
        out.append(x + x * torch.sigmoid(x) if self.use_enhance else x)

        x = self.blocks4(x)
        out.append(x + x * torch.sigmoid(x) if self.use_enhance else x)

        x = self.blocks5(x)
        out.append(x + x * torch.sigmoid(x) if self.use_enhance else x)

        return out
