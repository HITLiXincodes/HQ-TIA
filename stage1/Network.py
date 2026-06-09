
from torch.nn import functional as F
#from mamba_ssm import Mamba
import torch
import torch.nn as nn
import numbers
from einops import rearrange
from torchinfo import summary


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class De_residual(nn.Module):
    def __init__(self, channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, out_channels, kernel_size=1,  stride=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

        self.skip = nn.Sequential(
            nn.Conv2d(channels,out_channels,kernel_size=1,stride=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
        
    def forward(self, data):
        conv1 = self.conv1(data)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        skip = self.skip(data)
        return conv3+skip

#net=De_residual(512, 512).cuda()
#summary(net,(2,512,2,2))

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
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x 

#net=TransformerBlock(64,8,2.66,False,'WithBias').cuda()
#summary(net,(2,64,16,16))

class Spatial_Block(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, depth):
        super().__init__()
        self.conv = De_residual(in_channels, out_channels)
        self.trans = nn.ModuleList([TransformerBlock(in_channels, num_heads) for i in range(depth)])
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
    def forward(self, x):
        skip = self.skip(x)
        for layer in self.trans:
            x=x+layer(x)
        out = self.conv(x)
        return out+skip

#net = Spatial_Block(512,4,4).cuda()
#summary(net, (2, 512, 1, 1))

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        # stage1 512 x 2 x 2 
        self.spatial1 = Spatial_Block(512, 512, 4, 4)#in_channels, out_channels, num_heads, depth
        self.sscale1 = 'interpolate'
        
        # stage2 256 x 4 x 4
        self.spatial2 = Spatial_Block(512, 256, 4, 4)
        self.sscale2 = 'interpolate'

        # stage3 128 x 8 x 8
        self.spatial3 = Spatial_Block(256, 128, 8, 4)
        self.sscale3 = 'interpolate'

        # stage4 64 x 16 x 16
        self.spatial4 = Spatial_Block(128, 64, 8, 4)
        self.sscale4 = 'interpolate'

        # stage5 32 x 30 x 30
        self.spatial5 = Spatial_Block(64, 32, 16, 2)
        self.sscale5 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 2, bias=False),
            nn.BatchNorm2d(64)
        )
        
        # stage6 16 x 58 x 58
        self.spatial6 = Spatial_Block(32, 16, 16, 2)
        self.sscale6 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 4, 2, 2, bias=False),
            nn.BatchNorm2d(32)
        )

        # stage2 3 x 112 x 112
        self.spatial7 = Spatial_Block(16, 3, 16, 2)
        self.sscale7 = nn.Sequential(
            nn.ConvTranspose2d(16, 16, 4, 2, 3, bias=False),
            nn.BatchNorm2d(16)
        )

    def forward(self, deep_feature):
        x = F.interpolate(deep_feature, scale_factor=2, mode='nearest')
        x = self.spatial1(x)

        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.spatial2(x)

        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.spatial3(x)

        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.spatial4(x)

        x = self.sscale5(x)
        x = self.spatial5(x)

        x = self.sscale6(x)
        x = self.spatial6(x)

        x = self.sscale7(x)
        x = self.spatial7(x)

        #x = nn.Sigmoid()(x)
        return x

#net = Generator().cuda("cuda:0")
#summary(net, (2, 512, 1, 1))
'''
class Discriminator(nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
        self.level1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=1, stride=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2)
        )
        self.level2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2)
        )
        self.level3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2)
        )
        self.level4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2)
        )
        self.level5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=7),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(256, 256),
            #nn.Sigmoid()
        )

    def forward(self, x):
        level1 = self.level1(x)
        level2 = self.level2(level1)
        level3 = self.level3(level2)
        level4 = self.level4(level3)
        level5 = self.level5(level4)
        level5 = nn.functional.normalize(level5, p=2, dim=1)
        return level5
'''
#net = Discriminator(3)
#summary(net,(2,3,112,112))
import torchvision
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = torchvision.models.vgg16(pretrained=True).features.eval()
        self.blocks1 = vgg[:4]
        self.blocks2 = vgg[4:9]
        self.blocks3 = vgg[9:16]
        self.blocks4 = vgg[16:23]
        self.blocks5 = vgg[23:30]
        for p in self.blocks1:
            p.requires_grad = False
        for p in self.blocks2:
            p.requires_grad = False
        for p in self.blocks3:
            p.requires_grad = False
        for p in self.blocks4:
            p.requires_grad = False
        for p in self.blocks5:
            p.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    def forward(self, input):
        out=[]
        input = (input-self.mean) / self.std
        
        x = self.blocks1(input)
        temp = x+x*torch.sigmoid(x)
        out.append(temp)
        
        #print(x.shape)
        x = self.blocks2(x)
        temp = x+x*torch.sigmoid(x)
        out.append(temp)

        x = self.blocks3(x)
        temp = x+x*torch.sigmoid(x)
        out.append(temp)

        x = self.blocks4(x)
        temp = x+x*torch.sigmoid(x)
        out.append(temp)

        x = self.blocks5(x)
        temp = x+x*torch.sigmoid(x)
        out.append(temp)

        return out

#net = Discriminator()
#summary(net,(2,3,112,112))