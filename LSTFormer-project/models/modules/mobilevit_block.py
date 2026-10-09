
from torch import nn
import math
import torch
from torch.nn import functional as F
from torch.nn import init as init
from timm.models.layers import trunc_normal_
from einops import rearrange
import numbers
# from .transformer import TransformerEncoder
from .architectures import ConvLayer, CBAMBlock, Res_CBAM, ChannelAttention, SpatialAttention
# from .LSTformer_arch import TransformerBlock

class LFEM(nn.Module):
    """
        LFEM: Low Frequency Enhancement Module
        in_channels: :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H, W)
        transformer_dim: Input dimension to the transformer unit. Default: 144
        ffn_dim: Dimension of the FFN block. Default: 288
        n_transformer_blocks: Number of transformer blocks. Default: 3
        head_dim: Head dimension in the multi-head attention. Default: 18
        attn_dropout: Dropout in multi-head attention. Default: 0.0
        dropout: Dropout rate. Default: 0.0
        ffn_dropout: Dropout between FFN layers in transformer. Default: 0.0
        atch_h: Patch height for unfolding operation. Default: 2
        patch_w: Patch width for unfolding operation. Default: 2
        transformer_norm_layer: Normalization layer in the transformer block. Default: layer_norm
        conv_ksize: The kernel size of convolution. Default: 3
        The partial network settings were inspired by the paper:
        MobileViT: Light-weight, general-purpose, and mobile-friendly vision transformer
            https://arxiv.org/abs/2110.02178

    """
    def __init__(self, in_channels=96, transformer_dim=144, ffn_dim=288,num_topk=50,
                 n_transformer_blocks=3, head_dim=18, attn_dropout=0.0, dropout=0.0,
                 ffn_dropout=0.0, patch_h=2, patch_w=2,transformer_norm_layer="layer_norm", conv_ksize=3):
        super(LFEM, self).__init__()
        #
        self.local_rep0 = ConvLayer(
            in_channels=in_channels,
            out_channels=transformer_dim,
            kernel_size=conv_ksize,
            stride=1,
            use_norm=False,
            use_act=True
        )

        global_rep = [
            TransformerBlock(dim=transformer_dim,
                             num_heads=8,
                             ffn_expansion_factor=2.66,
                             bias=False,
                             LayerNorm_type='WithBias')
            for i in range(n_transformer_blocks)]

        self.global_rep = nn.Sequential(*global_rep)

        self.conv_1x1_out = ConvLayer(
            in_channels=transformer_dim*2,
            out_channels=transformer_dim,
            kernel_size=1,
            stride=1,
            use_norm=False,
            use_act=True
        )

        self.conv_3x3_out = ConvLayer(
            in_channels=transformer_dim,
            out_channels=in_channels,
            kernel_size=conv_ksize,
            use_norm=False,
            use_act=True
        )

        self.n_transformer_blocks = n_transformer_blocks
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = self.patch_w * self.patch_h
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):

        res1 = x.clone()
        fm = self.local_rep0(x)
        # fm = self.local_rep1(fm)
        # fm = self.local(fm)
        res2 = fm
        # # LSTFormer
        # fm = self.global_rep(x)
        fm = self.global_rep(fm)
        fm = self.conv_1x1_out(torch.cat((res2, fm), dim=1))
        fm = self.conv_3x3_out(fm)
        fm = res1 + fm
        return fm

##########################################################################
# Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.act = nn.GELU()
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = self.act(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Trans_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Trans_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature  # @矩阵乘法运算符
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

##使用**可学习的软阈值（Learnable Soft-Thresholding）**替代原来的硬性 Top-K 机制
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads

        # 温度系数
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # [新增] 可学习的阈值参数
        # 初始化为一个较小的负数，经过 Sigmoid 后变成一个很小的正数
        # 这样在训练初期不会过滤掉太多信息，防止梯度消失
        self.threshold = nn.Parameter(torch.full((1, num_heads, 1, 1), -5.0))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # [移除] 移除了原有的 attn1, attn2, attn3, attn4 权重参数
        # 因为软阈值机制不需要多尺度硬截断来集成

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        # 计算原始注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.temperature

        # 1. 正常的 Softmax 归一化
        attn = attn.softmax(dim=-1)

        # [核心修改] 2. 应用可学习的软阈值
        # 这里的逻辑是：概率低于 threshold 的被视为噪声，置为0；
        # 高于 threshold 的保留（并减去阈值，保持连续性）
        soft_threshold = torch.sigmoid(self.threshold)  # 限制在 (0, 1) 之间
        attn = torch.relu(attn - soft_threshold)

        # 3. 重新归一化 (Renormalization)
        # 这一步非常重要，因为 ReLU 之后概率和不为1了
        # 添加 1e-8 防止除以零
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)

        # 输出
        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)

        return out




# 没有TOP-K的cross Attention
class Cross_Attention_1(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Cross_Attention_1, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        dwkernel_size = 3  #默认为3
        paddings = dwkernel_size // 2
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=dwkernel_size, stride=1, padding=paddings, groups=dim * 2, bias=bias)

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=dwkernel_size , stride=1, padding=paddings, groups=dim, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape

        kv = self.kv_dwconv(self.kv(x))
        k, v = kv.chunk(2, dim=1)
        q = self.q_dwconv(self.q(y))

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

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

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
        return x / torch.sqrt(sigma+1e-5) * self.weight

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

##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, init_values=1e-5, use_layer_scale=True):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):

        # pre-LN
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

# class Cross_Attention(nn.Module):
#     def __init__(self, dim, num_heads, bias):
#         super(Cross_Attention, self).__init__()
#         self.num_heads = num_heads
#         self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
#
#         self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
#         self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
#
#         self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
#         self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
#
#         self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
#
#     def forward(self, x, y):
#         b, c, h, w = x.shape
#
#         kv = self.kv_dwconv(self.kv(x))
#         k, v = kv.chunk(2, dim=1)
#         q = self.q_dwconv(self.q(y))
#
#         q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#
#         q = torch.nn.functional.normalize(q, dim=-1)
#         k = torch.nn.functional.normalize(k, dim=-1)
#
#         attn = (q @ k.transpose(-2, -1)) * self.temperature
#         attn = attn.softmax(dim=-1)
#
#         out = (attn @ v)
#
#         out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
#
#         out = self.project_out(out)
#         return out

##使用**可学习的软阈值（Learnable Soft-Thresholding）**替代原来的硬性 Top-K 机制

class Cross_Attention(nn.Module):
    # [关键修复] 必须在这里接收 attn_mode
    def __init__(self, dim, num_heads, bias, attn_mode='lst'):
        super(Cross_Attention, self).__init__()
        self.num_heads = num_heads
        self.attn_mode = attn_mode
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # [消融] 仅在 LST 模式下初始化阈值
        if self.attn_mode == 'lst':
            self.threshold = nn.Parameter(torch.full((1, num_heads, 1, 1), -5.0))

        dwkernel_size = 3
        paddings = dwkernel_size // 2
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=dwkernel_size, stride=1, padding=paddings,
                                   groups=dim * 2, bias=bias)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=dwkernel_size, stride=1, padding=paddings, groups=dim,
                                  bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
    # def __init__(self, dim, num_heads, bias):
    #     super(Cross_Attention, self).__init__()
    #     self.num_heads = num_heads
    #     self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
    #
    #     # [新增] 可学习的阈值参数
    #     self.threshold = nn.Parameter(torch.full((1, num_heads, 1, 1), -5.0))
    #
    #     dwkernel_size = 3
    #     paddings = dwkernel_size // 2
    #
    #     self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
    #     self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=dwkernel_size, stride=1, padding=paddings,
    #                                groups=dim * 2, bias=bias)
    #
    #     self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
    #     self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=dwkernel_size, stride=1, padding=paddings, groups=dim,
    #                               bias=bias)
    #
    #     self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # [移除] 移除了原有的 attn1, attn2, attn3, attn4 权重参数

    def forward(self, x, y):
        b, c, h, w = x.shape
        kv = self.kv_dwconv(self.kv(x))
        k, v = kv.chunk(2, dim=1)
        q = self.q_dwconv(self.q(y))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        # [消融逻辑]
        if self.attn_mode == 'lst':
            soft_threshold = torch.sigmoid(self.threshold)
            attn = torch.relu(attn - soft_threshold)
        elif self.attn_mode == 'topk':
            k_val = int(attn.shape[-1] * 0.5)
            if k_val > 0:
                topk_values, _ = torch.topk(attn, k_val, dim=-1)
                min_values = topk_values[..., -1:]
                mask = (attn >= min_values).float()
                attn = attn * mask
        elif self.attn_mode == 'softmax':
            pass

            # Renormalize
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class Cross_TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(Cross_TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Cross_Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, y):
        x1 = self.norm1(x)
        y = self.norm1(y)
        x = x + self.attn(x1, y)
        x = x + self.ffn(self.norm2(x))

        return x

class Pixel_Attention(nn.Module):
    def __init__(self, kernel_size, stride):
        super(Pixel_Attention, self).__init__()
        self.max_pooling = nn.MaxPool2d(
            kernel_size=kernel_size,
            stride=stride
        )
        self.ave_pooling = nn.AvgPool2d(
            kernel_size=kernel_size,
            stride=stride
        )
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=1)
        self.nonlinear = nn.Sigmoid()

    def forward(self, x):

        # (b, w, h, c)
        out = x.transpose(1, 3)
        max_out = self.max_pooling(out)
        ave_out = self.ave_pooling(out)
        max_out = max_out.transpose(1, 3)
        ave_out = ave_out.transpose(1, 3)
        out = torch.cat([max_out, ave_out], dim=1)
        out = self.conv(out)
        out = self.nonlinear(out)
        out = x * out

        return out




def get_dwconv(dim, kernel, bias):
    return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel - 1) // 2, bias=bias, groups=dim)


class GlobalLocalFilter(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.dw = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, bias=False, groups=dim // 2)
        self.complex_weight = nn.Parameter(torch.randn(dim // 2, h, w, 2, dtype=torch.float32) * 0.02)
        trunc_normal_(self.complex_weight, std=.02)
        self.pre_norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.post_norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')

    def forward(self, x):
        x = self.pre_norm(x)
        x1, x2 = torch.chunk(x, 2, dim=1)
        x1 = self.dw(x1)

        x2 = x2.to(torch.float32)
        B, C, a, b = x2.shape
        x2 = torch.fft.rfft2(x2, dim=(2, 3), norm='ortho')

        weight = self.complex_weight
        if not weight.shape[1:3] == x2.shape[2:4]:
            weight = F.interpolate(weight.permute(3, 0, 1, 2), size=x2.shape[2:4], mode='bilinear', align_corners=True).permute(1, 2, 3, 0)

        weight = torch.view_as_complex(weight.contiguous())

        x2 = x2 * weight
        x2 = torch.fft.irfft2(x2, s=(a, b), dim=(2, 3), norm='ortho')

        x = torch.cat([x1.unsqueeze(2), x2.unsqueeze(2)], dim=2).reshape(B, 2 * C, a, b)
        x = self.post_norm(x)
        return x


class gnconv(nn.Module):
    def __init__(self, dim, order=5, gflayer=None, h=14, w=8, s=1.0):
        super().__init__()
        self.order = order
        self.dims = [dim // 2 ** i for i in range(order)]
        self.dims.reverse()
        self.proj_in = nn.Conv2d(dim, 2 * dim, 1)

        if gflayer is None:
            self.dwconv = get_dwconv(sum(self.dims), 7, True)
        else:
            self.dwconv = gflayer(sum(self.dims), h=h, w=w)

        self.proj_out = nn.Conv2d(dim, dim, 1)

        self.pws = nn.ModuleList(
            [nn.Conv2d(self.dims[i], self.dims[i + 1], 1) for i in range(order - 1)]
        )

        self.scale = s
        self.softmax = nn.Softmax(dim=-1)
        print('[gnconv]', order, 'order with dims=', self.dims, 'scale=%.4f' % self.scale)

    def forward(self, x, mask=None, dummy=False):
        B, C, H, W = x.shape

        fused_x = self.proj_in(x)
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)

        dw_abc = self.dwconv(abc) * self.scale

        dw_list = torch.split(dw_abc, self.dims, dim=1)
        x = pwa * dw_list[0]

        for i in range(self.order - 1):
            if i < self.order - 2:
                x = self.pws[i](x) * dw_list[i + 1]
                x = self.softmax(x)
            else:
                x = self.pws[i](x) * dw_list[i + 1]

        x = self.proj_out(x)

        return x


class Block(nn.Module):
    r""" HorNet block
    """

    def __init__(self, dim, layer_scale_init_value=1e-6, gnconv=gnconv):
        super().__init__()

        self.norm1 = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.gnconv = gnconv(dim)  # depthwise conv
        self.norm2 = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                   requires_grad=True) if layer_scale_init_value > 0 else None

        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                   requires_grad=True) if layer_scale_init_value > 0 else None


    def forward(self, x):
        B, C, H, W = x.shape
        if self.gamma1 is not None:
            gamma1 = self.gamma1.view(C, 1, 1)
        else:
            gamma1 = 1
        x = x + gamma1 * self.gnconv(self.norm1(x))

        input = x
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm2(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma2 is not None:
            x = self.gamma2 * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + x
        return x


class Dual_TransformerBlock(nn.Module):
    # [修改] 这里增加了 attn_mode='lst' 参数
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, init_values=1e-5,
                 use_layer_scale=True, attn_mode='lst'):
        super(Dual_TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.attn1 = Attention(dim, num_heads, bias)  # Self-Attention 保持默认

        self.norm3 = LayerNorm(dim, LayerNorm_type)

        # [修改] 将 attn_mode 传入 Cross_Attention
        self.attn2 = Cross_Attention(dim, num_heads, bias, attn_mode=attn_mode)

        self.norm4 = LayerNorm(dim, LayerNorm_type)
        self.ffn1 = FeedForward(dim, ffn_expansion_factor, bias)

        # [修改] 将 attn_mode 传入 Cross_Attention
        self.attn3 = Cross_Attention(dim, num_heads, bias, attn_mode=attn_mode)

        self.norm5 = LayerNorm(dim, LayerNorm_type)
        self.ffn2 = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x1 = self.norm1(x[0])  # Encoder feature

        x2 = self.attn1(self.norm2(x[1])) + x[1]  # Self-Attention on Decoder
        x2 = self.attn2(x1, self.norm3(x2)) + x2  # Cross-Attention 1
        x2 = self.ffn1(self.norm4(x2)) + x2

        x1 = self.attn3(x2, x1) + x[0]  # Cross-Attention 2 (bi-directional info flow)
        x1 = self.ffn2(self.norm5(x1)) + x1

        return x1


# class Dual_TransformerBlock(nn.Module):
#     # 1. 在 __init__ 中增加 attn_mode 参数
#     def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, attn_mode='lst'):
#         super(Dual_TransformerBlock, self).__init__()
#         self.norm1 = LayerNorm(dim, LayerNorm_type)
#         self.norm2 = LayerNorm(dim, LayerNorm_type)
#         # Self Attention 也可以加 attn_mode，但在你的消融设计中主要针对 Cross Attention，这里暂不改 Self
#         self.attn1 = Attention(dim, num_heads, bias)
#
#         self.norm3 = LayerNorm(dim, LayerNorm_type)
#         # 2. 将 attn_mode 传入 Cross_Attention
#         self.attn2 = Cross_Attention(dim, num_heads, bias, attn_mode=attn_mode)
#         self.norm4 = LayerNorm(dim, LayerNorm_type)
#         self.ffn1 = FeedForward(dim, ffn_expansion_factor, bias)
#
#         # 3. 将 attn_mode 传入 Cross_Attention
#         self.attn3 = Cross_Attention(dim, num_heads, bias, attn_mode=attn_mode)
#
#         self.norm5 = LayerNorm(dim, LayerNorm_type)
#         self.ffn2 = FeedForward(dim, ffn_expansion_factor, bias)

    # def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
    #     super(Dual_TransformerBlock, self).__init__()
    #     self.norm1 = LayerNorm(dim, LayerNorm_type)
    #     self.norm2 = LayerNorm(dim, LayerNorm_type)
    #     self.attn1 = Attention(dim, num_heads, bias)
    #
    #     self.norm3 = LayerNorm(dim, LayerNorm_type)
    #     self.attn2 = Cross_Attention(dim, num_heads, bias)
    #     self.norm4 = LayerNorm(dim, LayerNorm_type)
    #     self.ffn1 = FeedForward(dim, ffn_expansion_factor, bias)
    #
    #     self.attn3 = Cross_Attention(dim, num_heads, bias)
    #
    #     self.norm5 = LayerNorm(dim, LayerNorm_type)
    #     self.ffn2 = FeedForward(dim, ffn_expansion_factor, bias)


    # def forward(self, x):
    #     x1 = self.norm1(x[0])
    #
    #     x2 = self.attn1(self.norm2(x[1])) + x[1]
    #     x2 = self.attn2(x1, self.norm3(x2)) + x2
    #     x2 = self.ffn1(self.norm4(x2)) + x2
    #
    #     x1 = self.attn3(x2, x1) + x[0]
    #     x1 = self.ffn2(self.norm5(x1)) + x1
    #
    #
    #     return x1

