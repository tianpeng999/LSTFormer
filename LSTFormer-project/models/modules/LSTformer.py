import torch
import torch.nn as nn

import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
import math

from einops import rearrange
from .architectures import ChannelAttention, InvertedResidualBlock, SFT, \
    Res_block, ConvLayer, DWT_2D, IDWT_2D, RDB
from .mobilevit_block import Cross_Attention, Attention, FeedForward, TransformerBlock, Dual_TransformerBlock


class Image_Downconv(nn.Module):
    def __init__(self, in_channels):
        super(Image_Downconv, self).__init__()
        self.conv = nn.Sequential(
            ConvLayer(in_channels=3, out_channels=in_channels // 4, kernel_size=3, stride=1, use_act=False),
            ConvLayer(in_channels=in_channels // 4, out_channels=in_channels // 2, kernel_size=3, stride=1, use_act=False),
            ConvLayer(in_channels=in_channels // 2, out_channels=in_channels, kernel_size=3, stride=1, use_act=False)
        )

    def forward(self, x):

        return self.conv(x)


class Image_UPconv(nn.Module):
    def __init__(self, in_channels):
        super(Image_UPconv, self).__init__()

        self.conv = nn.Sequential(
            ConvLayer(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=3, stride=1, use_act=False),
            ConvLayer(in_channels=in_channels // 2, out_channels=32, kernel_size=3, stride=1, use_act=False)
        )

    def forward(self, x):

        return self.conv(x)

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

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
class Self_TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, init_values=1e-5, use_layer_scale=True):
        super(Self_TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn1 = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)

        self.attn2 = Cross_Attention(dim, num_heads, bias)
        # self.attn2 = Attention(dim, num_heads, bias)
        self.norm3 = LayerNorm(dim, LayerNorm_type)
        self.ffn1 = FeedForward(dim, ffn_expansion_factor, bias)
        self.norm4 = LayerNorm(dim, LayerNorm_type)

    def forward(self, x):

        x2 = x + self.attn1(self.norm1(x))
        # x2 = x2 + self.attn2(x, self.norm2(x2))  # 以浅一层的特征x作KV, 更深层次的特征x2作Q默认
        # x2 = x + self.attn2(x2, self.norm2(x))  # 以浅一层的特征作Q, 更深层次的特征作KV
        x2 = x2 + self.attn2(self.norm4(x), self.norm2(x2))  # 以浅一层的特征x作KV, 更深层次的特征x2作Q
        # x2 = x + self.attn2(self.norm2(x2), self.norm4(x))  # 以深层次的特征x2作KV, 以浅一层的特征x作Q
        # x2 = x2 + self.attn2(self.norm2(x2))  # 取消自Transformer支路
        x2 = x2 + self.ffn1(self.norm3(x2))

        return x2


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class Mix_Block(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, LayerNorm_type, num_heads=8, num_blocks=2,use_layer_scale=False):
        super(Mix_Block, self).__init__()
        #
        self.Global = nn.Sequential(*[
            TransformerBlock(dim=dim,
                             num_heads=num_heads,
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias,
                             LayerNorm_type=LayerNorm_type,
                             use_layer_scale=use_layer_scale)
            for i in range(num_blocks)]
                                    )

    def forward(self, x):
        x = self.Global(x)
        return x




# 1. 替代 DWT 的卷积下采样
# DWT: C -> 4C, H -> H/2. 为了对齐，ConvDown 也设为 C -> 4C
class ConvDown(nn.Module):
    def __init__(self, in_c):
        super(ConvDown, self).__init__()
        self.conv = nn.Conv2d(in_c, in_c * 4, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x):
        return self.conv(x)


# 2. 替代 IDWT 的卷积上采样
# IDWT: 4C -> C, H -> 2H. PixelShuffle(2) 正好实现 Channel/4, H*2
class ConvUp(nn.Module):
    def __init__(self, in_c):
        super(ConvUp, self).__init__()
        self.up = nn.PixelShuffle(upscale_factor=2)

    def forward(self, x):
        return self.up(x)


# 3. 替代 CAFB (Dual_TransformerBlock) 的简单拼接融合
# 仅做 Concat + 降维 + 残差块
class SimpleConcatBlock(nn.Module):
    def __init__(self, dim, **kwargs):  # 接收多余参数(heads等)以兼容接口
        super(SimpleConcatBlock, self).__init__()
        # 假设 encoder 和 decoder 特征维度相同，拼接后为 dim*2
        self.reduce = ConvLayer(dim * 2, dim, kernel_size=1, stride=1, use_act=True)
        self.block = Res_block(in_channels=dim)

    def forward(self, x):
        # x 是一个元组 (encoder_feat, decoder_feat)
        enc_feat, dec_feat = x[0], x[1]
        out = torch.cat([enc_feat, dec_feat], dim=1)
        out = self.reduce(out)
        out = self.block(out) + out  # 简单的残差连接
        return out


# =========================================================================
# [Main Model]
# =========================================================================

class LSTformer(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=32,
                 num_blocks=[2, 2, 2, 2],
                 heads=[8, 8, 8, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 use_layer_scale=False,
                 use_act=False,
                 conv_bias=False,
                 # [新增消融参数]
                 sampling_mode='dwt',  # 'dwt' (Ours) or 'conv' (Baseline)
                 fusion_mode='cafb',# 'cafb' (Ours) or 'concat' (Baseline)
                 attn_mode='lst'
                 ):
        super(LSTformer, self).__init__()

        self.sampling_mode = sampling_mode
        self.fusion_mode = fusion_mode

        self.patch_embed = ConvLayer(in_channels=inp_channels,
                                     out_channels=dim,
                                     kernel_size=3,
                                     stride=1,
                                     use_act=use_act,
                                     bias=conv_bias)

        self.encoder_level1 = nn.Sequential(*[
            Res_block(in_channels=dim)
            for i in range(6)
        ])

        # ============================================================
        # [Ablation: Sampling] 定义下采样和上采样算子
        # ============================================================
        if self.sampling_mode == 'dwt':
            # DWT 是无参数的，可以共享
            self.down_op = DWT_2D(wave='haar')
            self.up_op = IDWT_2D(wave='haar')
        else:
            # Conv 是有参数的，必须为每一层单独定义
            # Level 1 -> 2
            # 输入 reduce_channel2 后的维度: dim // 2
            self.down_layer2 = ConvDown(in_c=dim // 2)

            # Level 2 -> 3
            # 输入 reduce_channel3 后的维度: dim
            self.down_layer3 = ConvDown(in_c=dim)

            # Level 3 -> 4
            # 输入 reduce_channel4 后的维度: dim * 2
            self.down_layer4 = ConvDown(in_c=dim * 2)

            # Decoder 上采样层
            # Level 4 -> 3 (输入 expand_channel4 后的维度)
            self.up_layer3 = ConvUp(in_c=int(dim * 2 ** 4))
            # Level 3 -> 2
            self.up_layer2 = ConvUp(in_c=int(dim * 2 ** 3))
            # Level 2 -> 1
            self.up_layer1 = ConvUp(in_c=int(dim * 2 ** 2))

        # ============================================================
        # Encoder 通道调整层 (保持不变)
        # ============================================================
        self.reduce_channel2 = ConvLayer(in_channels=dim, out_channels=dim // 2, kernel_size=3, use_act=use_act,
                                         bias=conv_bias)
        self.reduce_channel3 = ConvLayer(in_channels=int(dim * 2 ** 1), out_channels=dim, kernel_size=3,
                                         use_act=use_act, bias=conv_bias)
        self.reduce_channel4 = ConvLayer(in_channels=int(dim * 2 ** 2), out_channels=dim * 2, kernel_size=3,
                                         use_act=use_act, bias=conv_bias)

        # ============================================================
        # Transformer Encoders (保持不变)
        # ============================================================
        self.encoder_level2 = Self_TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1],
                                                    ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                                    LayerNorm_type=LayerNorm_type, use_layer_scale=use_layer_scale)
        self.encoder_level3 = Self_TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2],
                                                    ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                                    LayerNorm_type=LayerNorm_type, use_layer_scale=use_layer_scale)
        self.latent = Self_TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3],
                                            ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                            LayerNorm_type=LayerNorm_type, use_layer_scale=use_layer_scale)

        # ============================================================
        # Decoder 通道扩充层 (保持不变)
        # ============================================================
        self.expand_channel4 = ConvLayer(in_channels=int(dim * 2 ** 3), out_channels=int(dim * 2 ** 4), kernel_size=3,
                                         use_act=use_act, bias=conv_bias)
        self.expand_channel3 = ConvLayer(in_channels=int(dim * 2 ** 2), out_channels=int(dim * 2 ** 3), kernel_size=3,
                                         use_act=use_act, bias=conv_bias)
        self.expand_channel2 = ConvLayer(in_channels=int(dim * 2 ** 1), out_channels=int(dim * 2 ** 2), kernel_size=3,
                                         use_act=use_act, bias=conv_bias)

        # ============================================================
        # [Ablation: Fusion] 定义解码器融合模块
        # ============================================================
        if self.fusion_mode == 'cafb':
            FusionBlock = Dual_TransformerBlock
        else:
            FusionBlock = SimpleConcatBlock

        # Decoder Level 3
        self.decoder_level3 = nn.Sequential(*[
            FusionBlock(dim=int(dim * 2 ** 2),
                        num_heads=heads[2],
                        ffn_expansion_factor=ffn_expansion_factor,
                        bias=bias,
                        LayerNorm_type=LayerNorm_type,
                        use_layer_scale=use_layer_scale,
                        attn_mode=attn_mode)
            for i in range(1)]
                                            )

        # Decoder Level 2
        self.decoder_level2 = nn.Sequential(*[
            FusionBlock(dim=int(dim * 2 ** 1),
                        num_heads=heads[1],
                        ffn_expansion_factor=ffn_expansion_factor,
                        bias=bias,
                        LayerNorm_type=LayerNorm_type,
                        use_layer_scale=use_layer_scale,
                        attn_mode=attn_mode)
            for i in range(1)]
                                            )

        # Decoder Level 0 (Output Level)
        self.decoder_level0 = nn.Sequential(*[
            FusionBlock(dim=int(dim * 2 ** 0),
                        num_heads=heads[0],
                        ffn_expansion_factor=ffn_expansion_factor,
                        bias=bias,
                        LayerNorm_type=LayerNorm_type,
                        attn_mode=attn_mode)
            for i in range(1)]
                                            )

        self.output = ConvLayer(in_channels=(dim * 2 ** 0),
                                out_channels=3,
                                kernel_size=3,
                                stride=1,
                                use_act=False,
                                bias=conv_bias)

    def forward(self, inp_img):

        # --- Encoder Level 1 ---
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        # --- Downsample 1 -> 2 ---
        feat_reduced_2 = self.reduce_channel2(out_enc_level1)
        if self.sampling_mode == 'dwt':
            inp_enc_level2 = self.down_op(feat_reduced_2)
        else:
            inp_enc_level2 = self.down_layer2(feat_reduced_2)

        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        # --- Downsample 2 -> 3 ---
        feat_reduced_3 = self.reduce_channel3(out_enc_level2)
        if self.sampling_mode == 'dwt':
            inp_enc_level3 = self.down_op(feat_reduced_3)
        else:
            inp_enc_level3 = self.down_layer3(feat_reduced_3)

        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        # --- Downsample 3 -> 4 ---
        feat_reduced_4 = self.reduce_channel4(out_enc_level3)
        if self.sampling_mode == 'dwt':
            inp_enc_level4 = self.down_op(feat_reduced_4)
        else:
            inp_enc_level4 = self.down_layer4(feat_reduced_4)

        latent = self.latent(inp_enc_level4)

        # ==========================================================
        # Decoder Part
        # ==========================================================

        # --- Upsample Latent -> 3 ---
        feat_expanded_4 = self.expand_channel4(latent)
        if self.sampling_mode == 'dwt':
            inp_dec_level3 = self.up_op(feat_expanded_4)
        else:
            inp_dec_level3 = self.up_layer3(feat_expanded_4)

        out_dec_level3 = self.decoder_level3((out_enc_level3, inp_dec_level3))

        # --- Upsample 3 -> 2 ---
        feat_expanded_3 = self.expand_channel3(out_dec_level3)
        if self.sampling_mode == 'dwt':
            inp_dec_level2 = self.up_op(feat_expanded_3)
        else:
            inp_dec_level2 = self.up_layer2(feat_expanded_3)

        out_dec_level2 = self.decoder_level2((out_enc_level2, inp_dec_level2))

        # --- Upsample 2 -> 1 ---
        feat_expanded_2 = self.expand_channel2(out_dec_level2)
        if self.sampling_mode == 'dwt':
            inp_dec_level1 = self.up_op(feat_expanded_2)
        else:
            inp_dec_level1 = self.up_layer1(feat_expanded_2)

        out_dec_level1 = self.decoder_level0((out_enc_level1, inp_dec_level1))

        # --- Output ---
        out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1


if __name__ == '__main__':
    import ptflops
    from thop import profile

    print("=== Testing Full Model (Ours) ===")
    model = LSTformer(dim=32, sampling_mode='dwt', fusion_mode='cafb')
    inputs = torch.randn(1, 3, 256, 256)
    flops, params = profile(model, (inputs,))
    print(f'Mode: DWT + CAFB | FLOPs: {flops / 1e9:.3f}G, Params: {params / 1e6:.3f}M')

    print("\n=== Testing Baseline (Conv + Concat) ===")
    model_base = LSTformer(dim=32, sampling_mode='conv', fusion_mode='concat')
    inputs = torch.randn(1, 3, 256, 256)
    flops, params = profile(model_base, (inputs,))
    print(f'Mode: Conv + Concat | FLOPs: {flops / 1e9:.3f}G, Params: {params / 1e6:.3f}M')




