# PromptIR: Prompting for All-in-One Blind Image Restoration
# Vaishnav Potlapalli, Syed Waqas Zamir, Salman Khan, and Fahad Shahbaz Khan
# https://arxiv.org/abs/2306.13090


import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers

from einops import rearrange
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import LayerNorm2d

# Layer Norm


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
        self.norm_type = LayerNorm_type
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = LayerNorm2d(dim, 1e-5)

    def forward(self, x):
        if self.norm_type == 'BiasFree':
            h, w = x.shape[-2:]
            return to_4d(self.body(to_3d(x)), h, w)
        else:
            return self.body(x)


##########################################################################
# Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
# Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class resblock(nn.Module):
    def __init__(self, dim):

        super(resblock, self).__init__()
        # self.norm = LayerNorm(dim, LayerNorm_type='BiasFree')

        self.body = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PReLU(),
                                  nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, x):
        res = self.body((x))
        res += x
        return res


##########################################################################
# Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


##########################################################################
# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################
# Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3,
                              stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


##########################################################################
# ---------- Prompt Gen Module -----------------------
class PromptGenBlock(nn.Module):
    def __init__(self, prompt_dim=128, prompt_len=5, prompt_size=96, lin_dim=192, num_expert=1):
        super(PromptGenBlock, self).__init__()
        self.prompt_param = nn.Parameter(torch.rand(
            1, prompt_len, prompt_dim, prompt_size, prompt_size), requires_grad=True)
        self.linear_layer = nn.Linear(lin_dim, prompt_len)
        self.conv3x3 = nn.Conv2d(
            prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.num_expert = num_expert
        self.prompt_len = prompt_len

    def forward(self, x, is_neg=False):
        """
        Args:
            x: Input features
            is_neg (bool): 如果为 True，则返回与当前输入不匹配的随机负样本 Prompt
        """
        B, C, H, W = x.shape
        emb = x.mean(dim=(-2, -1))

        # 计算每个 Prompt 的权重
        prompt_weights = F.softmax(self.linear_layer(emb), dim=1).to(x.dtype)

        # 获取当前认为最好的 Top-K (Positive Candidates)
        topk_weights, topk_experts = torch.topk(
            prompt_weights, self.num_expert)

        if not is_neg:
            # === 正样本模式 (Normal Path) ===
            selected_weights = prompt_weights
            selected_experts = topk_experts

            # 构建稀疏权重矩阵
            exp_weights = torch.zeros_like(prompt_weights)
            exp_weights.scatter_(1, selected_experts,
                                 selected_weights.gather(1, selected_experts))

            # 生成 Prompt
            prompt = self._generate_prompt(exp_weights, B)

        else:
            # === 负样本模式 (Counterfactual / Negative Path) ===
            with torch.no_grad():
                # 1. 创建掩码：标记 Top-K 位置为 False，其他位置为 True
                mask = torch.ones_like(prompt_weights, dtype=torch.bool)
                mask.scatter_(1, topk_experts, False)

                # 2. 生成随机噪声作为选择依据
                noise = torch.rand_like(prompt_weights)

                # 3. 将 Top-K 位置的噪声设为负无穷，确保不会被选中
                noise.masked_fill_(~mask, -float('inf'))

                # 4. 在剩余的 (非 Top-K) 候选中选取 Top-K 个作为负样本
                _, neg_experts = torch.topk(noise, self.num_expert)

                # 5. 获取负样本对应的原始权重
                neg_weights_values = prompt_weights.gather(1, neg_experts)

                # 6. 构建稀疏权重
                exp_weights = torch.zeros_like(prompt_weights)
                exp_weights.scatter_(1, neg_experts, neg_weights_values)

                # 7. 生成 Prompt 并 Detach (确保没有梯度传回 Prompt 参数)
                prompt = self._generate_prompt(exp_weights, B).detach()

        # 后处理沿用：上采样 + 卷积
        prompt = F.interpolate(
            prompt, (H, W), mode="bilinear", align_corners=False)
        prompt = self.conv3x3(prompt)

        return prompt

    def _generate_prompt(self, exp_weights, batch_size):
        """辅助函数：根据权重组合 Prompt"""
        # (B, prompt_len, 1, 1, 1)
        weights = exp_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # (B, prompt_len, dim, size, size)
        params = self.prompt_param.unsqueeze(0).repeat(
            batch_size, 1, 1, 1, 1, 1).squeeze(1)

        # Weighted sum: (B, dim, size, size)
        prompt = torch.sum(weights * params, dim=1)
        return prompt

##########################################################################
# ---------- PromptIR -----------------------


@ARCH_REGISTRY.register()
class CPLIR(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  # Other option 'BiasFree'
                 decoder=False,
                 prompt_len=5,
                 fix_backbone=False,
                 use_uncertainty=False,
                 privilege_post=False,
                 fix_uncertainty=False,
                 num_expert=1
                 #  save_path='/data_share/csgwu/paper_with_code/uclsr/model_zoo/uncertainty_aaai/prompt/ours/denoise'
                 ):

        super(CPLIR, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.privilege_post = privilege_post
        self.decoder = decoder

        if self.decoder:
            self.prompt1 = PromptGenBlock(
                prompt_dim=64, prompt_len=prompt_len, prompt_size=64, lin_dim=96, num_expert=num_expert)
            self.prompt2 = PromptGenBlock(
                prompt_dim=128, prompt_len=prompt_len, prompt_size=32, lin_dim=192, num_expert=num_expert)
            self.prompt3 = PromptGenBlock(
                prompt_dim=320, prompt_len=prompt_len, prompt_size=16, lin_dim=384, num_expert=num_expert)

        # self.chnl_reduce1 = nn.Conv2d(64,64,kernel_size=1,bias=bias)
        # self.chnl_reduce2 = nn.Conv2d(128,128,kernel_size=1,bias=bias)
        # self.chnl_reduce3 = nn.Conv2d(320,256,kernel_size=1,bias=bias)

        # self.reduce_noise_channel_1 = nn.Conv2d(dim + 64,dim,kernel_size=1,bias=bias)
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  # From Level 1 to Level 2

        # self.reduce_noise_channel_2 = nn.Conv2d(int(dim*2**1) + 128,int(dim*2**1),kernel_size=1,bias=bias)
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1],
                                            ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim*2**1))  # From Level 2 to Level 3

        # self.reduce_noise_channel_3 = nn.Conv2d(int(dim*2**2) + 256,int(dim*2**2),kernel_size=1,bias=bias)
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2],
                                            ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2))  # From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3],
                                    ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim*2**2))  # From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(
            int(dim*2**1)+192, int(dim*2**2), kernel_size=1, bias=bias)
        self.noise_level3 = TransformerBlock(dim=int(
            dim*2**2) + 512, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level3 = nn.Conv2d(
            int(dim*2**2)+512, int(dim*2**2), kernel_size=1, bias=bias)

        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2],
                                            ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim*2**2))  # From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(
            int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.noise_level2 = TransformerBlock(dim=int(
            dim*2**1) + 224, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level2 = nn.Conv2d(
            int(dim*2**1)+224, int(dim*2**2), kernel_size=1, bias=bias)

        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1],
                                            ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        # From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.up2_1 = Upsample(int(dim*2**1))

        self.noise_level1 = TransformerBlock(dim=int(
            dim*2**1)+64, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level1 = nn.Conv2d(
            int(dim*2**1)+64, int(dim*2**1), kernel_size=1, bias=bias)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0],
                                            ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                        bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        self.output = nn.Conv2d(
            int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        self.count = 0

        # self.save_path = save_path
        self.use_uncertainty = use_uncertainty
        if use_uncertainty:

            self.var_conv = nn.Sequential(
                nn.Conv2d(int(dim*2**1), int(dim*2**1), 3, 1, 1), nn.ELU(),
                nn.Conv2d(int(dim*2**1), int(dim*2**1), 3, 1, 1), nn.ELU(),
                nn.Conv2d(int(dim*2**1), out_channels, 3, 1, 1), nn.Softplus()
            )
            if fix_uncertainty:
                for name, p in self.var_conv.named_parameters():
                    p.requires_grad = False

        if fix_backbone:
            for name, p in self.named_parameters():
                if (not 'prompt' in name) and (not 'var_conv' in name):
                    p.requires_grad = False

    def forward(self, inp_img, is_neg=False):
        # sample_prompt = {}

        inp_enc_level1 = self.patch_embed(inp_img)

        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)

        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)

        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)

        latent = self.latent(inp_enc_level4)

        if self.decoder:
            dec3_param = self.prompt3(latent, is_neg=is_neg)
            # sample_prompt['prompt3'] = dec3_param

            latent = torch.cat([latent, dec3_param], 1)
            latent = self.noise_level3(latent)
            latent = self.reduce_noise_level3(latent)

        inp_dec_level3 = self.up4_3(latent)

        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)

        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        if self.decoder:
            dec2_param = self.prompt2(out_dec_level3, is_neg=is_neg)
            # sample_prompt['prompt2'] = dec2_param

            out_dec_level3 = torch.cat([out_dec_level3, dec2_param], 1)
            out_dec_level3 = self.noise_level2(out_dec_level3)
            out_dec_level3 = self.reduce_noise_level2(out_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)

        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        if self.decoder:
            dec1_param = self.prompt1(out_dec_level2, is_neg=is_neg)
            # sample_prompt['prompt1'] = dec1_param

            out_dec_level2 = torch.cat([out_dec_level2, dec1_param], 1)
            out_dec_level2 = self.noise_level1(out_dec_level2)
            out_dec_level2 = self.reduce_noise_level1(out_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        out_dec_level1 = self.refinement(out_dec_level1)

        output = self.output(out_dec_level1) + inp_img

        if self.use_uncertainty:
            un = self.var_conv(out_dec_level1)

            return output, un
        else:
            return output
