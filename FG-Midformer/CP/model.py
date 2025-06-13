import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from former import FormerModel
from transformers.models.bert.modeling_bert import BertAttention
from transformers.models.bert.configuration_bert import BertConfig
from transformers import BertModel

class Embeddings(nn.Module):
    def __init__(self, n_token, d_model):
        super().__init__()
        self.lut = nn.Embedding(n_token, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # Check if input x has 3 or 4 dimensions
        if len(x.size()) == 3:
            # If 3D, add a dummy dimension to make it 4D (B, C, 1, L) or (B, C, L, 1)
            x = x.unsqueeze(2)  # B, C, 1, L
            # or
            # x = x.unsqueeze(3)  # B, C, L, 1

        # Get batch size, channels, height, and width
        b, c, h, w = x.size()  # Now x is 4D, so this should work

        group_x = x.reshape(b * self.groups, -1, h, w)
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class SAFM(nn.Module):
    def __init__(self, dim, n_levels=4):
        super().__init__()
        # n_levels表示特征会被分割成多少个不同的尺度
        self.n_levels = n_levels
        # 每个尺度的特征通道数
        chunk_dim = dim // n_levels

        # Spatial Weighting：针对每个尺度的特征，使用深度卷积进行空间加权
        self.mfr = nn.ModuleList([nn.Conv2d(chunk_dim, chunk_dim, 3, 1, 1, groups=chunk_dim) for i in range(self.n_levels)])

        # Feature Aggregation：用于聚合不同尺度处理过的特征
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)

        # Activation：使用GELU激活函数
        self.act = nn.GELU()

    def forward(self, x):
        # x的形状为(B,C,H,W)，其中B是批次大小，C是通道数，H和W是高和宽
        h, w = x.size()[-2:]

        # 将输入特征在通道维度上分割成n_levels个尺度
        xc = x.chunk(self.n_levels, dim=1)

        out = []
        for i in range(self.n_levels):
            if i > 0:
                # 计算每个尺度下采样后的大小
                p_size = (h // 2**i, w // 2**i)
                # 对特征进行自适应最大池化，降低分辨率
                s = F.adaptive_max_pool2d(xc[i], p_size)
                # 对降低分辨率的特征应用深度卷积
                s = self.mfr[i](s)
                # 使用最近邻插值将特征上采样到原始大小
                s = F.interpolate(s, size=(h, w), mode='nearest')
            else:
                # 第一尺度直接应用深度卷积，不进行下采样
                s = self.mfr[i](xc[i])
            out.append(s)

        # 将处理过的所有尺度的特征在通道维度上进行拼接
        out = torch.cat(out, dim=1)
        # 通过1x1卷积聚合拼接后的特征
        out = self.aggr(out)
        # 应用GELU激活函数并与原始输入相乘，实现特征调制
        out = self.act(out) * x
        return out


# MidiFormer model with SEAttention
class MidiFormer(nn.Module):
    def __init__(self, formerConfig, e2w, w2e, use_fif, use_safm=False):
        super().__init__()

        self.former = FormerModel(formerConfig)
        formerConfig.d_model = formerConfig.hidden_size
        self.hidden_size = formerConfig.hidden_size
        self.formerConfig = formerConfig

        self.n_tokens = []
        for key in e2w:
            self.n_tokens.append(len(e2w[key]))
        self.emb_sizes = [256, 256, 256, 256]
        self.e2w = e2w
        self.w2e = w2e

        self.bar_pad_word = self.e2w['Bar']['Bar <PAD>']
        self.mask_word_np = np.array([self.e2w[etype]['%s <MASK>' % etype] for etype in self.e2w], dtype=np.long)
        self.pad_word_np = np.array([self.e2w[etype]['%s <PAD>' % etype] for etype in self.e2w], dtype=np.long)

        self.word_emb = []
        for i, key in enumerate(self.e2w):
            self.word_emb.append(Embeddings(self.n_tokens[i], self.emb_sizes[i]))
        self.word_emb = nn.ModuleList(self.word_emb)

        self.in_linear = nn.Linear(np.sum(self.emb_sizes), formerConfig.d_model)

        attn_config = BertConfig(hidden_size=256, num_attention_heads=4, intermediate_size=512)
        self.interaction_attn = BertAttention(attn_config)

        # Add EMA module
        self.ema = EMA(formerConfig.d_model)

        # Add SAFM module if specified
        self.use_safm = use_safm
        if self.use_safm:
            self.safm = SAFM(formerConfig.d_model)

        self.use_fif = use_fif

    def forward(self, input_ids, attn_mask=None, output_hidden_states=True, mode="mlm"):
        print(input_ids.shape)
        bs, slen, _ = input_ids.shape
        if mode == "mlm":
            special_mark = torch.zeros((bs, 1)).long().to(input_ids.device)
        else:
            special_mark = torch.zeros((bs, 1)).long().to(input_ids.device) + 1
        special_emb = self.former.embeddings.word_embeddings(special_mark)

        if not self.use_fif:
            embs = []
            for i, key in enumerate(self.e2w):
                embs.append(self.word_emb[i](input_ids[..., i]))
            embs = torch.cat([*embs], dim=-1)
        else:
            embs = []
            for i, key in enumerate(self.e2w):
                embs.append(self.word_emb[i](input_ids[..., i]).unsqueeze(2))  # B x L x 1 x d
            embs = torch.cat([*embs], dim=-2)  # B x L x F x d

            embs_shape = embs.shape
            embs = embs.view(-1, embs_shape[2], embs_shape[3])  # (B x L) x F x d

            self_attention_outputs = self.interaction_attn(embs)
            embs_interaction = self_attention_outputs[0]

            embs = embs_interaction.view(embs_shape[0], embs_shape[1], embs_shape[2], embs_shape[3]).reshape(
                (embs_shape[0], embs_shape[1], embs_shape[2] * embs_shape[3]))

        emb_linear = self.in_linear(embs)
        emb_linear = torch.cat([special_emb, emb_linear], dim=1)
        attn_mask = torch.cat([torch.ones((bs, 1)).to(input_ids.device), attn_mask], dim=1)

        # Apply EMA module
        b, seq_len, d_model = emb_linear.shape
        emb_linear_4d = emb_linear.view(b, d_model, 1, seq_len)  # reshape to 4D tensor for EMA
        ema_output = self.ema(emb_linear_4d)
        ema_output_3d = ema_output.view(b, seq_len, d_model)  # reshape back to 3D tensor

        # Apply SAFM module if specified
        if self.use_safm:
            ema_output_3d = self.safm(ema_output_3d)

        # feed to former
        y = self.former(inputs_embeds=ema_output_3d, attention_mask=attn_mask, output_hidden_states=output_hidden_states)
        return y

    def get_rand_tok(self):
        c1, c2, c3, c4 = self.n_tokens[0], self.n_tokens[1], self.n_tokens[2], self.n_tokens[3]
        return np.array(
            [random.choice(range(c1)), random.choice(range(c2)), random.choice(range(c3)), random.choice(range(c4))])
