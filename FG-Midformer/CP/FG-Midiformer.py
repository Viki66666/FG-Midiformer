import math
import numpy as np
import random

import torch
import torch.nn as nn
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
class SimamModule(nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(SimamModule, self).__init__()
        self.act = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.act(y)

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
        b, c, h, w = x.size()
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

# Former model: similar approach to "felix"
class MidiFormer(nn.Module):
    def __init__(self, formerConfig, e2w, w2e, use_fif):
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

        #Add SimAM module
        self.simam_module = SimamModule()

        self.use_fif = use_fif

    def forward(self, input_ids, attn_mask=None, output_hidden_states=True, mode="mlm"):
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

        #Apply SimAM module
        emb_linear_4d = emb_linear.unsqueeze(2)
        emb_linear_4d = self.simam_module(emb_linear_4d)
        emb_linear = emb_linear_4d.squeeze(2)

        # Apply EMA module
        b, seq_len, d_model = emb_linear.shape
        emb_linear_4d = emb_linear.view(b, d_model, 1, seq_len)  # reshape to 4D tensor for EMA
        ema_output = self.ema(emb_linear_4d)
        ema_output_3d = ema_output.view(b, seq_len, d_model)  # reshape back to 3D tensor

        # feed to former
        y = self.former(inputs_embeds=ema_output_3d, attention_mask=attn_mask, output_hidden_states=output_hidden_states)
        return y

    def get_rand_tok(self):
        c1, c2, c3, c4 = self.n_tokens[0], self.n_tokens[1], self.n_tokens[2], self.n_tokens[3]
        return np.array(
            [random.choice(range(c1)), random.choice(range(c2)), random.choice(range(c3)), random.choice(range(c4))])

if __name__ == '__main__':
    FormerModel = BertModel
    former_config = BertConfig(hidden_size=768, num_attention_heads=12, num_hidden_layers=12)

    e2w = {
        'Bar': {'Bar <PAD>': 0, 'Bar <MASK>': 1, 'Bar 1': 2, 'Bar 2': 3},
        'Position': {'Position <PAD>': 0, 'Position <MASK>': 1, 'Position 1': 2, 'Position 2': 3},
        'Pitch': {'Pitch <PAD>': 0, 'Pitch <MASK>': 1, 'Pitch 60': 2, 'Pitch 61': 3},
        'Duration': {'Duration <PAD>': 0, 'Duration <MASK>': 1, 'Duration 1': 2, 'Duration 2': 3}
    }
    w2e = {key: {v: k for k, v in e2w[key].items()} for key in e2w}

    midi_former = MidiFormer(former_config, e2w, w2e, use_fif=True)

    input_ids = torch.randint(0, 4, (2, 16, 4))  # 示例batch size 2, sequence length 16, 4个token类型
    attn_mask = torch.ones((2, 16))  # 示例注意力掩码

    output = midi_former(input_ids, attn_mask)

    print(output.last_hidden_state.shape)
