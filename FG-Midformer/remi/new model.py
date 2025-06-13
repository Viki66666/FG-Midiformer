import math
import torch
import torch.nn as nn
from former import FormerModel

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

class MidiFormer(nn.Module):
    def __init__(self, formerConfig, e2w, w2e):
        super().__init__()
        self.former = FormerModel(formerConfig)
        formerConfig.d_model = formerConfig.hidden_size
        self.hidden_size = formerConfig.hidden_size
        self.formerConfig = formerConfig
        self.n_token = len(e2w)
        self.emb_size = 256
        self.e2w = e2w
        self.w2e = w2e
        self.pad_word = self.e2w['Pad_None']
        self.mask_word = self.e2w['Mask_None']
        self.word_emb = Embeddings(self.n_token, self.emb_size)
        self.in_linear = nn.Linear(self.emb_size, formerConfig.d_model)
        self.ema = EMA(formerConfig.hidden_size)  # 将EMA模块集成到MidiFormer中

    def forward(self, input_id, attn_mask=None, output_hidden_states=True, mode="mlm"):
        bs, slen = input_id.shape
        if mode == "mlm":
            special_mark = torch.zeros((bs, 1)).long().to(input_id.device)
        else:
            special_mark = torch.zeros((bs, 1)).long().to(input_id.device) + 1
        special_emb = self.former.embeddings.word_embeddings(special_mark)
        emb = self.word_emb(input_id)
        emb_linear = self.in_linear(emb)
        emb_linear = torch.cat([special_emb, emb_linear], dim=1)
        attn_mask = torch.cat([torch.ones((bs, 1)).to(input_id.device), attn_mask], dim=1)
        y = self.former(inputs_embeds=emb_linear, attention_mask=attn_mask, output_hidden_states=output_hidden_states)
        y = self.ema(y.last_hidden_state)  # 应用EMA模块处理最后的隐藏状态
        return y

# 测试EMA模块
if __name__ == '__main__':
    block = EMA(64)
    input = torch.rand(1, 64, 64, 64)
    output = block(input)
    print(output.shape)
