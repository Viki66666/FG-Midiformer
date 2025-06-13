import numpy as np
import random
import math
import torch
import torch.nn as nn
from former import FormerModel
from thop import profile  # 引入 thop 库来计算模型的 FLOPs 和参数数量
from transformers.models.bert.modeling_bert import BertAttention
from transformers.models.bert.configuration_bert import BertConfig
from transformers import BertModel

# 定义 SimAM 模块
class Simam_module(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(Simam_module, self).__init__()
        self.act = nn.Sigmoid()  # 使用 Sigmoid 激活函数
        self.e_lambda = e_lambda  # 定义平滑项 e_lambda，防止分母为0

    def forward(self, x):
        b, c, h, w = x.size()  # 获取输入 x 的尺寸
        n = w * h - 1  # 计算特征图的元素数量减一，用于下面的归一化
        # 计算输入特征 x 与其均值之差的平方
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        # 计算注意力权重 y，这里实现了 SimAM 的核心计算公式
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        # 返回经过注意力加权的输入特征
        return x * self.act(y)

# 定义 EMA 模块
class EMA_module(nn.Module):
    def __init__(self, alpha=0.9):
        super(EMA_module, self).__init__()
        self.alpha = alpha
        self.register_buffer('ema', None)

    def forward(self, x):
        if self.ema is None:
            self.ema = x.clone()
        else:
            self.ema = self.alpha * self.ema + (1.0 - self.alpha) * x
        return self.ema

# 定义 Embeddings 模块
class Embeddings(nn.Module):
    def __init__(self, n_token, d_model):
        super().__init__()
        self.lut = nn.Embedding(n_token, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

# 定义 MidiFormer 模型
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
        self.simam = Simam_module()  # 添加 SimAM 模块
        self.ema = EMA_module()      # 添加 EMA 模块
        self.in_linear = nn.Linear(self.emb_size, formerConfig.d_model)

    def forward(self, input_id, attn_mask=None, output_hidden_states=True, mode="mlm"):
        bs, slen = input_id.shape
        if mode == "mlm":
            special_mark = torch.zeros((bs, 1)).long().to(input_id.device)
        else:
            special_mark = torch.zeros((bs, 1)).long().to(input_id.device) + 1
        special_emb = self.former.embeddings.word_embeddings(special_mark)

        emb = self.word_emb(input_id)
        emb = emb.permute(0, 2, 1).unsqueeze(-1)  # 调整维度以适应 SimAM 模块
        emb = self.simam(emb)  # 应用 SimAM 模块
        emb = emb.squeeze(-1).permute(0, 2, 1)  # 恢复维度

        emb = self.ema(emb)  # 应用 EMA 模块

        emb_linear = self.in_linear(emb)
        emb_linear = torch.cat([special_emb, emb_linear], dim=1)
        attn_mask = torch.cat([torch.ones((bs, 1)).to(input_id.device), attn_mask], dim=1)

        y = self.former(inputs_embeds=emb_linear, attention_mask=attn_mask,
                        output_hidden_states=output_hidden_states)
        return y

# 示例使用
if __name__ == '__main__':
    # FormerModel = BertModel
    # former_config = BertConfig(hidden_size=768, num_attention_heads=12, num_hidden_layers=12)
    #
    # e2w = {
    #     'Bar': {'Bar <PAD>': 0, 'Bar <MASK>': 1, 'Bar 1': 2, 'Bar 2': 3},
    #     'Position': {'Position <PAD>': 0, 'Position <MASK>': 1, 'Position 1': 2, 'Position 2': 3},
    #     'Pitch': {'Pitch <PAD>': 0, 'Pitch <MASK>': 1, 'Pitch 60': 2, 'Pitch 61': 3},
    #     'Duration': {'Duration <PAD>': 0, 'Duration <MASK>': 1, 'Duration 1': 2, 'Duration 2': 3}
    # }
    # w2e = {key: {v: k for k, v in e2w[key].items()} for key in e2w}
    #
    # midi_former = MidiFormer(former_config, e2w, w2e)
    #
    # input_ids = torch.randint(0, 4, (2, 16, 4))  # 示例batch size 2, sequence length 16, 4个token类型
    # attn_mask = torch.ones((2, 16))  # 示例注意力掩码
    #
    # output = midi_former(input_ids, attn_mask)
    #
    # print(output.last_hidden_state.shape)
    model = Simam_module() # 实例化SimAM模块并移到GPU上
    x = torch.randn(1, 3, 64, 64)  # 创建一个随机输入并移到GPU上
    y = model(x)  # 将输入传递给模型
    print(y.size())  # 打印输出尺寸
    # 使用thop库计算模型的FLOPs和参数数量
    flops, params = profile(model, (x,))
    print(flops / 1e9)  # 打印以Giga FLOPs为单位的浮点操作数
    print(params)  # 打印模型参数数量

    import math
    import numpy as np
    import random

    import torch
    import torch.nn as nn
    from former import FormerModel
    from former import FormerModel
    from transformers.models.bert.modeling_bert import BertAttention
    from transformers.models.bert.configuration_bert import BertConfig
    from transformers import BertModel


    # 定义EMA模块
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
            self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1,
                                     padding=0)
            self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1,
                                     padding=1)

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


    class Embeddings(nn.Module):
        def __init__(self, n_token, d_model):
            super().__init__()
            self.lut = nn.Embedding(n_token, d_model)
            self.d_model = d_model

        def forward(self, x):
            return self.lut(x) * math.sqrt(self.d_model)


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
            self.ema = EMA(self.emb_size)  # 添加EMA模块
            self.in_linear = nn.Linear(self.emb_size, formerConfig.d_model)

        def forward(self, input_id, attn_mask=None, output_hidden_states=True, mode="mlm"):
            bs, slen = input_id.shape
            if mode == "mlm":
                special_mark = torch.zeros((bs, 1)).long().to(input_id.device)
            else:
                special_mark = torch.zeros((bs, 1)).long().to(input_id.device) + 1
            special_emb = self.former.embeddings.word_embeddings(special_mark)

            emb = self.word_emb(input_id)
            emb = emb.permute(0, 2, 1).unsqueeze(-1)  # 调整维度以适应EMA模块
            emb = self.ema(emb)  # 应用EMA模块
            emb = emb.squeeze(-1).permute(0, 2, 1)  # 恢复维度

            emb_linear = self.in_linear(emb)
            emb_linear = torch.cat([special_emb, emb_linear], dim=1)
            attn_mask = torch.cat([torch.ones((bs, 1)).to(input_id.device), attn_mask], dim=1)

            y = self.former(inputs_embeds=emb_linear, attention_mask=attn_mask,
                            output_hidden_states=output_hidden_states)
            return y

