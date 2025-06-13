import math
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import List, Dict, Tuple
from sklearn.decomposition import PCA
from transformers import GPT2Config, GPT2Model

from model import MidiFormer
#from modelstart import TransformerModel
from transformers import BertModel
from transformers.models.bert.configuration_bert import BertConfig

class TransformerModel(nn.Module):
    def __init__(
        self,
        n_dims: int,
        n_positions: int,
        n_embd: int = 128,
        n_layer: int = 12,
        n_head: int = 4,
        n_y: int = 1,
        model_name: str = None,
    ):
        super(TransformerModel, self).__init__()
        configuration = GPT2Config(
            n_positions=(n_y + 1) * n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"

        self.n_positions = n_positions
        self.n_dims = n_dims
        self._read_in = nn.Linear(n_dims, n_embd)
        self._backbone = GPT2Model(configuration)
        self._read_out = nn.Linear(n_embd, 1)
        self.y_step_size = n_y + 1
        self.n_y = n_y
        self.sigmoid = torch.nn.Sigmoid()

    @staticmethod
    def _combine(xs_b: torch.Tensor, ys_b: torch.Tensor):
        """Interleaves the x's and the y's into a single sequence."""
        bsize, points, dim = xs_b.shape
        ys_b_wide = torch.cat(
            (
                ys_b.view(bsize, points, 1),
                torch.zeros(bsize, points, dim - 1, device=ys_b.device),
            ),
            axis=2,
        )
        zs = torch.stack((xs_b, ys_b_wide), dim=2)
        zs = zs.view(bsize, 2 * points, dim)
        return zs

    def _combine_gen(self, xs_b: torch.Tensor, ys_b: torch.Tensor):
        """For sequences with more than one y's, Interleaves the x's
        and the y's into a single sequence."""
        bsize, points, dim = xs_b.shape
        ys_list = []
        for i in range(self.n_y):
            ys_b_i = ys_b[i, ::]
            ys_b_i_wide = torch.cat(
                (
                    ys_b_i.view(bsize, points, 1),
                    torch.zeros(bsize, points, dim - 1, device=ys_b.device),
                ),
                axis=2,
            )
            ys_list.append(ys_b_i_wide)
        zs = torch.stack((xs_b, *ys_list), dim=2)
        zs = zs.view(bsize, (self.n_y + 1) * points, dim)

        return zs

    def _step(self, zs: torch.Tensor):
        inds = torch.arange(int(zs.shape[1] / 2))
        embeds = self._read_in(zs)
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        y_outs = self._read_out(output)

        predictions = y_outs[:, ::2, 0][:, inds]
        return predictions

    def predict(self, zs: torch.Tensor):
        inds = torch.arange(int(zs.shape[1] / 2))
        embeds = self._read_in(zs)
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        y_outs = self._read_out(output)

        predictions = y_outs[:, ::2, 0][:, inds]
        pred = self.sigmoid(predictions)[0][-1].item()

        if pred >= 0.5:
            return 1
        else:
            return 0

    def forward(self, xs: torch.Tensor, ys: torch.Tensor, inds=None):
        # Predicting a *sequence* of y's
        if len(ys.shape) > 2:
            inds = torch.arange(ys.shape[-1])
            zs = self._combine_gen(xs, ys)
            embeds = self._read_in(zs)

            output = self._backbone(
                inputs_embeds=embeds,
            ).last_hidden_state
            prediction = self._read_out(output)

            preds_y = []
            for i in range(self.n_y):
                preds_y.append(prediction[:, i :: self.y_step_size, 0][:, inds])
            return preds_y
        # Predicting a single y
        else:
            # if predicting a single y
            if inds is None:
                inds = torch.arange(ys.shape[1])
            else:
                inds = torch.tensor(inds)
                if max(inds) >= ys.shape[1] or min(inds) < 0:
                    raise ValueError(
                        "inds contain indices where xs and ys are not defined"
                    )
            zs = self._combine(xs, ys)
            embeds = self._read_in(zs)
            output = self._backbone(inputs_embeds=embeds).last_hidden_state
            prediction = self._read_out(output)
            return prediction[:, ::2, 0][
                :, inds
            ]  # return hiddens pertaining to x's indexes






class SelfAttention(nn.Module):
    def __init__(self, input_dim, da, r):
        '''
        Args:
            input_dim (int): batch, seq, input_dim
            da (int): number of features in hidden layer from self-attn
            r (int): number of aspects of self-attn
        '''
        super(SelfAttention, self).__init__()
        self.ws1 = nn.Linear(input_dim, da, bias=False)
        self.ws2 = nn.Linear(da, r, bias=False)

    def forward(self, h):
        attn_mat = F.softmax(self.ws2(torch.tanh(self.ws1(h))), dim=1)
        attn_mat = attn_mat.permute(0, 2, 1)
        return attn_mat


class TokenClassification(nn.Module):
    def __init__(self, midi_former, class_num, hs, mode="mlm"):
        super().__init__()

        self.midi_former = midi_former
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hs, 256),
            nn.ReLU(),
            nn.Linear(256, class_num)
        )

        self.mode = mode

    def forward(self, y, attn, layer):
        # feed to former
        y = self.midi_former(y, attn, output_hidden_states=True, mode=self.mode)
        y = y.hidden_states[layer]
        y = y[:, 1:, :]
        return self.classifier(y)


class SequenceClassification(nn.Module):
    def __init__(self, midi_former, class_num, hs, da=128, r=4, mode="mlm"):
        super(SequenceClassification, self).__init__()
        self.midi_former = midi_former
        self.attention = SelfAttention(hs, da, r)
        self.classifier = nn.Sequential(
            nn.Linear(hs * r, 256),
            nn.ReLU(),
            nn.Linear(256, class_num)
        )

        self.mode = mode

    def forward(self, x, attn, layer):
        x = self.midi_former(x, attn, output_hidden_states=True)
        x = x.hidden_states[layer]
        x = x[:, 1:, :]
        attn_mat = self.attention(x)
        m = torch.bmm(attn_mat, x)
        flatten = m.view(m.size()[0], -1)
        res = self.classifier(flatten)
        return res


class TartReasoningHead:
    def __init__(
            self,
            n_dims: int,
            n_positions: int,
            n_embd: int,
            n_head: int,
            n_layer: int,
            n_y: int,
            path_to_pretrained_head: str,
    ):
        self.n_dims = n_dims
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.n_y = n_y

        self.tart_head = TransformerModel(
            n_dims=n_dims,
            n_positions=n_positions,
            n_embd=n_embd,
            n_head=n_head,
            n_layer=n_layer,
            n_y=n_y,
        )

        tweights = torch.load(path_to_pretrained_head)
        self.tart_head.load_state_dict(tweights, strict=False)
        self.tart_head = self.tart_head.cuda()


class TartEmbeddingLayer:
    domain: str
    embed_type: str

    def __init__(
            self,
            embed_model_name: str,
            num_pca_components: int,
    ):
        self.embed_model_name = embed_model_name
        self.num_pca_components = num_pca_components

    def _load_model_tokenizer(self):
        raise NotImplementedError

    def compute_pca_with_whitening(self, X_tr_embed, X_tst_embed):
        pca = PCA(n_components=self.num_pca_components)
        pca.fit(X_tr_embed)
        X_tr_pca_cor = pca.transform(X_tr_embed)
        X_tst_pca_cor = pca.transform(X_tst_embed)

        X_tr_pca_cor_mean = X_tr_pca_cor.mean(axis=0)
        X_tr_pca_cor_m0 = X_tr_pca_cor - X_tr_pca_cor_mean
        X_tst_pca_cor_m0 = X_tst_pca_cor - X_tr_pca_cor_mean

        cov_X_cor = np.cov(X_tr_pca_cor_m0, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_X_cor)
        D = np.diag(1.0 / np.sqrt(eigenvalues))
        X_tr_pca_cor_white = (eigenvectors @ D @ eigenvectors.T @ X_tr_pca_cor_m0.T).T
        X_tst_pca_cor_white = (eigenvectors @ D @ eigenvectors.T @ X_tst_pca_cor_m0.T).T

        return X_tr_pca_cor_white, X_tst_pca_cor_white

    def embed(self):
        raise NotImplementedError

    def get_domain(self):
        return self.domain

    def get_embed_strategy(self):
        return self.embed_type

    def get_embed_model_name(self):
        return self.embed_model_name


class TartEmbeddingLayerAC(TartEmbeddingLayer):
    _domain: str
    _embed_type: str
    _hf_model_family: str

    def __init__(
            self,
            embed_model_name: str,
            num_pca_components: int,
    ):
        super().__init__(embed_model_name, num_pca_components)

    def _load_model_tokenizer(self):
        pass

    def embed(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pass


class Tart:
    def __init__(
            self,
            domain: str,
            embed_model_name: str,
            path_to_pretrained_head: str,
            tart_head_config: dict,
            embed_method: str = "stream",
            num_pca_components: int = None,
            path_to_finetuned_embed_model: str = None,
            cache_dir: str = None,
    ):
        from .registry import EMBEDDING_REGISTRY_AC, DOMAIN_REGISTRY

        assert embed_method in ["loo", "vanilla", "stream"], "embed_method not valid. Must be one of: loo, base, stream"

        assert domain in DOMAIN_REGISTRY["supported"], "domain not valid. Must be one of: text, audio, image"

        self.embed_method = embed_method
        self.config = tart_head_config

        if num_pca_components is None:
            num_pca_components = self.config["n_dims"]

        self.num_pca_components = num_pca_components
        self.domain = domain
        model_name = embed_model_name.split("/")[-1]
        self.embed_layer = EMBEDDING_REGISTRY_AC[model_name][embed_method](
            embed_model_name=embed_model_name,
            num_pca_components=self.num_pca_components,
            cache_dir=cache_dir,
            path_to_finetuned_embed_model=path_to_finetuned_embed_model,
        )
        self._load_tart_head(path_to_pretrained_head, tart_head_config)

    def set_embed_model(self, embed_model_name: str, embed_method: str = "stream", cache_dir: str = None):
        from .registry import EMBEDDING_REGISTRY_AC

        model_name = embed_model_name.split("/")[-1]

        print(f"loading embed model: {embed_model_name} ...")

        self.embed_layer = EMBEDDING_REGISTRY_AC[model_name][embed_method](
            embed_model_name=embed_model_name,
            num_pca_components=self.num_pca_components,
            cache_dir=cache_dir,
        )

    def _load_tart_head(self, path_to_pretrained_head, tart_head_config):
        self.tart_head = TartReasoningHead(
            n_dims=tart_head_config["n_dims"],
            n_positions=tart_head_config["n_positions"],
            n_embd=tart_head_config["n_embd"],
            n_head=tart_head_config["n_head"],
            n_layer=tart_head_config["n_layer"],
            n_y=tart_head_config["n_y"],
            path_to_pretrained_head=path_to_pretrained_head,
        ).tart_head

    def _format_eval_sequence(
            self,
            X_train_hidden: torch.Tensor,
            y_train_hidden: torch.Tensor,
            X_test_hidden: torch.Tensor,
            y_test_hidden: torch.Tensor,
    ) -> List:
        eval_seqs = []
        for test_idx in range(y_test_hidden.shape[-1]):
            xs = torch.cat(
                [X_train_hidden, X_test_hidden[test_idx, :].unsqueeze(0)],
                dim=0,
            ).unsqueeze(0)
            ys = torch.cat(
                [y_train_hidden, y_test_hidden[test_idx:test_idx + 1]],
                dim=0,
            ).unsqueeze(0)
            eval_seqs.append(
                {
                    "xs": xs,
                    "ys": ys,
                }
            )
        return eval_seqs

    def _format_train_sequence(
            self,
            X_train_hidden: torch.Tensor,
            y_train_hidden: torch.Tensor,
    ) -> List:
        xs = X_train_hidden.unsqueeze(0)
        ys = y_train_hidden.unsqueeze(0)
        return [
            {
                "xs": xs,
                "ys": ys,
            }
        ]

    def _set_tr_mode(self):
        self.tart_head.train()
        self.embed_layer.train()

    def _set_ev_mode(self):
        self.tart_head.eval()
        self.embed_layer.eval()

    def eval_tart(
            self,
            X_train: torch.Tensor,
            y_train: torch.Tensor,
            X_test: torch.Tensor,
            y_test: torch.Tensor,
            ret_np: bool = False,
    ) -> Dict[str, torch.Tensor]:

        X_train_embed, y_train_embed, X_test_embed, y_test_embed = self.embed_layer.embed()

        print(f"computing PCA with whitening...")
        X_train_embed_pca, X_test_embed_pca = self.embed_layer.compute_pca_with_whitening(
            X_train_embed,
            X_test_embed,
        )

        print(f"formatting sequences for TART evaluation...")
        eval_seqs = self._format_eval_sequence(
            X_train_hidden=torch.tensor(X_train_embed_pca),
            y_train_hidden=torch.tensor(y_train_embed),
            X_test_hidden=torch.tensor(X_test_embed_pca),
            y_test_hidden=torch.tensor(y_test_embed),
        )

        print(f"evaluating TART model...")
        y_preds, attns = self._eval_seqs(eval_seqs)

        if ret_np:
            y_preds = y_preds.cpu().numpy()

        return {"y_preds": y_preds, "attns": attns}

    def train_tart(
            self,
            X_train: torch.Tensor,
            y_train: torch.Tensor,
            ret_np: bool = False,
    ) -> Dict[str, torch.Tensor]:

        X_train_embed, y_train_embed = self.embed_layer.embed(X_train, y_train)

        print(f"computing PCA with whitening...")
        X_train_embed_pca, _ = self.embed_layer.compute_pca_with_whitening(
            X_train_embed,
            X_train_embed,
        )

        print(f"formatting sequences for TART training...")
        train_seqs = self._format_train_sequence(
            X_train_hidden=torch.tensor(X_train_embed_pca),
            y_train_hidden=torch.tensor(y_train_embed),
        )

        print(f"training TART model...")
        self._train_seqs(train_seqs)

        if ret_np:
            y_train = y_train.cpu().numpy()

        return {"y_train": y_train}

    def _eval_seqs(self, eval_seqs: List) -> Tuple[torch.Tensor, torch.Tensor]:
        y_preds = []
        attns = []
        for seq in tqdm(eval_seqs):
            xs = seq["xs"].cuda()
            ys = seq["ys"].cuda()
            with torch.no_grad():
                y_pred, attn = self.tart_head(xs, ys)
            y_preds.append(y_pred)
            attns.append(attn)
        y_preds = torch.cat(y_preds, dim=0)
        attns = torch.cat(attns, dim=0)
        return y_preds, attns

    def _train_seqs(self, train_seqs: List) -> None:
        for seq in tqdm(train_seqs):
            xs = seq["xs"].cuda()
            ys = seq["ys"].cuda()
            self.tart_head(xs, ys)


# if __name__ == "__main__":
#     FormerModel = BertModel
#     former_config = BertConfig(hidden_size=768, num_attention_heads=12, num_hidden_layers=12)
#
#     e2w = {
#         'Bar': {'Bar <PAD>': 0, 'Bar <MASK>': 1, 'Bar 1': 2, 'Bar 2': 3},
#         'Position': {'Position <PAD>': 0, 'Position <MASK>': 1, 'Position 1': 2, 'Position 2': 3},
#         'Pitch': {'Pitch <PAD>': 0, 'Pitch <MASK>': 1, 'Pitch 60': 2, 'Pitch 61': 3},
#         'Duration': {'Duration <PAD>': 0, 'Duration <MASK>': 1, 'Duration 1': 2, 'Duration 2': 3}
#     }
#     w2e = {key: {v: k for k, v in e2w[key].items()} for key in e2w}
#
#     midi_former = MidiFormer(former_config, e2w, w2e, use_fif=True)
#
#     # 创建 SequenceClassification 模型
#     sequence_classification_model = SequenceClassification(midi_former, class_num=10, hs=768)
#
#     # 假设输入维度是 (batch_size, sequence_length, feature_dim)
#     dummy_input = torch.randn(2, 100, 768)
#     dummy_attn = torch.randn(2, 100, 768)
#
#     # 计算输出
#     output = sequence_classification_model(dummy_input, dummy_attn, layer=5)
#     print(f"Sequence Classification output shape: {output.shape}")
#
#     # 创建 TokenClassification 模型
#     token_classification_model = TokenClassification(midi_former, class_num=10, hs=768)
#
#     # 计算输出
#     output = token_classification_model(dummy_input, dummy_attn, layer=5)
#     print(f"Token Classification output shape: {output.shape}")

