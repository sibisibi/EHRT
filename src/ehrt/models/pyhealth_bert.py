import math
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from pyhealth.datasets import SampleEHRDataset
from pyhealth.models import BaseModel
from pyhealth.tokenizer import Tokenizer


# VALID_OPERATION_LEVEL = ["visit", "event"]
# SPECIAL_TOKENS = ["<pad>", "<unk>", "<mask>", "<cls>"]
VALID_MODE = ["binary", "multiclass"]


class Attention(nn.Module):
    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = torch.softmax(scores, dim=-1)
        if mask is not None:
            p_attn = p_attn.masked_fill(mask == 0, 0)
        if dropout is not None:
            p_attn = dropout(p_attn)
 
        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
    
        self.linear_layers = nn.ModuleList(
            [nn.Linear(d_model, d_model, bias=False) for _ in range(3)]
        )
        self.output_linear = nn.Linear(d_model, d_model, bias=False)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

        self.attn_gradients = None
        self.attn_map = None

    # helper functions for interpretability
    def get_attn_map(self):
        return self.attn_map 
    
    def get_attn_grad(self):
        return self.attn_gradients

    def save_attn_grad(self, attn_grad):
        self.attn_gradients = attn_grad 

    # register_hook option allows us to save the gradients in backwarding
    def forward(self, query, key, value, mask=None, register_hook = False):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linear_layers, (query, key, value))
        ]
        
        # 2) Apply attention on all the projected vectors in batch.
        if mask is not None:
            mask = mask.unsqueeze(1)
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        
        self.attn_map = attn # save the attention map
        if register_hook:
            attn.register_hook(self.save_attn_grad)
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
  
        return self.output_linear(x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x, mask=None):
        x = self.w_2(self.dropout(self.activation(self.w_1(x))))
        if mask is not None:
            mask = mask.sum(dim=-1) > 0
            x[~mask] = 0
        return x


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class TransformerBlock(nn.Module):
    """Transformer block.

    MultiHeadedAttention + PositionwiseFeedForward + SublayerConnection

    Args:
        hidden: hidden size of transformer.
        attn_heads: head sizes of multi-head attention.
        dropout: dropout rate.
    """

    def __init__(self, hidden, attn_heads, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(
            d_model=hidden, d_ff=4 * hidden, dropout=dropout
        )
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask=None, register_hook = False):
        """Forward propagation.

        Args:
            x: [batch_size, seq_len, hidden]
            mask: [batch_size, seq_len, seq_len]

        Returns:
            A tensor of shape [batch_size, seq_len, hidden]
        """
        x = self.input_sublayer(x, lambda _x: self.attention(_x, _x, _x, mask=mask, register_hook=register_hook))
        x = self.output_sublayer(x, lambda _x: self.feed_forward(_x, mask=mask))
        return self.dropout(x)


class TransformerLayer(nn.Module):
    def __init__(self, feature_size, heads=1, dropout=0.5, num_layers=1):
        super(TransformerLayer, self).__init__()
        self.transformer = nn.ModuleList(
            [TransformerBlock(feature_size, heads, dropout) for _ in range(num_layers)]
        )

    def forward(
        self, x: torch.tensor, mask: Optional[torch.tensor] = None, register_hook = False
    ) -> Tuple[torch.tensor, torch.tensor]:
        """Forward propagation.

        Args:
            x: a tensor of shape [batch size, sequence len, feature_size].
            mask: an optional tensor of shape [batch size, sequence len], where
                1 indicates valid and 0 indicates invalid.

        Returns:
            emb: a tensor of shape [batch size, sequence len, feature_size],
                containing the output features for each time step.
            cls_emb: a tensor of shape [batch size, feature_size], containing
                the output features for the first time step.
        """
        if mask is not None:
            mask = torch.einsum("ab,ac->abc", mask, mask)
        for transformer in self.transformer:
            x = transformer(x, mask, register_hook)
        emb = x
        
        return emb


class Bert(BaseModel):
    def __init__(
        self,
        dataset: SampleEHRDataset,
        feature_keys: List[str],
        pretrained_emb: str = None,
        embedding_dim: int = 128,
        feat_tokenizers = None,
        mlm_probability = 0.15,
        **kwargs
    ):
        super().__init__(
            dataset=dataset,
            feature_keys=feature_keys,
            label_key="",
            mode=None,
            pretrained_emb=pretrained_emb,
        )
        self.label_key = ""
        self.mode = None
        self.embedding_dim = embedding_dim
        assert "feature_size" not in kwargs

        self.feat_tokenizers = feat_tokenizers
        self.embeddings = nn.ModuleDict()
        self.transformer = nn.ModuleDict()
        self.mlm_heads = nn.ModuleDict()
        self.mlm_probability = mlm_probability
        self.fc = None
        
        for feature_key in self.feature_keys:
            vocab_size = self.feat_tokenizers[feature_key].get_vocabulary_size()
            padding_idx = self.feat_tokenizers[feature_key].get_padding_index()
            self.embeddings[feature_key] = nn.Embedding(
                vocab_size, self.embedding_dim,
                padding_idx=padding_idx
                )
            self.transformer[feature_key] = TransformerLayer(
                feature_size=embedding_dim,
                **kwargs
                )
            self.mlm_heads[feature_key] = nn.Linear(self.embedding_dim, vocab_size)


    def mask_tokens(self, feature_key: str, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        tokenizer = self.feat_tokenizers[feature_key]
        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability, device=inputs.device)
        special_token_ids = [tokenizer.vocabulary(token) for token in ["<pad>", "<unk>", "<cls>"]]
        special_tokens_mask = torch.zeros(labels.shape, dtype=torch.bool, device=inputs.device)
        for st_id in special_token_ids:
            special_tokens_mask |= (inputs == st_id)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # Replace masked input tokens with '<mask>' token
        mask_token_id = tokenizer.vocabulary("<mask>")
        inputs[masked_indices] = mask_token_id

        return inputs, labels


    def switch_to_finetune(self, dataset, label_key, mode, dropout=0.1):
        assert mode in VALID_MODE, f"mode must be one of {VALID_MODE}"
        self.mode = mode
        self.dataset = dataset
        self.label_key = label_key        
        self.label_tokenizer = self.get_label_tokenizer()
        output_size = self.get_output_size(self.label_tokenizer)
        self.fc = nn.Sequential(
            nn.Linear(len(self.feature_keys) * self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.embedding_dim, output_size)
        )

        for name, param in self.named_parameters():
            if "fc" not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True


    def forward(self, **kwargs):
        is_pretrain = self.fc is None

        if is_pretrain:
            mlm_losses = []  
        else:
            patient_emb = []

        for feature_key in self.feature_keys:
            tokens = [
                [["<cls>"] + sorted(visit) for visit in patient_visits]
                for patient_visits in kwargs[feature_key]
            ]
            x = self.feat_tokenizers[feature_key].batch_encode_3d(tokens)
            x = torch.tensor(x, dtype=torch.long, device=self.device) # (batch_size, seq_len, event_len)

            batch_size, seq_len, event_len = x.shape

            # Create attention mask
            attention_mask = (x != self.feat_tokenizers[feature_key].get_padding_index())
            attention_mask &= (x != self.feat_tokenizers[feature_key].vocabulary("<unk>"))

            if is_pretrain:
                # Flatten x for masking
                x_flat = x.view(-1, event_len)  # (batch_size * seq_len, event_len)

                # Apply masking
                x_masked_flat, labels_flat = self.mask_tokens(feature_key, x_flat)
                x_masked = x_masked_flat.view(batch_size, seq_len, event_len)
                labels = labels_flat.view(batch_size, seq_len, event_len)
            else:
                x_masked = x

            # Embedding
            x_embedded = self.embeddings[feature_key](x_masked)  # (batch, seq_len, event_len, embedding_dim)

            # Reshape for transformer input
            x_embedded = x_embedded.view(batch_size, -1, self.embedding_dim)  # (batch, total_length, embedding_dim)
            attention_mask = attention_mask.view(batch_size, -1)  # (batch, total_length)
            
            # Transformer
            x_transformed = self.transformer[feature_key](x_embedded, attention_mask, kwargs.get('register_hook'))

            if is_pretrain:
                # MLM prediction
                prediction_scores = self.mlm_heads[feature_key](x_transformed)  # (batch, total_length, vocab_size)

                # Compute MLM loss
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                mlm_loss = loss_fct(
                    prediction_scores.view(-1, prediction_scores.size(-1)),
                    labels.view(-1)
                )
                mlm_losses.append(mlm_loss)
            else:
                cls_emb = x_transformed[:, 0, :]
                patient_emb.append(cls_emb)

        if is_pretrain:
            results = { "loss": sum(mlm_losses) / len(mlm_losses) }
        else:
            patient_emb = torch.cat(patient_emb, dim=1)
            logits = self.fc(patient_emb)  # (batch_size, num_classes)
            # obtain y_true, loss, y_prob
            y_true = self.prepare_labels(kwargs[self.label_key], self.label_tokenizer)
            loss = self.get_loss_function()(logits, y_true)
            y_prob = self.prepare_y_prob(logits)
            results = { "loss": loss, "y_prob": y_prob, "y_true": y_true }

        return results