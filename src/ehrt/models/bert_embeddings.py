import torch
from torch import nn
from einops import repeat
import numpy as np

from src.ehrt.graph_transformer import GraphTransformer


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, segment, age
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        #self.word_embeddings = nn.Linear(config.vocab_size, config.hidden_size)
        self.word_embeddings = GraphTransformer(config)
        self.type_embeddings = nn.Embedding(11, config.hidden_size//5, padding_idx=0)

        self.age_embeddings = nn.Embedding(config.age_vocab_size, config.hidden_size//5). \
            from_pretrained(embeddings=self._init_posi_embedding(config.age_vocab_size, config.hidden_size//5))

        self.time_embeddings = nn.Embedding(367, config.hidden_size//5). \
            from_pretrained(embeddings=self._init_posi_embedding(367, config.hidden_size//5))

        self.delta_embeddings = nn.Embedding(config.delta_size, config.hidden_size//5). \
            from_pretrained(embeddings=self._init_posi_embedding(config.delta_size, config.hidden_size//5))

        self.los_embeddings = nn.Embedding(1192, config.hidden_size//5). \
            from_pretrained(embeddings=self._init_posi_embedding(1192, config.hidden_size//5))

        self.posi_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size//5). \
            from_pretrained(embeddings=self._init_posi_embedding(config.max_position_embeddings, config.hidden_size//5))

        self.seq_layers = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU()
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size)
        self.acti = nn.GELU()
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))

    def forward(self, nodes, edge_index, edge_index_readout, edge_attr, batch, age_ids, time_ids, delta_ids, type_ids, posi_ids, los):
        word_embed = self.word_embeddings(nodes, edge_index, edge_index_readout, edge_attr, batch)
        type_embeddings = self.type_embeddings(type_ids)
        age_embed = self.age_embeddings(age_ids)
        los_embed = self.los_embeddings(los)

        time_embeddings = self.time_embeddings(time_ids)
        delta_embeddings = self.delta_embeddings(delta_ids)
        posi_embeddings = self.posi_embeddings(posi_ids)


        word_embed = torch.reshape(word_embed, type_embeddings.shape)
        embeddings = torch.cat((word_embed, type_embeddings, posi_embeddings, age_embed, time_embeddings), dim=2)
        b, n, _ = embeddings.shape   

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        embeddings = self.seq_layers(embeddings)
        embeddings = self.LayerNorm(embeddings)

        return embeddings

    def _init_posi_embedding(self, max_position_embedding, hidden_size):
        def even_code(pos, idx):
            return np.sin(pos / (10000 ** (2 * idx / hidden_size)))

        def odd_code(pos, idx):
            return np.cos(pos / (10000 ** (2 * idx / hidden_size)))

        # initialize position embedding table
        lookup_table = np.zeros((max_position_embedding, hidden_size), dtype=np.float32)

        # reset table parameters with hard encoding
        # set even dimension
        for pos in range(max_position_embedding):
            for idx in np.arange(0, hidden_size, step=2):
                lookup_table[pos, idx] = even_code(pos, idx)
        # set odd dimension
        for pos in range(max_position_embedding):
            for idx in np.arange(1, hidden_size, step=2):
                lookup_table[pos, idx] = odd_code(pos, idx)

        return torch.tensor(lookup_table)