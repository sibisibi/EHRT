from torch import nn
import pytorch_pretrained_bert as Bert

from src.ehrt.models.bert_model import BertModel


class BertForMTR(Bert.modeling.BertPreTrainedModel):
    def __init__(self, config):
        super(BertForMTR, self).__init__(config)
        self.num_labels = 1
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        #self.gru = nn.GRU(config.hidden_size, config.hidden_size // 2, 1, batch_first = True, bidirectional=True)
        #self.gru = nn.Linear(config.hidden_size * 50, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.relu = nn.ReLU()
        self.apply(self.init_bert_weights)
    def forward(self, nodes, edge_index, edge_index_readout, edge_attr, batch, age_ids, time_ids, delta_ids, type_ids, posi_ids, attention_mask=None, labels=None, masks=None, los=None):
        _, pooled_output = self.bert(nodes, edge_index, edge_index_readout, edge_attr, batch, age_ids, time_ids, delta_ids, type_ids, posi_ids, attention_mask,los,
                                     output_all_encoded_layers=False)
        #pooled_output = self.dropout(pooled_output)
        #pooled_output = pooled_output * attention_mask.unsqueeze(-1)
        #pooled_output = torch.sum(pooled_output, axis=1) / torch.sum(attention_mask, axis=1).unsqueeze(-1)
        #pooled_output = torch.mean(_, axis=1)
        #pooled_output, x = self.gru(pooled_output)
        #pooled_output = self.gru(torch.flatten(pooled_output, start_dim=1))
        #pooled_output = self.relu(self.dropout(pooled_output))
        logits = self.classifier(pooled_output).squeeze(dim=1)
        bce_logits_loss = nn.BCEWithLogitsLoss(reduction='mean')
        discr_supervised_loss = bce_logits_loss(logits, labels)

        return discr_supervised_loss, logits
