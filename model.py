from transformers import BertModel
import torch
from torch import nn

class NERModel(nn.Module):
    def __init__(self):
        super(NERModel, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased") #bet
        for param in self.bert.parameters():
            param.requires_grad = False
        self.dropout = nn.Dropout(0.1)
        self.bilstm = nn.LSTM(input_size=768, hidden_size=100, num_layers=2, batch_first=True, bidirectional=True) #bilstn
        self.linear_layer = nn.Linear(200, 1)  # 100 * 2 (bidirectional) linear
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)

        # Pass the output through the BiLSTM layer
        lstm_output, _ = self.bilstm(sequence_output)

        # Apply the linear layer to get NER label logits
        logits = self.linear_layer(lstm_output)
        logits = self.sigmoid(logits)

        return logits
    
class MetaLearner(nn.Module):
    def __init__(self):
        super(MetaLearner, self).__init__()
        self.model = NERModel()
    
    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)
