import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel


class BiEncoder(nn.Module):
    def __init__(self, model_name="bert-base-uncased", temperature=0.05):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.temperature = temperature

    def encode(self, input_ids, attention_mask, token_type_ids=None):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return F.normalize(output.last_hidden_state[:, 0], dim=-1)

    def forward(self, queries, passages):
        q = self.encode(**queries)      
        p = self.encode(**passages)     
                                        
        scores = q @ p.T / self.temperature
        labels = torch.arange(q.size(0), device=q.device)

        return F.cross_entropy(scores, labels)
