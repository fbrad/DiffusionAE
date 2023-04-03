import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import math
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, batch_first=False):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model).float() * (-math.log(10000.0) / d_model))
        pe += torch.sin(position * div_term)
        pe += torch.cos(position * div_term)
        if self.batch_first:
            pe = pe.unsqueeze(0)
        else: 
            pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, pos=0):
        if self.batch_first:
            x = x + self.pe[pos:pos+x.size(1), :]
        else:
            x = x + self.pe[pos:pos+x.size(0), :]
        return self.dropout(x)
