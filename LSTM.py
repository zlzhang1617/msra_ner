import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self,n_vocab,hidden_size,n_cat,bs=1,nl=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.bs = bs
        self.nl = nl
        self.e = nn.Embedding(n_vocab,hidden_size)
        self.lstm = nn.LSTM(hidden_size,hidden_size,nl)
        self.fc2 = nn.Linear(hidden_size,n_cat)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,inp):
        bs = inp.size()[1]
        if bs != self.bs:
            self.bs = bs
        e_out = self.e(inp)
        h0 = c0 = Variable(e_out.data.new(*(self.nl,self.bs,self.hidden_size)).zero_())
        lstm_o,_ = self.lstm(e_out,(h0,c0))
        # fc = F.dropout(self.fc2(lstm_o),p=0.8)
        fc = self.fc2(lstm_o)
        # return self.softmax(fc)
        return fc