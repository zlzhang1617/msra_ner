import torch
import torch.nn as nn
import random

#将字符转为数字向量
def prepare_sequence(seq, to_ix):
    idxs = []
    for i in seq:
        if i in to_ix:
            idxs.append(to_ix[i])
        else:
            idxs.append(random.randint(0,len(to_ix)))
    # idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

class LSTM(nn.Module):
    def __init__(self,n_vocab,label2idx,embedding_dim,hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_cat = len(label2idx)
        self.e = nn.Embedding(n_vocab,hidden_dim)
        self.lstm = nn.LSTM(embedding_dim,hidden_dim // 2,
                            num_layers=1, bidirectional=True)
        self.fc2 = nn.Linear(hidden_dim,self.n_cat)
        self.hidden = self.init_hidden()

    def init_hidden(self,):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def forward(self,inp):
        e_out = self.e(inp).view(len(inp), 1, -1)
        lstm_o,_ = self.lstm(e_out,self.hidden)
        lstm_o = lstm_o.view(len(inp), self.hidden_dim)
        # fc = F.dropout(self.fc2(lstm_o),p=0.8)
        fc = self.fc2(lstm_o)
        return fc