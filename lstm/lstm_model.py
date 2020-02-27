import torch.nn as nn
import torch
from torch import optim
import LSTM as lstm
from utils import prepare_data
from utils import test_performance as tp

# 参数设置
EPOCHS = 10
HIDDEN_DIM = 150
EMBEDDING_DIM = 150

##数据准备
pd = prepare_data.Prepare_data('../MSRA/msra_train_bio')
data = pd.data
word2idx = pd.word2idx
label2idx = pd.label2idx
idx2label = {value:key for key,value in label2idx.items()}

n_vocab = len(word2idx)
model = lstm.LSTM(n_vocab,label2idx,EMBEDDING_DIM,HIDDEN_DIM)
model.load_state_dict(torch.load('lstm.model'))

def train(data):
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=0.01)
    for epoch in range(EPOCHS):
        running_loss = 0
        step = 0
        iter_data = data
        for sentence, tags in iter_data:
            step = step + 1
            model.zero_grad()
            sentence_in = lstm.prepare_sequence(sentence, word2idx)
            targets = lstm.prepare_sequence(tags,label2idx)
            pre = model(sentence_in)
            loss = loss_func(pre,targets)
            running_loss = running_loss + float(loss)
            loss.backward()
            optimizer.step()
            if step % 100 == 0:
                print('loss',float(loss))
        print('epoch:',epoch,end='')
        print('running loss is:',running_loss)

def predict(input_str):
    inp = lstm.prepare_sequence(input_str,word2idx)
    out = model(inp)
    values,indices = torch.max(out,dim=-1)
    indices = indices.tolist()
    return [idx2label[i] for i in indices]

def test(data):
    precision = 0
    recall = 0
    num = 0
    for sentence, gold_tags in data:
        num = num + 1
        pre = predict(sentence)
        precision_, recall_ = tp.compare2tags(sentence, gold_tags, pre)
        precision = precision + precision_
        recall = recall + recall_
    precision = precision / num
    recall = recall / num
    f1 = (2 * precision * recall) / (precision + recall)
    print(precision, recall, f1)
    return precision, recall, f1


if __name__ == '__main__':
    # data = prepare_data.Prepare_data('../MSRA/msra_test_bio').data
    test(data)
