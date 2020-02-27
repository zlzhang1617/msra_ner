import torch
from torch import optim
import LSTM_CRF as lc
from utils import prepare_data
from utils import test_performance

# 参数设置
EPOCHS = 5
HIDDEN_DIM = 150

##数据准备
pd = prepare_data.Prepare_data('../MSRA/msra_train_bio')
data = pd.data
word2idx = pd.word2idx
label2idx = pd.label2idx
idx2label = {value:key for key,value in label2idx.items()}

n_cat = len(word2idx)

model = lc.BiLSTM_CRF(n_cat, label2idx, HIDDEN_DIM, HIDDEN_DIM)
model.load_state_dict(torch.load('lstm_crf.model'))

def train(data):
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
    for epoch in range(EPOCHS):
        running_loss = 0
        step = 0
        iter_data = data
        for sentence, tags in iter_data:
            step = step + 1
            model.zero_grad()
            sentence_in = lc.prepare_sequence(sentence, word2idx)
            targets = list(tags)
            loss = model.neg_log_likelihood(sentence_in, targets)
            running_loss = running_loss + float(loss)
            loss.backward()
            optimizer.step()
            # print('step%d,loss%f:'%(step,float(loss)))
        print('epoch:',epoch)

def save_model(model_name):
    torch.save(model.state_dict(),model_name)

def predict(input_str):
    inp = lc.prepare_sequence(input_str,word2idx)#准备输入
    out = model(inp)
    out_seq = out[1]
    out_seq = [idx2label[t] for t in out_seq]
    return out_seq

def test(data):
    precision = 0
    recall = 0
    num = 0
    for sentence, gold_tags in data:
        num = num + 1
        pre = predict(sentence)
        precision_, recall_ = test_performance.compare2tags(sentence, gold_tags, pre)
        precision = precision + precision_
        recall = recall + recall_
    precision = precision / num
    recall = recall / num
    f1 = (2 * precision * recall) / (precision + recall)
    print(precision, recall, f1)
    return precision, recall, f1


if __name__ == '__main__':
    predict('去河南大学玩')