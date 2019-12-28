import torch
import torch.nn.functional as F
from torch import optim
import jieba
from LSTM import LSTM
from word_read import Word_read
from data_loader import DataLoader

class Model_training():
    def __init__(self,epochs,max_length,hidden_dim,batch_size,model_state_dict='',data_number=0,file='words.txt'):
        self.epochs = epochs
        self.max_length = max_length
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.w = Word_read(file)
        if data_number == 0:
            self.dl = DataLoader(sents=self.w.sents, batch_size=self.batch_size, fix_length=self.max_length)
        else:
            self.dl = self.dl = DataLoader(sents=self.w.sents[0:data_number],batch_size=self.batch_size,fix_length=self.max_length)
        self.word2idx = self.dl.TEXT.vocab.stoi
        self.idx2word = self.dl.reverse_vocab(self.word2idx)
        self.label2idx = self.dl.LABEL.vocab.stoi
        self.idx2label = self.dl.reverse_vocab(self.label2idx)
        self.n_vocab = len(self.word2idx)
        self.n_cat = len(self.label2idx)
        self.model = LSTM(n_vocab=self.n_vocab,hidden_size=self.hidden_dim,n_cat=self.n_cat)
        if model_state_dict != '':
            print('load model state dict ...')
            self.model.load_state_dict(torch.load(model_state_dict))
        self.train_data,self.test_data = self.dl.get_batch_iter()

    def caculate(self,input):
        return self.model(input)

    def training(self,lr,data='',saved_model_name='lstm.model'):
        if data == '':
            data = self.train_data
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        # loss_function = torch.nn.NLLLoss()
        loss_function = torch.nn.CrossEntropyLoss()
        for epoch in range(self.epochs):
            it = iter(data)
            running_loss = 0
            for step, batch in enumerate(it):
                text, label = batch.TEXT, batch.LABEL
                pre = self.caculate(text)
                batch_size = int(pre.size()[1])
                pre_ = pre.reshape(self.max_length * batch_size, self.n_cat)
                label_ = label.reshape(self.max_length * batch_size)
                loss = loss_function(pre_, label_)
                running_loss = running_loss + float(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if step % 100 == 0:
                    print('epoch:', epoch, 'step:', step, 'loss:', loss.data)
            print('epoch %s ,running_loss is %s' % (epoch, running_loss))
        torch.save(self.model.state_dict(),saved_model_name)

    def predict_single_sent(self,input_str):
        words = '.'.join(jieba.cut(input_str))
        words = words.split('.')
        word_tensor = self.dl.create_one_data(words)
        return words,self.caculate(word_tensor)

    def max_label_probability(self,input_tensor):
        #从model生成的张量中，计算出每个词的最大概率所在，并将最终的概率矩阵返回。
        seq_len = input_tensor.size()[0]
        batch_len = input_tensor.size()[1]
        prob_label = torch.max(input_tensor,dim=-1)
        prob_label = prob_label.indices
        return prob_label

    def generate_ner_tag(self,max_probability_tensor):
        #将最大概率转为NER词标记，整理成列表格式返回。
        tag_matrix = torch.t(max_probability_tensor)
        tag_matrix = tag_matrix.numpy().tolist()
        for i in range(len(tag_matrix)):
            for j in range(len(tag_matrix[i])):
                n = tag_matrix[i][j]
                tag_matrix[i][j] = self.idx2label[n]
        return tag_matrix

EPOCHS = 10
MAX_LENGTH = 300
HIDDEN_DIM = 150
BATCH_SIZE = 10
DATA_NUMBER = 0#使用多少条数据训练，如果设置为0表示使用全部数据
FILE = 'words.txt'

if __name__ == '__main__':
    # m_t = Model_training(EPOCHS, MAX_LENGTH, HIDDEN_DIM, BATCH_SIZE, data_number=DATA_NUMBER)
    m_t = Model_training(EPOCHS, MAX_LENGTH, HIDDEN_DIM, BATCH_SIZE, data_number=DATA_NUMBER,model_state_dict='lstm.model')
    # m_t.training(lr=0.01)
    words,x = m_t.predict_single_sent('人民日报发表声明，称昨日民航总医院一名医生遇害，国家表示非常遗憾，并予以强烈谴责')
    y = m_t.max_label_probability(x)
    y_ = m_t.generate_ner_tag(y)

