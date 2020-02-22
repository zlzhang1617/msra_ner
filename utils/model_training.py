import torch
from torch import optim
import jieba
from LSTM import LSTM

class Model_training():
    def __init__(self,data_loader,epochs,max_length,hidden_dim,batch_size,model_state_dict='',data_number=0):
        self.epochs = epochs
        self.max_length = max_length
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.dl = data_loader
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

    def training(self,lr,saved_model_path='',data='',saved_model_name='lstm.model'):
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
        if saved_model_path != '':
            path = saved_model_path+'/'+saved_model_name
        else:
            path = saved_model_name
        torch.save(self.model.state_dict(),path)

    def predict_single_sent(self,input_str):
        words = input_str
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
MAX_LENGTH = 500
HIDDEN_DIM = 150
BATCH_SIZE = 10