import torch
from torchtext import data

class DataLoader:
    def __init__(self,sents,batch_size=1,fix_length=500,init_token='<bos>',eos_token='<eos>',stop_words=''):
        self.batch_size = batch_size
        self.sents = sents
        self.fix_length = fix_length
        self.init_token = init_token
        self.eos_token = eos_token
        self.stop_words = stop_words
        self.TEXT = data.Field(sequential=True, init_token=self.init_token,eos_token=self.eos_token, stop_words=self.stop_words,
                               fix_length=self.fix_length)
        self.LABEL = data.Field(sequential=True, init_token=self.init_token,eos_token=eos_token, stop_words=self.stop_words,
                               fix_length=self.fix_length)
        self.fields = [("TEXT", self.TEXT), ("LABEL", self.LABEL)]
        self.examples = self.create_examples()
        self.dataset = data.Dataset(examples=self.examples, fields=self.fields)
        self.TEXT.build_vocab(self.dataset)
        self.LABEL.build_vocab(self.dataset)

    def create_examples(self):
        examples = []
        e = data.Example()
        for i in self.sents:
            e_ = e.fromlist(i, fields=self.fields)
            examples.append(e_)
        return examples

    def create_one_data(self,sent):
        word_tensor = torch.ones(1,self.fix_length,dtype=torch.long)
        word2idx = self.TEXT.vocab.stoi
        word_tensor[0,0] = word2idx[self.TEXT.init_token]
        for i in range(len(sent)):
            c = sent[i]
            if c not in word2idx:
                word_tensor[0,i+1] = word2idx[self.TEXT.unk_token]
            else:
                word_tensor[0,i+1] = word2idx[c]
        word_tensor = word_tensor.view(self.fix_length,1)
        return word_tensor

    def reverse_vocab(self,vocab):
        vacab_ = dict([val, key] for key, val in vocab.items())
        return vacab_

    def save_vocab(self,path):
        word2idx = self.TEXT.vocab.stoi
        label2idx = self.LABEL.vocab.stoi
        path1 = '../'+path+'/word2idx.txt'
        with open(path1, 'w', encoding='utf-8') as f:
            for i in word2idx:
                f.write(str(i))
                f.write(' ')
                f.write(str(word2idx[i]))
                f.write('\n')
        path2 = '../' + path + '/label2idx.txt'
        with open(path2, 'w', encoding='utf-8') as f:
            for i in label2idx:
                f.write(str(i))
                f.write(' ')
                f.write(str(label2idx[i]))
                f.write('\n')

    def get_dataset(self):
        return self.dataset

    def get_batch_iter(self,shuffle=True):
        ds_train, ds_test = self.get_dataset().split()
        train_iter, test_iter = data.BucketIterator.splits((ds_train, ds_test), batch_size=self.batch_size,
                                                           sort_key=lambda x: len(x.TEXT), shuffle=shuffle)
        return train_iter,test_iter