import torch
import torch.nn as nn

class Word_read():
    def __init__(self,word_file):
        self.word_file = word_file
        self.sents = self.read_data()

    def split_ner(self,word_ner):
        if word_ner[-1] == 'O':
            return [word_ner[0:-1],word_ner[-1]]
        else:
            return [word_ner[0:-3],word_ner[-3:]]

    def read_data(self):
        sents = []
        with open(self.word_file,'r',encoding='utf-8') as f:
            line = f.readline()
            while line:
                words = []
                tags = []
                line = line.replace('\n','')
                seg = line.split(' ')
                seg.pop()
                for i in seg:
                    w,t = self.split_ner(i)
                    words.append(w)
                    tags.append(t)
                words = tuple(words)
                tags = tuple(tags)
                sents.append((words,tags))
                line = f.readline()
        return sents
