class Prepare_data():
    def __init__(self,data_file):
        self.word2idx = {}
        self.label2idx = {'<bos>':0,'<eos>':1,'O':2,'B-LOC':3,'I-LOC':4,'B-PER':5,'I-PER':6
            ,'B-ORG':7,'I-ORG':8}
        self.data = self._from_file(data_file)
        for sentence, tags in self.data:
            for word in sentence:
                if word not in self.word2idx:
                    self.word2idx[word] = len(self.word2idx)

    def _from_file(self,file):
        sents = []
        with open(file,'r',encoding='utf-8') as f:
            line = f.readline()
            sent = []
            chars = []
            tags = []
            while line:
                if line == '\n':
                    sent.append(tuple(chars))
                    sent.append(tuple(tags))
                    sents.append(tuple(sent))
                    sent = []
                    chars = []
                    tags = []
                    line = f.readline()
                else:
                    line = line.replace('\n', '')
                    i, j = line.split('\t')
                    chars.append(i)
                    tags.append(j)
                    line = f.readline()
        return sents

if __name__ == '__main__':
    p = Prepare_data('../MSRA/msra_train_bio')