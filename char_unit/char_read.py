class Char_read():
    def __init__(self,file):
        self.file = file
        self.sents = self.read_data()

    def read_data(self):
        sents = []
        with open(self.file,'r',encoding='utf-8') as f:
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
                    line = line.replace('\n','')
                    i,j = line.split('\t')
                    chars.append(i)
                    tags.append(j)
                    line = f.readline()
        return sents