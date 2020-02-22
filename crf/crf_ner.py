import pycrfsuite

class CRF_NER:
    def __init__(self,data_file='',model_path=''):
        self.data_file = data_file
        self.model_path = model_path

    def read_data(self,data_file):
        # 读取训练集
        train_sents = []
        with open(data_file, 'r', encoding='utf-8') as f:
            l = f.readline()
            sent = []
            while l:
                l_n = l.replace('\n', '')
                words = l_n.split('\t')
                sent.append(tuple(words))
                if l == '\n':
                    train_sents.append(sent)
                    sent = []
                l = f.readline()
        for i in range(len(train_sents)):
            train_sents[i].pop()
        return train_sents

    # 特征建立
    def word2features(self,sent, i):  # sent是一个完整句子集合
        word = sent[i][0]
        features = [
            'bias',
            'word=' + word,
        ]
        if i > 0:
            word1 = sent[i - 1][0]
            features.extend([
                '-1.word=' + word1,
            ])
            if i > 1:
                word2 = sent[i - 2][0]
                features.extend([
                    '-2.word=' + word2
                ])
        else:
            features.append('BOS')

        if i < (len(sent) - 1):
            word1 = sent[i + 1][0]
            features.extend([
                '+1:word=' + word1,
            ])
            if i < (len(sent) - 2):
                word2 = sent[i + 2][0]
                features.extend([
                    '+2:word=' + word2
                ])
        else:
            features.append('EOS')

        return features

    def sent2features(self,sent):
        return [self.word2features(sent, i) for i in range(len(sent))]

    def sent2labels(self,sent):
        return [label for token, label in sent]

    def sent2tokens(self,sent):
        return [token for token, label in sent]

    def crf_trainer(self):
        train_sents = self.read_data(self.data_file)
        X_train = [self.sent2features(s) for s in train_sents]
        Y_train = [self.sent2labels(s) for s in train_sents]
        trainer = pycrfsuite.Trainer(verbose=False)
        trainer.set_params({
            'c1': 1.0,
            'c2': 1e-3,
            'max_iterations': 50,
            'feature.possible_transitions': True
        })
        for xseq, yseq in zip(X_train, Y_train):
            trainer.append(xseq, yseq)
        return trainer

    def train(self,trainer,saved_name):
        trainer.train(saved_name)

    def tagger(self):
        tagger = pycrfsuite.Tagger()
        if self.model_path == '':
            print('there is no crf model selected.')
            return None
        else:
            try:
                tagger.open(self.model_path)
                return tagger
            except BaseException:
                print('read model error.')
                return None

    def predict_single(self,tagger,sentence):
        X_ = self.sent2features(sentence)
        return tagger.tag(X_)

if __name__ == '__main__':
    ner = NER(data_file='./MSRA/msra_train_bio',model_path='ner_code.crf')
    tagger = ner.tagger()
    pre = ner.predict_single(tagger,sentence='剧烈运动后咯血,是怎么了?')
    # trainer = ner.crf_trainer()