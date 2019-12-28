import jieba

class Segment_words:
    def __init__(self,train_file,test_file):
        self.train_file = train_file
        self.test_file = test_file
        self.loc_tag = ['B-LOC', 'I-LOC']
        self.org_tag = ['B-ORG', 'I-ORG']
        self.per_tag = ['B-PER', 'I-PER']
        self.train_sents = self.read_data(self.train_file)
        self.test_sents = self.read_data(self.test_file)
        self.sents = self.train_sents + self.test_sents

    def read_data(self,file):
        train_sents = []
        with open(file, 'r', encoding='utf-8') as f:
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
        training_data = []
        for i in range(len(train_sents)):
            sent = train_sents[i]
            words = []
            tags = []
            for w, t in sent:
                words.append(w)
                tags.append(t)
            training_data.append((words, tags))
        return training_data

    def char_index(self,list,char):
        id1 = [i for i, x in enumerate(list) if x == char]
        return id1

    def find_NER_seq(self,pos_B,pos_I):
        pos = pos_I+pos_B
        pos.sort()
        result = []
        for i in range(len(pos_B)-1):
            x = pos_B[i]
            y = pos_B[i+1]
            result.append(pos[pos.index(x):pos.index(y)])
        result.append(pos[pos.index(pos_B[-1]):])
        return result

    #识别数据集（train+test）所有的loc、per、org 返回字典
    def recognize_NER(self):
        loc_words = []
        per_words = []
        org_words = []
        for i in self.sents:
            list_0 = i[0]
            list_1 = i[1]
            if self.loc_tag[0] in list_1:
                pos_B = self.char_index(list_1, self.loc_tag[0])
                pos_I = self.char_index(list_1, self.loc_tag[1])
                re = self.find_NER_seq(pos_B=pos_B,pos_I=pos_I)
                for i in re:
                    if len(i) == 0:
                        continue
                    else:
                        loc_words.append(''.join(list_0[i[0]:i[-1]+1]))
            if self.per_tag[0] in list_1:
                pos_B = self.char_index(list_1, self.per_tag[0])
                pos_I = self.char_index(list_1, self.per_tag[1])
                re = self.find_NER_seq(pos_B=pos_B, pos_I=pos_I)
                for i in re:
                    if len(i) == 0:
                        continue
                    else:
                        per_words.append(''.join(list_0[i[0]:i[-1]+1]))

            if self.org_tag[0] in list_1:
                pos_B = self.char_index(list_1, self.org_tag[0])
                pos_I = self.char_index(list_1, self.org_tag[1])
                re = self.find_NER_seq(pos_B=pos_B, pos_I=pos_I)
                for i in re:
                    if len(i) == 0:
                        continue
                    else:
                        org_words.append(''.join(list_0[i[0]:i[-1]+1]))
        loc_words = list(set(loc_words))
        per_words = list(set(per_words))
        org_words = list(set(org_words))
        return {'loc':loc_words,'per':per_words,'org':org_words}

    def generate_user_dict(self, user_dict, ner_dic):
        with open(user_dict, 'w',encoding='utf-8') as f:
            loc = ner_dic['loc']
            per = ner_dic['per']
            org = ner_dic['org']
            for i in loc:
                f.write(str(i+'\n'))
            for i in per:
                f.write(str(i+'\n'))
            for i in org:
                f.write(str(i+'\n'))

    def segment_to_words(self,ner_dic):
        jieba.load_userdict('NER_dict.txt')
        loc = ner_dic['loc']
        per = ner_dic['per']
        org = ner_dic['org']
        sents = self.sents
        seg_sents = []
        for i in sents:
            s = ''.join(i[0])
            seg_words = jieba.cut(s)
            bio_sent = []
            seg_sent = []
            for j in seg_words:
                seg_sent.append(j)
                if j in loc:
                    bio_sent.append('LOC')
                elif j in per:
                    bio_sent.append('PER')
                elif j in org:
                    bio_sent.append('ORG')
                else:
                    bio_sent.append('O')
            seg_sents.append((seg_sent,bio_sent))
        return seg_sents

    def save_segment_words(self,seg_sents,words_file):
        with open(words_file,'w',encoding='utf-8') as f:
            for i in seg_sents:
                words = i[0]
                labels = i[1]
                if len(words) != len(labels):
                    print('error.')
                    return None
                else:
                    for j in range(len(words)):
                        f.write(words[j])
                        f.write(labels[j])
                        f.write(' ')
                    f.write('\n')

if __name__ == '__main__':
    s = Segment_words('MSRA/msra_train_bio','MSRA/msra_test_bio')
    ner_dic = s.recognize_NER()
    a = s.segment_to_words(ner_dic)
    s.save_segment_words(seg_sents=a,words_file='words.txt')