import operator
import model_training as m
from segment_words import Segment_words

m_t = m.Model_training(epochs=m.EPOCHS,max_length=m.MAX_LENGTH,hidden_dim=m.HIDDEN_DIM,batch_size=m.BATCH_SIZE,model_state_dict='lstm.model')

# words,x = m_t.predict_single_sent('去苏州大学玩')
# y = m_t.max_label_probability(x)
# y_= m_t.generate_ner_tag(y)

s = Segment_words('./MSRA/msra_train_bio','./MSRA/msra_test_bio')
s.sents = s.test_sents
ner_dic = s.recognize_NER()
seg_sents = s.segment_to_words(ner_dic)

with open('precision_result.txt','r',encoding='utf-8') as f:
    line = f.readline()
    labels = []
    precisions = []
    while line:
        line = line.replace('\n', '')
        c = line[0]
        if c == 't':
            line = f.readline()
            continue
        elif c == 'l':
            line = line[6:]
            label = line.split(' ')
            labels.append(label)
        elif c=='p':
            line = line[5:]
            pre = line.split(' ')
            precisions.append(pre)
        line = f.readline()

all = 0
right = 0
for i in range(len(labels)):
    label = labels[i]
    pre = precisions[i]
    for j in range(len(label)):
        l = label[j]
        p = pre[j]
        all = all + 1
        if l == p:
            right = right + 1
print(right/all)


def save_precision_result():
    with open('precision_result.txt', 'w', encoding='utf-8') as f:
        for i in seg_sents:
            text = i[0]
            label = i[1]
            if len(text) >= 300: continue
            word_tensor = m_t.dl.create_one_data(text)
            pre = m_t.caculate(word_tensor)
            y = m_t.max_label_probability(pre)
            y_ = m_t.generate_ner_tag(y)
            y_ = y_[0]
            f.write('text:')
            f.write(' '.join(text))
            f.write('\n')
            f.write('label:')
            f.write(' '.join(label))
            f.write('\n')
            f.write('pre:')
            f.write(' '.join(y_[1:(len(text) + 1)]))
            f.write('\n')



