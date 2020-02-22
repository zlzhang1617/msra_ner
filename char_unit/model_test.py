from char_unit.data_loader import Char_data_loader
import utils.model_training as m_t
import operator

d = Char_data_loader(batch_size=32,file1='../MSRA/msra_train_bio',file2='../MSRA/msra_test_bio').dl

# 参数设置
EPOCHS = m_t.EPOCHS
MAX_LENGTH = m_t.MAX_LENGTH
HIDDEN_DIM = m_t.HIDDEN_DIM
BATCH_SIZE = m_t.BATCH_SIZE
##模型使用
model = m_t.Model_training(epochs=EPOCHS, data_loader=d, max_length=MAX_LENGTH, hidden_dim=HIDDEN_DIM,
                                 batch_size=BATCH_SIZE,model_state_dict='lstm.model')

d = Char_data_loader(batch_size=32,file2='../MSRA/msra_test_bio').dl
d = d.sents
all = 0
right = 0

def tag_one_sent(str):
    word,tensor = model.predict_single_sent(str)
    max_val = model.max_label_probability(tensor)
    tag_seq = model.generate_ner_tag(max_val)
    return tag_seq[0][1:len(word)+1]

def test():
    all = 0
    right = 0
    for i, j in d:
        i = ''.join(i)
        j = list(j)
        if len(j) > 500:
            continue
        pre_tag = tag_one_sent(i)
        for k in range(len(j)):
            all  = all + 1
            if pre_tag[k] == j[k]:
                right = right + 1
    return right/all