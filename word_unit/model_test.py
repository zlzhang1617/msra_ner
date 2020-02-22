from utils import data_loader
from word_unit.word_read import Word_read
from utils import model_training as mt


w = Word_read('words.txt')
d = data_loader.DataLoader(sents=w.sents)
m_t = mt.Model_training(data_loader=d,epochs=mt.EPOCHS,max_length=mt.MAX_LENGTH,hidden_dim=mt.HIDDEN_DIM,batch_size=mt.BATCH_SIZE,model_state_dict='lstm.model')
m_t.predict_single_sent('中国真漂亮')