import utils.model_training as m_t
from char_unit.data_loader import Char_data_loader

##数据准备
d = Char_data_loader(batch_size=32,file1='../MSRA/msra_train_bio',file2='../MSRA/msra_test_bio').dl

# 参数设置
EPOCHS = m_t.EPOCHS
MAX_LENGTH = m_t.MAX_LENGTH
HIDDEN_DIM = m_t.HIDDEN_DIM
BATCH_SIZE = m_t.BATCH_SIZE
##模型使用
train_model = m_t.Model_training(epochs=EPOCHS, data_loader=d, max_length=MAX_LENGTH, hidden_dim=HIDDEN_DIM,
                                 batch_size=BATCH_SIZE)
# 训练模型
train_model.training(lr=0.01, saved_model_path='char_unit')
