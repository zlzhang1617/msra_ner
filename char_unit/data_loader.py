from char_unit.char_read import Char_read
from utils.data_loader import DataLoader

class Char_data_loader():
    def __init__(self,batch_size,**kwargs):
        self._batch_size = batch_size
        self._sents = []
        for i in kwargs:
            self._sents = self._sents + Char_read(kwargs[i]).sents
        self.dl = DataLoader(batch_size=self._batch_size,sents=self._sents)

if __name__ == '__main__':
    d = Char_data_loader(batch_size=32,file1='../MSRA/msra_train_bio',file2='../MSRA/msra_test_bio').dl