# msra_ner
LSTM网络在msra数据集上实现命名实体识别

一、数据准备
该模型基于MSRA数据集训练，数据存放在‘MSRA’目录下。MSRA数据集存放的数据格式为：
当	O
希	O
望	O
工	O
程	O
……

我训练的LSTM模型基于词单位而不是字符单位，因此在准备数据过程中有以下几步：
  1、识别出数据集中的每个句子和对应标记；
  2、识别每个句子中的NER，将NER存放在jieba分词用户字典中，确保分词过程中不会将同一个命名实体切分开来；
  3、用jieba分词，合并tag标记。例如：北LOC-B 京LOC-I ---> 北京LOC；
  4、生成text和label的词汇表，词汇表中每个词对应一个编号，例如：北京：40；
  5、整理数据，设置每个句子最大长度，不足的地方用‘<pad>’填充，句子开始添加‘<bos>’标记。

整理过的数据格式存放在‘words.txt’文件，文件内容形如：
首先O ，O 我O 对O 东盟LOC 成立O 三十周年O ，O 表示O 热烈O 的O 祝贺O ！O 
我O 相信O ，O 这次O 会晤O 将O 标志O 着O 中国LOC 与O 东盟LOC 关系O 进入O 一个O 新LOC 的O 发展O 阶段O 。O 
……
通过torchtext工具生成一个个batch，每个batch包含TEXT和LABEL两部分，作为LSTM网络的输入。
batch.TEXT:
[[w11 w12 w13 ... w1n]
[w21 w22 w23 ... w2n]
.
.
[wm1 wm2 wm3 ... wmn]]
m：sequence_length
n:batch_size
wmn 表示第n个句子的第m个词
wmn 是词汇表的单词编号，而不是具体的单词

batch.LABEL
[[l11 l12 l13 ... 1n]
[l21 l22 l23 ... l2n]
.
.
[lm1 lm2 lm3 ... lmn]]
m：sequence_length
n:batch_size
lmn 表示第n个句子的第m个词的标记
lmn 一共有7种取值，分别是bos、unk、pad、O、ORG、LOC、PER

二、模型结构搭建
LSTM模型见‘LSTM.py’文件，在一个LSTM网络结构中，包含：Embedding层-->LSTM层-->Linear层
我的设置：Embedding = hidden，dim = 20，Linear = 7
LSTM接收batch.TEXT作为输入，通过Linear映射为7为输出(output=7)，因此LSTM的输出大小为（seq_length,batch_size,output）
输出格式举例：
[[[o111 o112 o113 ... o117]
[o121 o122 o123 ... o127]
[o131 o132 o133 ... o137]
.
.
[o1n1 o1n2 o1n3 ... on7]]

[[o211 o212 o213 ... o217]
[o221 o222 o223 ... o227]
[o131 o132 o133 ... o137]
.
.
[o2n1 o2n2 o2n3 ... o2n7]]

...
...
...

[[om11 om12 om13 ... om17]
[om21 om22 om23 ... om27]
[om31 om32 om33 ... om37]
.
.
[omn1 omn2 omn3 ... omn7]]]

m:sequence_length n:batch_size

三、模型训练
NER问题本质上是一个分类问题，因此选用交叉熵损失函数：torch.nn.torch.nn.CrossEntropyLoss()
CrossEntropyLoss函数将logsoftmax，nllloss步骤合并在了一起，不需要自行添加logsoftmax步骤。
训练参数设置：
EPOCHS = 10#训练10轮次
MAX_LENGTH = 300#句子填充为300长度
HIDDEN_DIM = 150#embedding_dim = hidden_dim = 150
BATCH_SIZE = 10#每一批中包含10条完整数据
DATA_NUMBER = 0#默认使用全部数据

模型训练代码：
m_t = Model_training(EPOCHS, MAX_LENGTH, HIDDEN_DIM, BATCH_SIZE, data_number=DATA_NUMBER)
m_t.training(lr=0.01)
模型训练结束后保存为‘lstm.model’文件。

使用训练好地模型代码为：
m_t = Model_training(EPOCHS, MAX_LENGTH, HIDDEN_DIM, BATCH_SIZE, data_number=DATA_NUMBER,model_state_dict='lstm.model')
words,x = m_t.predict_single_sent('人民日报发表声明，称昨日民航总医院一名医生遇害，国家表示非常遗憾，并予以强烈谴责')
y = m_t.max_label_probability(x)
y_ = m_t.generate_ner_tag(y)

四、模型评估和问题分析
在‘msra_test_bio’数据集上，词标记的准确率约94%。
个人认为，这个模型在分词步骤上存在较大问题，分词步骤并没有一个特别标准的答案，即使在处理数据的label中也能发现个别不合理的现象存在，例如句子X：
“努力 地 学习” ，因为在某个例句中将词“地”标记为了ORG，jieba分词根据用户词典，默认将‘地’标记为了ORG。
这个问题个别存在，并不普遍，因此我在数据整理过程中手动地改善过数据，但不可否认地是，分词过程严重影响最终模型效果。

下一阶段地学习方向是：1、LSTM+CRF模型实现  2、用字符数据训练LSTM模型。
做到了以上两点，将结果相互比较，得出可靠结论。




