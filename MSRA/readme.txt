###########################################################
#                                                         #
#                                                         #
#	                  readme for MSRA                     #
#                                                         #
#                                                         #
###########################################################

语言：简体中文
编码：utf-8

msra_train_bio为训练集
msra_test_bio为测试集

包含专名：
	 ______________________
	|标签| LOC | ORG | PER |
	------------------------
	|含义|地名 |组织名|人名|
	------------------------

训练集：
	 _________________________________________________
	|  句数  |  字符数  |  LOC数  |  ORG数  |  PER数  |
	---------------------------------------------------
	|  45000 | 2171573  |  36860  |  20584  |  17615  |
	---------------------------------------------------

测试集：
	 _________________________________________________
	|  句数  |  字符数  |  LOC数  |  ORG数  |  PER数  |
	---------------------------------------------------
	|  3442  |  172601  |  2886   |  1331   |  1973   |
	---------------------------------------------------

标注格式：

	[字符]	[标签]	# 分隔符为"\t"

	其中标签采用BIO规则，即非专名为"O",专名首部字符为"B-[专名标签]"，专名中部字符为"I-[专名标签]"

	例如：

		历	B-LOC
		博	I-LOC
		、	O
		古	B-ORG
		研	I-ORG
		所	I-ORG