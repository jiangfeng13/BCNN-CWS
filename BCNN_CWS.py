"""
IDCNN(空洞CNN) 当卷积Conv1D的参数dilation_rate>1的时候，便是空洞CNN的操作
"""
import keras.layers
from keras.callbacks import EarlyStopping
from keras.models import Model
from keras.layers import Embedding, Dense, Dropout, Input, LSTM
from keras.layers import Conv1D
from keras_contrib.layers import CRF
from keras import backend as K

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6
set_session(tf.Session(config=config))
class IDCNNCRF(object):
    def __init__(self,
                 vocab_size: int,  # 词的数量(词表的大小)
                 n_class: int,  # 分类的类别(本demo中包括小类别定义了7个类别)
                 max_len: int = 100,  # 最长的句子最长长度
                 embedding_dim: int = 128,  # 词向量编码长度
                 drop_rate: float = 0.5,  # dropout比例
                 ):
        self.vocab_size = vocab_size
        self.n_class = n_class
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.drop_rate = drop_rate
        pass

    def creat_model(self):
        """
        本网络的机构采用的是，
           Embedding
           直接进行2个常规一维卷积操作
           接上一个空洞卷积操作
           连接全连接层
           最后连接CRF层

        kernel_size 采用2、3、4

        cnn  特征层数: 64、128、128
        """

        inputs = Input(shape=(self.max_len,))
        inputs1 = Input(shape=(self.max_len,))
        x = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim)

        x10 = x(inputs)#原始
        x20 = x(inputs1)#分词

        xx1 = keras.layers.Lambda(lambda x: K.reverse(x, axes=2))(x10)

        xx2 = keras.layers.Lambda(lambda x: K.reverse(x, axes=2))(x20)

        x1 = Conv1D(filters=128,
                    kernel_size=3,
                    activation='relu',
                    padding='causal',
                    dilation_rate=1)(x10)
        xx1 = Conv1D(filters=128,
                    kernel_size=3,
                    activation='relu',
                    padding='causal',
                    dilation_rate=1)(xx1)

        xx2 = Conv1D(filters=128,
                     kernel_size=3,
                     activation='relu',
                     padding='causal',
                     dilation_rate=1)(xx2)
        x2 = Conv1D(filters=128,
                     kernel_size=3,
                     activation='relu',
                     padding='causal',
                     dilation_rate=1)(x20)

        x1 = Dropout(0.5)(x1)
        xx1 = Dropout(0.5)(xx1)
        x2 = Dropout(0.5)(x2)
        xx2 = Dropout(0.5)(xx2)

        '''----------'''

        x1 = Conv1D(filters=128,
                    kernel_size=3,
                    activation='relu',
                    padding='causal',
                    dilation_rate=1)(x1)
        xx1 = Conv1D(filters=128,
                     kernel_size=3,
                     activation='relu',
                     padding='causal',
                     dilation_rate=2)(xx1)

        xx2 = Conv1D(filters=128,
                     kernel_size=3,
                     activation='relu',
                     padding='causal',
                     dilation_rate=2)(xx2)
        x2 = Conv1D(filters=128,
                    kernel_size=3,
                    activation='relu',
                    padding='causal',
                    dilation_rate=2)(x2)

        x1 = Dropout(0.5)(x1)
        xx1 = Dropout(0.5)(xx1)
        x2 = Dropout(0.5)(x2)
        xx2 = Dropout(0.5)(xx2)
        '''--------------'''

        x1 = Conv1D(filters=128,
                    kernel_size=3,
                    activation='relu',
                    padding='same',
                    dilation_rate=4)(x1)
        xx1 = Conv1D(filters=128,
                     kernel_size=3,
                     activation='relu',
                     padding='same',
                     dilation_rate=4)(xx1)

        xx2 = Conv1D(filters=128,
                     kernel_size=3,
                     activation='relu',
                     padding='same',
                     dilation_rate=4)(xx2)
        x2 = Conv1D(filters=128,
                    kernel_size=3,
                    activation='relu',
                    padding='same',
                    dilation_rate=4)(x2)

        xx1 = keras.layers.Lambda(lambda x: K.reverse(x, axes=2))(xx1)

        xx2 = keras.layers.Lambda(lambda x: K.reverse(x, axes=2))(xx2)

        x1 = keras.layers.Add()([x1, xx1])
        x2 = keras.layers.Add()([x2, xx2])

        x1 = keras.layers.Concatenate()([x1, x10])
        # x1 = keras.layers.Concatenate()([x1, x2])
        x2 = keras.layers.Concatenate()([x2, x20])



        x1 = Dropout(self.drop_rate)(x1)
        x2 = Dropout(self.drop_rate)(x2)
        x1 = Dense(self.n_class)(x1)
        x2 = Dense(5)(x2)
        self.crf = CRF(self.n_class, sparse_target=False)
        x1 = self.crf(x1)
        self.crf1 = CRF(5, sparse_target=False)
        x2 = self.crf1(x2)
        self.model = Model(inputs=[inputs,inputs1], outputs=[x1,x2])
        self.model.summary()
        self.compile()
        return self.model

    def compile(self):
        self.model.compile('adam',
                           loss=[self.crf.loss_function,self.crf1.loss_function],
                           metrics=[self.crf.accuracy,self.crf1.accuracy],loss_weights=[1,1])


if __name__ == '__main__':
    import sys
    from datetime import datetime as dt
    from args import read_options

    sys.path.append('../')
    ts = dt.now()

    args = read_options()


    from DataProcess.process_data import DataProcess
    from sklearn.metrics import f1_score, precision_score, recall_score,accuracy_score
    import numpy as np
    from keras.utils.vis_utils import plot_model

    dataname = str(args.data)
    dp = DataProcess(max_len=100, data_type=dataname)
    train_data, train_label, test_data, test_label = dp.get_data(one_hot=True)

    '''
        分词数据集处理
        '''
    import json
    def load(datapath):
        f = open(datapath, 'r')
        data = json.load(f)
        return data['dataset'], data['labels'], dict(data['word_index'])

    cws_data, cws_labels, wordvocab = load(r'PDdata.json')
    cws_data = np.array(cws_data)
    cws_labels = np.array(cws_labels)
    #one-hot 处理
    def label_to_one_hot(index: []) -> []:
        data = []
        for line in index:
            data_line = []
            for i, index in enumerate(line):
                line_line = [0] * 5
                line_line[index] = 1
                data_line.append(line_line)
            data.append(data_line)
        return np.array(data)


    cws_labels = label_to_one_hot(index=cws_labels)

    model_class = IDCNNCRF(vocab_size=dp.vocab_size, n_class=7, max_len=100)
    model_class.creat_model()
    model = model_class.model
    cws_data = cws_data[0:37092]
    cws_labels = cws_labels[0:37092]

    train_data = train_data[0:]
    train_label = train_label[0:]
    lens = len(train_data)
    cws_datas = cws_data[0:lens]
    cws_labelss =  cws_labels[0:lens]
    lenss = len(test_data)

    if dataname == 'msra':
        model.fit([train_data, cws_datas], [train_label, cws_labelss], batch_size=64,
              epochs=15,
              validation_data=[[test_data, cws_data[0:9273]], [test_label, cws_labels[0:9273]]])
    if dataname == 'renmin':
        model.fit([train_data,cws_datas], [train_label,cws_labelss], batch_size=64, epochs=15,
              validation_data=[[test_data,cws_data[lenss:4613]], [test_label,cws_labels[0:4613]]])
    if dataname == 'onte':
        model.fit([train_data, cws_datas], [train_label, cws_labelss], batch_size=64, epochs=15,
                  validation_data=[[test_data, cws_data[0:4346]], [test_label, cws_labels[0:4346]]])


    # 对比测试数据的tag
    y = model.predict([test_data,cws_data])[0]

    label_indexs = []
    pridict_indexs = []

    num2tag = dp.num2tag()
    i2w = dp.i2w()
    texts = []
    texts.append(f"字符\t预测tag\t原tag\n")
    for i, x_line in enumerate(test_data):
        for j, index in enumerate(x_line):
            if index != 0:
                char = i2w.get(index, ' ')
                t_line = y[i]
                t_index = np.argmax(t_line[j])
                tag = num2tag.get(t_index, 'O')
                pridict_indexs.append(t_index)

                t_line = test_label[i]
                t_index = np.argmax(t_line[j])
                org_tag = num2tag.get(t_index, 'O')
                label_indexs.append(t_index)

                texts.append(f"{char}\t{tag}\t{org_tag}\n")
        texts.append('\n')
    names = r'log/BCNN-CWS' + dataname + str(lens) + '.txt'

    log = open(names, 'w', encoding='utf-8')
    for i in texts:
        log.write(i)
    log.close()
    f1score = f1_score(label_indexs, pridict_indexs, average='macro')
    print(f"f1score:{f1score}")
    f1score = f1_score(label_indexs, pridict_indexs, average='macro')
    recalls = recall_score(label_indexs, pridict_indexs, average='macro')
    precision = precision_score(label_indexs, pridict_indexs, average='macro')
    accuracy = accuracy_score(label_indexs, pridict_indexs)
    accuracy = str(accuracy)
    print(f"f1score:{f1score}")
    print(f"recall:{recalls}")
    print(f"pre :{precision}")
    print(f"acc :{accuracy}")
    te = dt.now()
    spent = te - ts
    print('Time spend : %s' % spent)

    with open('log/all_joint_logs.txt','a',encoding='utf-8') as f:
        f.write('BCNN-CWS ')
        f.write('\n')
        f.write(dataname)
        f.write('\n')
        f.write(str(lens))
        f.write('\n')
        f.write('f1: '+ str(f1score))
        f.write('\n')
        f.write('pre: '+ str(precision))
        f.write('\n')
        f.write('recall: '+str(recalls))
        f.write('\n')
        f.write('acc: '+str(accuracy))
        f.write('\n')











