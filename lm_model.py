#! -*- coding:utf-8 -*-

import re, os
from tqdm import tqdm
import numpy as np
import pyhanlp
import json


def tokenize(s):
    return [i.word for i in pyhanlp.HanLP.segment(s)]


min_count = 10
word_size = 128
batch_size = 64
maxlen = 32
num_levels = 16


def text_generator():
    with open('question.txt') as f:
        for t in f:
            yield t.strip().decode('utf-8')


if os.path.exists('onlstm_config.json'):
    words, id2word, word2id = json.load(open('onlstm_config.json'))
    id2word = {int(i):j for i,j in id2word.items()}
else:
    words = {}
    for s in tqdm(text_generator()):
        for w in tokenize(s):
            words[w] = words.get(w, 0) + 1
    words = {i:j for i,j in words.items() if j >= min_count}
    # 0: padding, 1: unk, 2: <start>, 3: <end>
    id2word = {i+4:j for i,j in enumerate(words)}
    word2id = {j:i for i,j in id2word.items()}
    json.dump([words, id2word, word2id], open('onlstm_config.json', 'w'))


def string2id(s, start_end=True):
    _ = [word2id.get(w, 1) for w in s]
    if start_end:
        return [2] + _ + [3]
    else:
        return _


def data_generator():
    X = []
    while True:
        for s in text_generator():
            s = tokenize(s)[:maxlen]
            x = string2id(s)
            X.append(x)
            if len(X) == batch_size:
                l = max([len(x) for x in X])
                X = [x+[0]*(l-len(x)) for x in X]
                yield np.array(X), None
                X = []


train_data = data_generator()


from keras.models import Model
from keras.layers import *
import keras.backend as K
from keras.callbacks import Callback
from on_lstm_keras import ONLSTM


x_in = Input(shape=(None,), dtype='int32') # 句子输入
x = x_in

x = Embedding(len(words)+4, word_size)(x)
x = Dropout(0.25)(x)
onlstms = []

for i in range(3):
    onlstm = ONLSTM(word_size, num_levels, return_sequences=True, dropconnect=0.25)
    onlstms.append(onlstm)
    x = onlstm(x)

x = Dense(len(words)+4, activation='softmax')(x)

x_mask = K.cast(K.greater(x_in[:, :-1], 0), 'float32')
loss = K.sum(K.sparse_categorical_crossentropy(x_in[:, 1:], x[:, :-1]) * x_mask) / K.sum(x_mask)

lm_model = Model(x_in, x)
lm_model.add_loss(loss)
lm_model.compile(optimizer='adam')


class Evaluate(Callback):
    def __init__(self):
        self.lowest = 1e10
    def on_epoch_end(self, epoch, logs=None):
        # 保存最优结果
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            lm_model.save_weights('./best_model.weights')


evaluator = Evaluate()

lm_model.fit_generator(train_data,
                       steps_per_epoch=1000,
                       epochs=200,
                       callbacks=[evaluator])


lm_f = K.function([x_in], [onlstms[0].distance])

import uniout


s = u'水是生命的源泉'
s = u'案件仍在进一步侦查当中'
s = u'这样的设想是比较理想的'
s = u'计算机的鼠标有什么比较特殊的用途呢'
s = u'爱真的需要勇气'
s = u'这些模版所描述的都是句法级的语言现象'
s = u'苹果的颜色是什么'
s = u'北京在哪里'
s = u'苹果有几种颜色'
s = u'苹果和香蕉的颜色都是什么'
s = u'北京是中国的首都'


def build_tree(depth, sen):
    """该函数直接复制自原作者代码
    """
    assert len(depth) == len(sen)
    if len(depth) == 1:
        parse_tree = sen[0]
    else:
        idx_max = np.argmax(depth)
        parse_tree = []
        if len(sen[:idx_max]) > 0:
            tree0 = build_tree(depth[:idx_max], sen[:idx_max])
            parse_tree.append(tree0)
        tree1 = sen[idx_max]
        if len(sen[idx_max + 1:]) > 0:
            tree2 = build_tree(depth[idx_max + 1:], sen[idx_max + 1:])
            tree1 = [tree1, tree2]
        if parse_tree == []:
            parse_tree = tree1
        else:
            parse_tree.append(tree1)
    return parse_tree


def parse_sent(s):
    s = tokenize(s)
    sid = np.array([string2id(s)[:-1]])
    sl = lm_f([sid])[0][0][1:]
    # 用json.dumps的indent功能，最简单地可视化效果
    return json.dumps(build_tree(sl, s), indent=4, ensure_ascii=False)
