#!/usr/bin/env python
# encoding: utf-8

from __future__ import absolute_import
from __future__ import print_function

from itertools import chain

import numpy as np
import tensorflow as tf
from six.moves import range, reduce
from sklearn import cross_validation, metrics

from memn2n_kv.data_process import load_task, vectorize_data
from memn2n_kv.memn2n_kv import MemN2N_KV

tf.flags.DEFINE_float("learning_rate", 0.01, "Learning rate for Adam Optimizer.")
tf.flags.DEFINE_float("epsilon", 1e-8, "Epsilon value for Adam Optimizer.")
tf.flags.DEFINE_float("max_grad_norm", 40.0, "Clip gradients to this norm.")
tf.flags.DEFINE_integer("evaluation_interval", 50, "Evaluate and print results every x epochs")
tf.flags.DEFINE_integer("batch_size", 32, "Batch size for training.")
tf.flags.DEFINE_integer("hops", 3, "Number of hops in the Memory Network.")
tf.flags.DEFINE_integer("epochs", 200, "Number of epochs to train for.")
tf.flags.DEFINE_integer("embedding_size", 20, "Embedding size for embedding matrices.")
tf.flags.DEFINE_integer("memory_size", 50, "Maximum size of memory.")
tf.flags.DEFINE_integer("task_id", 1, "bAbI task id, 1 <= id <= 20")
tf.flags.DEFINE_integer("random_state", None, "Random state.")
tf.flags.DEFINE_string("data_dir", "data/tasks_1-20_v1-2/en/", "Directory containing bAbI tasks")
tf.flags.DEFINE_integer("feature_size", 40, "Feature size")
tf.flags.DEFINE_string("output", "tasks_scores.csv", "Name of output file.")
tf.flags.DEFINE_string("window_size", 5, "Window_size.")
FLAGS = tf.flags.FLAGS

print("Started Task:", FLAGS.task_id)

# task data
train, test = load_task(FLAGS.data_dir, FLAGS.task_id)  # 返回的是set组成的list，每一个元素的形式是(substory, q, a)
data = train + test    #[([['mary', 'moved', 'to', 'the', 'bathroom'], ['john', 'went', 'to', 'the', 'hallway']], ['where', 'is', 'mary'], ['bathroom']),
                       # ([['mary', 'moved', 'to', 'the', 'bathroom'], ['john', 'went', 'to', 'the', 'hallway'], ['daniel', 'went', 'back', 'to', 'the', 'hallway'], ['sandra', 'moved', 'to', 'the', 'garden']], ['where', 'is', 'daniel'], ['hallway']),
                       # ([[``````)

vocab = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q + a) for s, q, a in data)))  # 把data里面所有的单词按照字母降序的顺序排列  vocab：['back', 'bathroom', 'daniel', 'garden', 'hallway', 'is', 'john', 'mary', 'moved', 'sandra', 'the', 'to', 'went', 'where']
word_idx = dict((c, i + 1) for i, c in enumerate(vocab))  # data里面所有的word做成一个dict
                                                          #{'hallway': 5, 'bathroom': 2, 'garden': 4, 'is': 6, 'went': 13, 'moved': 9, 'back': 1, 'to': 12, 'daniel': 3, 'sandra': 10, 'the': 11, 'john': 7, 'where': 14, 'mary': 8}
                                                          #for i, c in enumerate(vocab):  print i, c      i是int型，c是str型
                                                          #0 back
                                                          #1 bathroom
                                                          #2 daniel
                                                          #3 garden
                                                          #4 ``````

max_story_size = max(map(len, (s for s, _, _ in data)))  #s是substory是一个由把前面所有的非query组成的list  [['mary', 'moved', 'to', 'the', 'bathroom'], ['john', 'went', 'to', 'the', 'hallway'],['daniel', 'went', 'back', 'to', 'the', 'hallway'], ['sandra', 'moved', 'to', 'the', 'garden']]
                                                         #求出最长的substory，返回的是一个数
                                                         # map(function, iterable, ...)  对可迭代函数'iterable'中的每一个元素应用‘function’方法，将结果作为list返回。
                                                         #>>> def add100(x): return x+100
                                                         # >>> hh = [11,22,33]    >>> map(add100,hh)
                                                         # [111, 122, 133]
mean_story_size = int(np.mean([len(s) for s, _, _ in data ]))  #所有的substory相加求平均数，[ len(s) for s, _, _ in data ]得到的是每个substory的长度组成的list[2,4,6,8,``````]
sentence_size = max(map(len, chain.from_iterable(s for s, _, _ in data)))  # map(len, chain.from_iterable(s for s, _, _ in data))：[5, 5, 5, 5, 6, 5]
query_size = max(map(len, (q for _, q, _ in data)))
memory_size = min(FLAGS.memory_size, max_story_size)
vocab_size = len(word_idx) + 1
window_size = min(query_size, FLAGS.window_size)

# train/validation/test sets
S, Q, A, C = vectorize_data(train, word_idx, window_size, memory_size)  # C: center    返回的都是list，参数：list，dict，num，num
trainS, valS, trainQ, valQ, trainA, valA, trainC, valC = cross_validation.train_test_split(S, Q, A, C, test_size=.1, random_state=FLAGS.random_state)  #返回的是list  #从样本中随机的按比例选取train data和test data。调用形式为：X_train, X_test, y_train, y_test = cross_validation.train_test_split(train_data, train_target, test_size=0.4, random_state=0)test_size是样本占比。如果是整数的话就是样本的数量。random_state是随机数的种子。不同的种子会造成不同的随机采样结果。相同的种子采样结果相同。
testS, testQ, testA, testC = vectorize_data(test, word_idx, window_size, memory_size)

# params
n_train = trainS.shape[0]  #第一维就是通过上一步取出的数量,也就是总的batch_size
n_test = testS.shape[0]
n_val = valS.shape[0]

train_labels = np.argmax(trainA, axis=1)
test_labels = np.argmax(testA, axis=1)
val_labels = np.argmax(valA, axis=1)

tf.set_random_seed(FLAGS.random_state)
batch_size = FLAGS.batch_size
optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, epsilon=FLAGS.epsilon)
batches = zip(range(0, n_train-batch_size, batch_size), range(batch_size, n_train, batch_size))  # 返回一个tuple:[(0, batch_size),(batch_size, 2*batch_size),```,(n_train-2*batch_size, n_train-batch_size) ]        # zip([0,batch_size,2*batch_size,````,n_train-2*batch_size],[batch_size, 2*batch_size,```,n_train-batch_size])
batches = [(start, end) for start, end in batches]

with tf.Session() as sess:
    model = MemN2N_KV(batch_size=batch_size, vocab_size=vocab_size, embedding_size=FLAGS.embedding_size,
                      memory_size=memory_size, window_size=window_size, feature_size=FLAGS.feature_size,
                      hops=FLAGS.hops, max_grad_norm=FLAGS.max_grad_norm, optimizer=optimizer, session=sess)

    for t in range(1, FLAGS.epochs+1):
        np.random.shuffle(batches)  # 将列表中的元素打乱
        total_cost = 0.0
        for start, end in batches:
            s = trainS[start:end]  # story(key-window)
            q = trainQ[start:end]  # query
            a = trainA[start:end]  # answer
            c = trainC[start:end]  # value(value-center of the window)
            cost_t = model.batch_fit(s, q, a, c)
            total_cost += cost_t

        if t % FLAGS.evaluation_interval == 0:  # 进行了一些epoch之后才进入下面的操作
            train_preds = []
            for start in range(0, n_train, batch_size):
                end = start + batch_size
                s = trainS[start:end]
                q = trainQ[start:end]
                c = trainC[start:end]
                pred = model.predict(s, q, c)
                train_preds += list(pred)

            val_preds = model.predict(valS, valQ, valC)
            train_acc = metrics.accuracy_score(np.array(train_preds), train_labels)  #？？两个参数的顺序反了#给预测的和真实的匹配程度打分  #sklearn.metrics.accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)
            val_acc = metrics.accuracy_score(val_preds, val_labels)

            print('-----------------------')
            print('Epoch', t)
            print('Total Cost:', total_cost)
            print('Training Accuracy:', train_acc)
            print('Validation Accuracy:', val_acc)
            print('-----------------------')


    test_preds = model.predict(testS, testQ, testC)
    test_acc = metrics.accuracy_score(test_preds, test_labels)
    print("Testing Accuracy:", test_acc)

    with open(FLAGS.output, 'a') as f:  # 'a': 追加新数据到已有文件
        f.write('{}, {}, {}, {}\n'.format(FLAGS.task_id, test_acc, train_acc, val_acc))
