#!/usr/bin/env python
# encoding: utf-8
"""
The implementation of Key Value Memory Networks for Directly Reading Documents in Tensorflow.
Window Level representation： window——key, window center——value
"""
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
from six.moves import range


def position_encoding(sentence_size, embedding_size):
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size+1
    le = embedding_size+1
    for i in range(1, le):  #（1~le-1）
        for j in range(1, ls):
            encoding[i-1, j-1] = (i - (le-1)/2) * (j - (ls-1)/2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    return np.transpose(encoding)  # (sentence_size, embedding_size)

def zero_nil_slot(t, name=None):  # 参数t是一个梯度
    """
    Overwrites the nil_slot (first row) of the input Tensor with zeros.
    The nil_slot is a dummy slot and should not be trained and influence
    the training algorithm.
    """
    # we have found it helpful to add “dummy”memories to regularize TA. That is, at training time we can randomly add 10% of empty memoriesto the stories

    with tf.op_scope([t], name, "zero_nil_slot") as name:
        t = tf.convert_to_tensor(t, name="t")# shape of t [？,s]
        s = tf.shape(t)[1]  # shape完了之1后取第二个数，也就是第二维的维数，是一个数
        z = tf.zeros(tf.pack([1, s]))  # z:[1,s]，即[[s个0]]
        return tf.concat(0, [z, tf.slice(t, [1, 0], [-1, -1])], name=name)  # [[s个0],[t的后面几个],[],````]  相当于把t的第一个小list换成了0组成的

def add_gradient_noise(t, stddev=1e-3, name=None):
    """
    Adds gradient noise as described in http://arxiv.org/abs/1511.06807 [2].
    The input Tensor `t` should be a gradient.
    The output will be `t` + gaussian noise.
    0.001 was said to be a good fixed value for memory networks [2].
    """

    with tf.op_scope([t, stddev], name, "add_gradient_noise") as name:
        t = tf.convert_to_tensor(t, name="t")
        gn = tf.random_normal(tf.shape(t), stddev=stddev)# 产生一个跟t的shape一样的由正态分布随机数组成的tensor，因为shape一样才能相加
        return tf.add(t, gn, name=name)


class MemN2N_KV(object):
    def __init__(self, batch_size, vocab_size, embedding_size, memory_size,
        window_size=5,
        feature_size=30,  # d
        hops=3,
        max_grad_norm=40.0,
        initializer=tf.random_normal_initializer(stddev=0.1),
        optimizer=tf.train.AdamOptimizer(learning_rate=1e-2),  # learning rate = 0.01
        session=tf.Session(),
        name='KeyValueMemN2N'):

        self._window_size = window_size
        self._batch_size = batch_size
        self._vocab_size = vocab_size
        self._memory_size = memory_size
        self._embedding_size = embedding_size
        self._hops = hops
        self._name = name
        self._feature_size = feature_size
        self._init = initializer
        self._max_grad_norm = max_grad_norm
        self._opt = optimizer


        self._build_inputs()
        self._build_vars()
        o = self._inference(self._stories, self._queries, self._memory_value)  # [None, self._feature_size]   # q(H+1)的转置

        y_tmp = tf.matmul(self.A, self.W_memory, transpose_b=True)  #[feature_zise, vocab_size]   # ？？candidate answers  #此处，令B=A
        logits = tf.matmul(o, y_tmp)  #[None, vocab_size]
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, tf.cast(self._answers, tf.float32),name='cross_entropy')  # [None, vocab_size]    answers: [None, vocab_size]
        cross_entropy_sum = tf.reduce_sum(cross_entropy, name="cross_entropy_sum")  # a number

        # loss op
        loss_op = cross_entropy_sum
        grads_and_vars = self._opt.compute_gradients(loss_op)  # self._opt = optimizer  #d得到的是多组grad&var
        grads_and_vars = [(tf.clip_by_norm(g, self._max_grad_norm), v) for g,v in grads_and_vars]  #d对于上一步得到的gv中，v不变，g = (tf.clip_by_norm(g, self._max_grad_norm)  防止gradient explosion
        grads_and_vars = [(add_gradient_noise(g), v) for g,v in grads_and_vars]#d对于上一步得到的gv中，v不变，g = (add_gradient_noise(g), v)
        nil_grads_and_vars = []
        for g, v in grads_and_vars:
            if v.name in self._nil_vars:
                nil_grads_and_vars.append((zero_nil_slot(g), v))  #RN
            else:
                nil_grads_and_vars.append((g, v))
        train_op = self._opt.apply_gradients(nil_grads_and_vars, name="train_op")

        # predict ops
        predict_op = tf.argmax(logits, 1, name="predict_op")  # 返回的是概率最大的那个即为预测的，即为与query匹配程度最大的input
        predict_proba_op = tf.nn.softmax(logits, name="predict_proba_op")
        predict_log_proba_op = tf.log(predict_proba_op, name="predict_log_proba_op")  #自然对数

        # assign ops
        self.loss_op = loss_op
        self.predict_op = predict_op
        self.predict_proba_op = predict_proba_op
        self.predict_log_proba_op = predict_log_proba_op
        self.train_op = train_op

        init_op = tf.initialize_all_variables()
        self._sess = session
        self._sess.run(init_op)

    def _build_inputs(self):
        self._stories = tf.placeholder(tf.int32, [None, self._memory_size, self._window_size], name="stories")  # key——window
        self._queries = tf.placeholder(tf.int32, [None, self._window_size], name="queries")  # question
        self._answers = tf.placeholder(tf.int32, [None, self._vocab_size], name="answers")  # label
        self._memory_value = tf.placeholder(tf.int32, [None, self._memory_size])  # value——window center

    def _build_vars(self):
        with tf.variable_scope(self._name):
            nil_word_slot = tf.zeros([1, self._embedding_size])  # [1, self._embedding_size]
            W = tf.concat(0, [nil_word_slot, self._init([self._vocab_size - 1, self._embedding_size])])
            W_memory = tf.concat(0, [nil_word_slot, self._init([self._vocab_size - 1, self._embedding_size])])
            self.W = tf.Variable(W, name="W")  # W: [self._vocab_size , self._embedding_size]
            self.W_memory = tf.Variable(W_memory, name="W_memory")  # W_memory: [self._vocab_size , self._embedding_size]

            self.A = tf.Variable(self._init([self._feature_size, self._embedding_size]), name='A')  # d*D
            self.TK = tf.Variable(self._init([self._memory_size, self._embedding_size]), name='TK')
            self.TV = tf.Variable(self._init([self._memory_size, self._embedding_size]), name='TK')
        self._nil_vars = set([self.W.name, self.W_memory.name]) # set([u'W:0', u'W_memory:0'])

    def _inference(self, stories, queries, memory_value):  # stories ：[None, self._memory_size, self._window_size]
                                                           # queries ：[None, self._window_size]
                                                           # memory_value : [None, self._memory_size]
        with tf.variable_scope(self._name):
            #embed
            self.embedded_query = tf.nn.embedding_lookup(self.W, queries)  # [[None, self._window_size, self._embedding_size]
            self.embedded_mkeys = tf.nn.embedding_lookup(self.W_memory, stories)  # [None, self._memory_size, self._window_size, self._embedding_size]
            self.embedded_mvalues = tf.nn.embedding_lookup(self.W_memory, memory_value)  # [None, self._memory_size, self._embedding_size]

            q = tf.reduce_sum(self.embedded_query, 1)  # [None, embedding_size]
            mk = tf.reduce_sum(self.embedded_mkeys, 2)  # [None, self._memory_size, embedding_size]
            mv = self.embedded_mvalues  # [None, self._memory_size, embedding_size]

            # R  (update the query with R, d*d)
            R_list = []
            for i in range(self._hops):
                R = tf.Variable(self._init([self._feature_size, self._feature_size]), name='R{}'.format(i))  #R: d*d
                R_list.append(R)  # R_list = [R1, R2, R3]

            u = tf.matmul(self.A, q, transpose_b=True)  # [self._feature_size,None]   #AFai(x)   # A:[self._feature_size, self._embedding_size]  q:[None, embedding_size]
            u = [u]  # [1,self._feature_size,None]
            for i in range(self._hops):
                # Key addressing
                R = R_list[i]
                u_temp = u[-1]  # [self._feature_size,None]
                mk_temp = mk + self.TK  # [None, self._memory_size, embedding_size]   # mk: [None, self._memory_size, embedding_size]  TK:[self._memory_size, self._embedding_size]
                k_temp = tf.reshape(tf.transpose(mk_temp, [2, 0, 1]), [self._embedding_size, -1])  # [embedding_size, None*self._memory_size]
                a_k_temp = tf.matmul(self.A, k_temp)  # [self._feature_size, None*self._memory_size]  #AFai(k)
                a_k = tf.reshape(tf.transpose(a_k_temp), [-1, self._memory_size, self._feature_size])  # [None, self._memory_size, self._feature_size]
                u_expanded = tf.expand_dims(tf.transpose(u_temp), [1])  # [None,1,self._feature_size]
                dotted = tf.reduce_sum(a_k * u_expanded, 2)  # [None, self._memory_size]   # [None, self._memory_size, self._feature_size]*[None,1,d]    feature_size = d

                probs = tf.nn.softmax(dotted)  # [None, self._memory_size]
                probs_expand = tf.expand_dims(probs, -1)  # [None, self._memory_size, 1]

                # Value Reading
                mv_temp = mv + self.TV  # [None, self._memory_size, embedding_size]    # mvalues: [None, self._memory_size, embedding_size]      self.TV :[self._memory_size, self._embedding_size]
                v_temp = tf.reshape(tf.transpose(mv_temp, [2, 0, 1]), [self._embedding_size, -1])  # [embedding_size, None*self._memory_size]
                a_v_temp = tf.matmul(self.A, v_temp)  # [self._feature_size, None*self._memory_size]   #AFai(v)   A:[self._feature_size, self._embedding_size]
                a_v = tf.reshape(tf.transpose(a_v_temp), [-1, self._memory_size, self._feature_size])  # [None,self._memory_size, self._feature_size]
                o_k = tf.reduce_sum(probs_expand * a_v, 1)  # [None,self._feature_size]
                o_k = tf.transpose(o_k)  # [self._feature_size, None]  为了能和后面的R相乘所以需要转置一下o
                u_k = tf.matmul(R, u[-1] + o_k)  #u_k: [self._feature_size, None]   # R:[self._feature_size, self._feature_size]
                u.append(u_k)

            return tf.transpose(u[-1])  # [None, self._feature_size]

    def batch_fit(self, stories, queries, answers, memory_value):
        """Runs the training algorithm over the passed batch
        Args:
            stories: Tensor (None, memory_size, window_size)
            queries: Tensor (None, window_size)
            answers: Tensor (None, vocab_size)
            memory_value: Tensor (None, memory_size)
        Returns:
            loss: floating-point number, the loss computed for the batch
        """
        feed_dict = {self._stories: stories, self._queries: queries, self._answers: answers, self._memory_value: memory_value}
        loss, _ = self._sess.run([self.loss_op, self.train_op], feed_dict=feed_dict)  #通过train_op让loss最小
        return loss

    def predict(self, stories, queries, memory_value):
        """Predicts answers as one-hot encoding.
        Args:
            stories: Tensor (None, memory_size, window_size)
            queries: Tensor (None, window_size)
            memory_value: Tensor (None, memory_size)
        Returns:
            answers: Tensor (None, vocab_size)
        """
        feed_dict = {self._stories: stories, self._queries: queries, self._memory_value: memory_value}
        return self._sess.run(self.predict_op, feed_dict=feed_dict)

    def predict_proba(self, stories, queries, memory_value):
        """Predicts probabilities of answers.
        Args:
            stories: Tensor (None, memory_size, window_size)
            queries: Tensor (None, window_size)
            memory_value: Tensor (None, memory_size)
        Returns:
            answers: Tensor (None, vocab_size)
        """
        feed_dict = {self._stories: stories, self._queries: queries, self._memory_value: memory_value}
        return self._sess.run(self.predict_proba_op, feed_dict=feed_dict)

    def predict_log_proba(self, stories, queries, memory_value):
        """Predicts log probabilities of answers.
        Args:
            stories: Tensor (None, memory_size, window_size)
            queries: Tensor (None, window_size)
            memory_value: Tensor (None, memory_size)
        Returns:
            answers: Tensor (None, vocab_size)
        """
        feed_dict = {self._stories: stories, self._queries: queries, self._memory_value: memory_value}
        return self._sess.run(self.predict_log_proba_op, feed_dict=feed_dict)