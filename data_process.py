#!/usr/bin/env python
# encoding: utf-8

from __future__ import absolute_import

import os
import re
import numpy as np


def load_task(data_dir, task_id, only_supporting=False):
    '''Load the nth task. There are 20 tasks in total.
    Returns a tuple containing the training and testing data for the task.
    '''

    #assert断言是声明其布尔值必须为真的判定，如果发生异常就说明表达示为假
    assert task_id > 0 and task_id < 21

    files = os.listdir(data_dir)  # os.listdir(dirname)：列出dirname下的目录和文件，返回的是list
    files = [os.path.join(data_dir, f) for f in files]  #获得每一个文件名的路径
    s = 'qa{}_'.format(task_id)  #s = qa{task_id}_  文件名称的形式
    train_file = [f for f in files if s in f and 'train' in f][0]  #选出满足条件的文件，即文件名中既有qa{task_id}_又有train，得到的train_file是一个文件名的字符串 # [f for f in files if s in f and 'train' in f]: ['/mnt/c/cyn/en/qa1_single-supporting-fact_train.txt']
    test_file = [f for f in files if s in f and 'test' in f][0]
    train_data = get_stories(train_file, only_supporting)
    test_data = get_stories(test_file, only_supporting)
    return train_data, test_data
    #data: [([['mary', 'moved', 'to', 'the', 'bathroom'], ['john', 'went', 'to', 'the', 'hallway']], ['where', 'is', 'mary'], ['bathroom']),
                  # ([['mary', 'moved', 'to', 'the', 'bathroom'], ['john', 'went', 'to', 'the', 'hallway'], ['daniel', 'went', 'back', 'to', 'the', 'hallway'], ['sandra', 'moved', 'to', 'the', 'garden']], ['where', 'is', 'daniel'], ['hallway']),
                 # ([[``````)
                 # ]


def tokenize(sent):

    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''

    #>>> re.split('\W+', 'Words, words, words.')     ['Words', 'words', 'words', '']
    #>>> re.split('(\W+)', 'Words, words, words.')   ['Words', ', ', 'words', ', ', 'words', '.', '']
    #>>> re.split('\W+', 'Words, words, words.', 1) ['Words', 'words, words.']
    #\W 匹配任意非数字和字母:[^a-zA-Z0-9]    "+"1 个或多个字符（贪婪匹配）

    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]  # 返回的是word和标点符号组成的list   #strip()删除所有空白符


def parse_stories(lines, only_supporting=False):   # lines是一个list，每一行一个元素
    '''Parse stories provided in the bAbI tasks format
    If only_supporting is true, only the sentences that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:  # line是lines这个list的一个元素
        line = str.lower(line)
        nid, line = line.split(' ', 1)  #把编号和后面的句子分开  #'1 Mary moved to the bathroom.\r\n'  # 1是分割次数
        nid = int(nid)  #分割出来的编号还是字符串的形式，把它转换成int形式
        if nid == 1:
            story = []
        if '\t' in line: #line里面有\t说明是一个question   # 'Where is Mary? \tbathroom\t1\r\n'
            q, a, supporting = line.split('\t')  #两个\t分别划分了question，answer，supporting line
            q = tokenize(q)  # q:Where is Mary?  得到一个词语和标点符号组成的list
            #a = tokenize(a)
            # answer is one vocab word even if it's actually multiple words
            a = [a]
            substory = None

            # remove question marks
            if q[-1] == "?":
                q = q[:-1]  # 不包括最后一个元素

            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]

            data.append((substory, q, a)) #data: [([['mary', 'moved', 'to', 'the', 'bathroom'], ['john', 'went', 'to', 'the', 'hallway']], ['where', 'is', 'mary'], ['bathroom']),
                                          # ([['mary', 'moved', 'to', 'the', 'bathroom'], ['john', 'went', 'to', 'the', 'hallway'], ['daniel', 'went', 'back', 'to', 'the', 'hallway'], ['sandra', 'moved', 'to', 'the', 'garden']], ['where', 'is', 'daniel'], ['hallway']),
                                          # ([[``````)   通过append向尾部添加一个元素，元素类型是set
            story.append('')  # story: [['mary', 'moved', 'to', 'the', 'bathroom'], ['john', 'went', 'to', 'the', 'hallway'], '']
                              # story: [['mary', 'moved', 'to', 'the', 'bathroom'], ['john', 'went', 'to', 'the', 'hallway'], '', ['daniel', 'went', 'back', 'to', 'the', 'hallway'], ['sandra', 'moved', 'to', 'the', 'garden'], '']
                              # story: ``````
            #story是前面所有的非query的句子组成的list，substory是这个query前所有的非query，data是以一个query为一个set而组成的list

        else: # regular sentence  # line  ：'Mary moved to the bathroom.'
            # remove periods
            sent = tokenize(line)
            if sent[-1] == ".":
                sent = sent[:-1]  # 取句号前面的
            story.append(sent)
    return data   #data: [([['mary', 'moved', 'to', 'the', 'bathroom'], ['john', 'went', 'to', 'the', 'hallway']], ['where', 'is', 'mary'], ['bathroom']),
                  # ([['mary', 'moved', 'to', 'the', 'bathroom'], ['john', 'went', 'to', 'the', 'hallway'], ['daniel', 'went', 'back', 'to', 'the', 'hallway'], ['sandra', 'moved', 'to', 'the', 'garden']], ['where', 'is', 'daniel'], ['hallway']),
                 # ([[``````)


def get_stories(f, only_supporting=False):  # f是文件名
    '''Given a file name, read the file, retrieve the stories, and then convert the sentences into a single story.
    If max_length is supplied, any stories longer than max_length tokens will be discarded.
    '''
    with open(f) as f:
        return parse_stories(f.readlines(), only_supporting=only_supporting)  # readlines返回的是list，每一行是一个元素，['一行','','',``````]

def vectorize_data(data, word_idx, window_size, memory_size):
    """
    Vectorize stories and queries.
    If a sentence length < sentence_size, the sentence will be padded with 0's.
    If a story length < memory_size, the story will be padded with empty memories.
    Empty memories are 1-D arrays of length sentence_size filled with 0's.
    The answer array is returned as a one-hot encoding.
    """
    #data: [([['mary', 'moved', 'to', 'the', 'bathroom'], ['john', 'went', 'to', 'the', 'hallway']], ['where', 'is', 'mary'], ['bathroom']), ([['mary', 'moved', 'to', 'the', 'bathroom'], ['john', 'went', 'to', 'the', 'hallway'], ['daniel', 'went', 'back', 'to', 'the', 'hallway'], ['sandra', 'moved', 'to', 'the', 'garden']], ['where', 'is', 'daniel'], ['hallway'])]
    S = []  # story(window)
    Q = []  # query
    A = []  # answer
    C = []  # window center
    for story, query, answer in data:  # 对于每一个set中的这三个list
        ss = []
        for i, sentence in enumerate(story, 1):
            for w in sentence:
                ss.append(word_idx[w])
        ss = ss[::-1][:memory_size][::-1]  # [8, 9, 12, 11, 2, 7, 13, 12, 11, 5]        [8, 9, 12, 11, 2, 7, 13, 12, 11, 5, 3, 13, 1, 12, 11, 5, 10, 9, 12, 11, 4]

        # Window Level Encoding,window--key,window center--value
        num = len(ss)
        window = []
        center = []
        for j in range(num - window_size + 1):
            list_tem = ss[j: window_size + j]  # [8, 9, 12, 11, 2]   [9, 12, 11, 2, 7]   [12, 11, 2, 7, 13] ``````
            window.append(list_tem)  # [[8, 9, 12, 11, 2], [9, 12, 11, 2, 7], [12, 11, 2, 7, 13], [11, 2, 7, 13, 12], [2, 7, 13, 12, 11], [7, 13, 12, 11, 5]],
                                    #  [[8, 9, 12, 11, 2], [9, 12, 11, 2, 7], [12, 11, 2, 7, 13], [11, 2, 7, 13, 12], [2, 7, 13, 12, 11], [7, 13, 12, 11, 5], [13, 12, 11, 5, 3], [12, 11, 5, 3, 13], [11, 5, 3, 13, 1], [5, 3, 13, 1, 12], [3, 13, 1, 12, 11], [13, 1, 12, 11, 5], [1, 12, 11, 5, 10], [12, 11, 5, 10, 9], [11, 5, 10, 9, 12], [5, 10, 9, 12, 11], [10, 9, 12, 11, 4]]
            center_word = list_tem[(window_size - 1) / 2]
            center.append(center_word)  # [12, 11, 2, 7, 13, 12]    [12, 11, 2, 7, 13, 12, 11, 5, 3, 13, 1, 12, 11, 5, 10, 9, 12]

        lm = max(0, memory_size - len(window))
        for _ in range(lm):  # 差几个memory_size,补齐0
            window.append([0] * window_size)
        S.append(window)  # [[[8, 9, 12, 11, 2], [9, 12, 11, 2, 7], [12, 11, 2, 7, 13], [11, 2, 7, 13, 12], [2, 7, 13, 12, 11], [7, 13, 12, 11, 5], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                          # [[9, 12, 11, 2, 7], [12, 11, 2, 7, 13], [11, 2, 7, 13, 12], [2, 7, 13, 12, 11], [7, 13, 12, 11, 5], [13, 12, 11, 5, 3], [12, 11, 5, 3, 13], [11, 5, 3, 13, 1], [5, 3, 13, 1, 12], [3, 13, 1, 12, 11], [13, 1, 12, 11, 5], [1, 12, 11, 5, 10], [12, 11, 5, 10, 9], [11, 5, 10, 9, 12], [5, 10, 9, 12, 11], [10, 9, 12, 11, 4], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]

        lc = max(0, memory_size - len(center))
        for _ in range(lc):  # 差几个memory_size,补齐0
            center.append(0 * window_size)
        C.append(center)

        lq = max(0, window_size - len(query))
        q = [word_idx[w] for w in query] + [0] * lq

        y = np.zeros(len(word_idx) + 1)  # 0 is reserved for nil word
        for a in answer:  #answer是一个one-hot vector
            y[word_idx[a]] = 1  #把y的相应那个维度令成1

        Q.append(q)
        A.append(y)
    return np.array(S), np.array(Q), np.array(A), np.array(C)
    # data = [([['mary', 'moved', 'to', 'the', 'bathroom'], ['john', 'went', 'to', 'the', 'hallway']], ['where', 'is', 'mary'], ['bathroom']),([['mary', 'moved', 'to', 'the', 'bathroom'], ['john', 'went', 'to', 'the', 'hallway'], ['daniel', 'went', 'back', 'to', 'the', 'hallway'], ['sandra', 'moved', 'to', 'the', 'garden']], ['where', 'is', 'daniel'], ['hallway'])]
    #word_idx = {'hallway': 5, 'bathroom': 2, 'garden': 4, 'is': 6, 'went': 13, 'moved': 9, 'back': 1, 'to': 12, 'daniel': 3, 'sandra': 10, 'the': 11, 'john': 7, 'where': 14, 'mary': 8}
    #window_size = 5   memory_size = 20

    """ S  [None, memory_size, window_size]
        [[[8, 9, 12, 11, 2],[9, 12, 11, 2, 7],[12, 11, 2, 7, 13],[11, 2, 7, 13, 12],[2, 7, 13, 12, 11],[7, 13, 12, 11, 5],[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
         [[9, 12, 11, 2, 7], [12, 11, 2, 7, 13], [11, 2, 7, 13, 12], [2, 7, 13, 12, 11], [7, 13, 12, 11, 5], [13, 12, 11, 5, 3], [12, 11, 5, 3, 13], [11, 5, 3, 13, 1], [5, 3, 13, 1, 12], [3, 13, 1, 12, 11], [13, 1, 12, 11, 5], [1, 12, 11, 5, 10], [12, 11, 5, 10, 9], [11, 5, 10, 9, 12], [5, 10, 9, 12, 11], [10, 9, 12, 11, 4], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]

        Q  [None, window_size]
        [[14, 6, 8, 0, 0],
         [14, 6, 3, 0, 0]]

        A  [None, vocab_size]
         [[ 0.  0.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
          [ 0.  0.  0.  0.  0.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.]]

        C  [None, memory_size]
         [[12, 11, 2, 7, 13, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [11, 2, 7, 13, 12, 11, 5, 3, 13, 1, 12, 11, 5, 10, 9, 12, 0, 0, 0, 0]]
"""




