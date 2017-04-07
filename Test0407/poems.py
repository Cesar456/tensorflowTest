import collections
import numpy as np
import tensorflow as tf
import jieba
from Test0407 import sqlUtil

# 选取特征词的个数
stopwords_set = [line.strip() for line in open('./stopword.txt', 'r', encoding='utf-8').readlines()]


# 2. 获取分词结果,词以空格隔开
# 并且按照词语数量对诗歌进行排序
def cut_word(contents):
    result = []
    for i in range(len(contents)):
        words = []
        segs = jieba.cut(contents[i])
        for seg in segs:
            seg = ''.join(seg.split())
            if 0 < len(seg) < 5 and seg not in stopwords_set:
                words.append(seg)
        result.append(words)
    poetrys = sorted(result, key=lambda line: len(line))
    return poetrys


# 统计没个词出现次数并取前x个常用词,x写在程序前几行
def get_feature_words(poetrys):
    all_word_dict = {}
    # 统计词频
    for words in poetrys:
        for word in words:
            if word in all_word_dict:
                all_word_dict[word] += 1
            else:
                all_word_dict[word] = 1
    # 将出现频率高的词汇放在前面
    all_words_tuple_list = sorted(all_word_dict.items(), key=lambda f: f[1], reverse=True)
    # 所有词汇列表,根据出现频率从高到低排过序
    all_word_list = list(zip(*all_words_tuple_list))[0]
    return all_word_list


# 获取诗歌对应的特征矩阵
def text_features(poem, feature_words):
    def text_features_inner(text_inner):
        features = [feature_words.index(word) for word in text_inner]
        return features

    feature_list = [text_features_inner(text) for text in poem]
    return feature_list


def get_matrix(poem_word_features):
    length = max(map(len, poem_word_features()))
    xdata = np.full((len(poem_word_features), length), 0, np.int32)
    for row in range(len(poem_word_features)):
        xdata[row, :len(poem_word_features[row])] = poem_word_features[row]
    ydata = np.copy(xdata)
    ydata[:, :-1] = xdata[:, 1:]
    return xdata, ydata


# 定义RNN
def neural_network(feature_words, len_poems, rnn_size=128, num_layers=2):
    input_data = tf.placeholder(tf.int32, [len_poems, None])
    output_targets = tf.placeholder(tf.int32, [len_poems, None])
    cell_fun = tf.contrib.rnn.BasicLSTMCell
    cell = cell_fun(rnn_size, state_is_tuple=True)
    cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
    # TODO
    initial_state = cell.zero_state(len_poems, tf.float32)
    with tf.variable_scope('rnnlm'):
        softmax_w = tf.get_variable("softmax_w", [rnn_size, len(feature_words) + 1])
        softmax_b = tf.get_variable("softmax_b", [len(feature_words) + 1])
        embedding = tf.get_variable("embedding", [len(feature_words) + 1, rnn_size])
        inputs = tf.nn.embedding_lookup(embedding, input_data)
    outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, scope='rnnlm')
    output = tf.reshape(outputs, [-1, rnn_size])
    logits = tf.matmul(output, softmax_w) + softmax_b
    probs = tf.nn.softmax(logits)
    return logits, last_state, probs, cell, initial_state


def train_rnn():
    poems = cut_word(sqlUtil.get_all_content())
    feature_wordss = get_feature_words(poems)
    poem_features = text_features(poems, feature_wordss)
    poem_matrix = get_matrix(poem_features)
    logits, last_state, _, _, _ = neural_network(feature_wordss, len(poems))
