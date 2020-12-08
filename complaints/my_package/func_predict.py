from time import time
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC


def predict_nb(complaints: pd.DataFrame) -> pd.DataFrame:
    print('나이브 베이즈 실행 : 2만 2천여건 돌리는 데 10분 정도?')
    words = complaints['_명사추출']
    complaints['_명사추출'].fillna('치약', inplace=True)
    x = TfidfVectorizer().fit_transform(words).toarray()
    print(x)

    complaints['_예측'] = None

    # 숫자 방식
    titles = complaints['민원제목*']
    complaints['y'] = None
    y_dic = dict()
    for idx, title in enumerate(sorted(set(titles))):
        y_dic[title] = idx
    print(y_dic)
    y_dic_inv = {v: k for k, v in y_dic.items()}
    # print(y_dic_inv)

    for idx, val in enumerate(complaints['민원제목*']):
        complaints.loc[idx, 'y'] = y_dic[val]
    y2 = np.array(complaints['y'], dtype='int')
    complaints.drop(['y'], axis='columns', inplace=True)
    print(y2)

    x_train, x_test, y_train, y_test = train_test_split(x, y2)

    svm = SVC(kernel='linear', gamma='auto')
    start = time()
    svm_model = svm.fit(X=x_train, y=y_train)
    print('fit 시간 :', time() - start)

    # y_pred = svm_model.predict(X=x_test)
    # print('predict 시간1 :', time() - start)

    y_pred = svm_model.predict(X=x)
    # acc = accuracy_score(y2, y_pred)
    # print('svm acc', acc)
    print('predict 시간2 :', time() - start)

    for idx, y_val in enumerate(y_pred):
        complaints.loc[idx, '_예측'] = y_dic_inv[y_val]
    return complaints


def word2vec(complaints: pd.DataFrame) -> pd.DataFrame:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    print('word2vec 실행')

    words = complaints['_명사추출']  # 기준이 될 x값
    label = complaints['민원제목*']  # 기준이 될 y값

    words.fillna('', inplace=True)

    x = TfidfVectorizer().fit_transform(words).toarray()

    one_hot = OneHotEncoder()
    label = label.to_numpy().reshape(-1, 1)
    y = one_hot.fit_transform(label).toarray()
    dic = {idx: key for idx, key in enumerate(one_hot.categories_[0])}
    print(dic)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35)
    ###############
    # DNN network #
    ###############
    learning_rate = 0.03
    epochs = 20
    batch_size = 500
    iter_size = y_train.shape[0] // batch_size

    input_size = 3450  # 단어 개수
    hidden_node1 = 128
    hidden_node2 = 64
    output_size = 33  # 라벨 개수

    # 2. images 전처리
    # 1) 정규화 (max가 1.0이라 생략)

    # 4. X, Y 변수 정의
    X = tf.placeholder(dtype=tf.float32, shape=[None, input_size], name='X')
    Y = tf.placeholder(dtype=tf.float32, shape=[None, output_size], name='Y')

    # 5. softmax
    # 1) model
    w1 = tf.Variable(tf.random_normal([input_size, hidden_node1]), name='w1')
    b1 = tf.Variable(tf.random_normal([hidden_node1]), name='b1')
    hidden_output1 = tf.nn.relu(tf.matmul(X, w1) + b1, name='output1')

    w2 = tf.Variable(tf.random_normal([hidden_node1, hidden_node2]), name='w2')
    b2 = tf.Variable(tf.random_normal([hidden_node2]), name='b2')
    hidden_output2 = tf.nn.relu(tf.matmul(hidden_output1, w2) + b2, name='output2')

    w3 = tf.Variable(tf.random_normal([hidden_node2, output_size]), name='w3')
    b3 = tf.Variable(tf.random_normal([output_size]), name='b3')
    model = tf.add(tf.matmul(hidden_output2, w3), b3, name='model')

    # size = [3450, 128, 64, 33]
    # weights = bias = output = list()
    # for i in range(len(size)-1):
    #     weights.append(tf.Variable(tf.random_normal([size[i], size[i+1]])))
    #     bias.append(tf.Variable(tf.random_normal([output_size])))
    #     output.append(tf.nn.relu(output[i], weights[i]) + bias[i])  # 마지막은 matmul이네
    # model = output[-1]

    # 2) softmax
    softmax = tf.nn.softmax(model, name='softmax')  # 활성함수

    # 3) loss function : Softmax + Cross Entropy
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=model), name='loss')

    # 4) optimizer
    train = tf.train.AdamOptimizer(learning_rate=learning_rate, name='optimizer').minimize(loss)

    # 5) encoding -> decoding
    y_pred = tf.argmax(softmax, axis=1, name='argmax_y_pred')

    # 6. model training
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # 반복학습 : 300회
        for epoch in range(epochs):
            total_loss = 0

            for step in range(iter_size):
                idx = np.random.choice(a=y_train.shape[0], size=batch_size, replace=False)  # a=60000, 비복원 추출
                _, loss_val = sess.run([train, loss], {X: x_train[idx], Y: y_train[idx]})
                total_loss += loss_val

            # 1 epoch 종료
            avg_loss = total_loss / iter_size
            print(f'epoch: {epoch + 1}, loss: {avg_loss}')

        # model test
        # y_pred_result = sess.run(y_pred, {X: x_test, Y: y_test})

        y_pred_result = sess.run(y_pred, {X: x, Y: y})

        # cmd에서 해당 폴더로 이동 후 "tensorboard --logdir=./" 실행
        tf.summary.merge_all()
        writer = tf.summary.FileWriter('C:/Users/user/NamJun/tensorboard', sess.graph)
        writer.close()


    print(y_pred_result.shape)
    complaints['result'] = None
    index_length = len(complaints.index)
    false_count = 0
    count = 0
    complaints['is_same'] = True
    for idx in range(index_length):
        complaints.loc[idx, 'y_pred'] = dic[y_pred_result[idx]]
        if complaints.loc[idx, '민원제목*'] != complaints.loc[idx, 'y_pred']:
            complaints.loc[idx, 'is_same'] = False
            false_count += 1
        count += 1
    print('false/total:',false_count,'/',count)
    select = list()
    for column in complaints.columns:
        if not column.startswith('_'):
            select.append(column)
    select = ['민원접수번호*', '민원제목*', 'y_pred', '민원내용*', '_명사추출', '_민원요지', '_민원제목', '_민원내용']
    complaints = complaints.loc[:, select]
    return complaints


def predict_lstm(complaints: pd.DataFrame) -> pd.DataFrame:
    import tensorflow as tf
    from keras.models import Sequential
    from keras.layers import Dense, LSTM

    words = complaints['_명사추출'].fillna('')  # 기준이 될 x값
    tfidf_vectorizer = TfidfVectorizer()
    x = tfidf_vectorizer.fit_transform(words).toarray()
    x = x.reshape((x.shape[0], x.shape[1], 1))

    label = complaints['민원제목*']  # 기준이 될 y값
    one_hot_encoder = OneHotEncoder()
    label_encoder = LabelEncoder()
    label = label.to_numpy().reshape(-1, 1)
    label_num = label_encoder.fit_transform(label).reshape(-1, 1)
    y = one_hot_encoder.fit_transform(label_num).toarray()

    print('X shape:', x.shape)
    print('Y shape:', y.shape)

    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    # print(x_train.shape, y_train.shape)
    # 2. 모델 구성
    model = Sequential()
    model.add(LSTM(600, activation='relu', return_sequences=True, input_shape=(x.shape[1], 1)))
    model.add(LSTM(100, activation='relu', input_shape=(200, 1)))  # 리턴시퀀스는 마지막꺼는 빼고
    # DENSE와 사용법 동일하나 input_shape=(열, 몇개씩잘라작업)
    # model.add(Dense(5))
    model.add(Dense(y.shape[1], activation='softmax'))

    model.summary()

    # 3. 실행
    print(x.shape, y.shape)
    model.compile(optimizer='adam', loss='mse')
    model.fit(x, y, epochs=10, batch_size=500)

    yhat = model.predict(x)
    yhat_argmax = np.argmax(yhat, axis=1)
    complaints['y_hat'] = yhat_argmax
    print(yhat)
    complaints['y_hat2'] = label_encoder.inverse_transform(yhat_argmax)
    return complaints


def similarity_check_n_gram(complaints: pd.DataFrame, ngram_n=4, similarity=0.6, min_num=10):
    class Node:
        """ Disjoint Set Node """

        def __init__(self, num):
            self.num = num
            self.parent = self

        def union(self, other):
            a = self.find()
            b = other.find()
            if a != b:
                b.parent = a

        def find(self):
            if self != self.parent:
                self.parent = self.parent.find()
            return self.parent

    def diff_ngram(sa, sb, n=2):
        def ngram(s, num):
            res = list()
            slen = len(s) - num + 1
            for i in range(slen):
                ss = s[i:i + num]
                res.append(ss)
            return res

        # nltk ngrams보다 빠른 것 같음
        a = ngram(sa, n)
        b = ngram(sb, n)

        cnt = 0
        for i in a:
            for j in b:
                if i == j:
                    cnt += 1
        maxi = max(len(a), len(b), 1)
        return cnt / maxi

    # 민원 경로 기타로 빠진 것 필터링
    # complaints = complaints[complaints['_민원경로'] == '-']

    words = complaints['_단어추출'].fillna('')
    tfidf_vectorizer = TfidfVectorizer()
    x = tfidf_vectorizer.fit_transform(words)

    nodes = [Node(i) for i in range(len(complaints.index))]

    # ngram_list 에 words 리스트로 만들기
    ngram_list = list()
    for idx, words in enumerate(complaints['_단어추출']):
        ngram_list.append(words.split(',') if isinstance(words, str) else list())

    start = time()
    # 유사도 검사 (22182개 num=2 기준 874초, num=5 기준 536초)
    # colab num=5 700초 정도 (115개였던가?), num=3 942초 165개 (5 >=)
    # num=3에 유사도 0.9면 시간은 비슷하고 170개...? 가 나올수 있나? / 묶일게 안묶이니까 늘어난듯
    # 유사도 0.8이나 1이나 차이 별로 없는 듯 (36개 36개 동일)
    for idx1, val1 in enumerate(ngram_list):
        if idx1 % 100 == 0:
            print(idx1, round(time() - start, 2))
        for idx2, val2 in enumerate(ngram_list[idx1 + 1:], start=idx1 + 1):
            sim = diff_ngram(val1, val2, ngram_n)  # ngram_n = 2
            if sim == similarity:  # similarity = 0.8
                nodes[idx1].union(nodes[idx2])

    # 출력 순서대로 (대장은 그냥 앞에 있는 게 대장)
    # 대장 번호, 부하 개수, 대장 내용, 부하 번호
    dic = defaultdict(list)
    for node in nodes:
        dic[node.parent.num].append(node.num)

    cnt = 0
    abc = list()
    for key, value in dic.items():
        if len(value) >= min_num:  # min_num = 5
            cnt += 1
            abc.append((key, len(value), ngram_list[key], value))
    abc.sort(key=lambda x: -x[1])
    for a in abc:
        print(a)

    print('time :', time() - start, cnt)