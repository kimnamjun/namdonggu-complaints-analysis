import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

tf.reset_default_graph()

def word2vec(complaints: pd.DataFrame) -> pd.DataFrame:
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

    # 5. softmax 알고리즘
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