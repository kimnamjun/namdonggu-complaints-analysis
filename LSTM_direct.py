"""
민원이 접수되면 이 민원이 어느 부서에서 처리하는 지 분류하는 것이
실제 업무에서는 매우 중요한 일이다.
그렇기 때문에 모든 민원에는 처리부서가 있어야 한다.

대부분 민원의 경우 처리부서 label이 있지만
몇몇 민원의 경우 처리부서가 공란으로 남겨져 있다.

학습용과 검증용으로 label이 붙어있는 민원 중
20% 정도의 처리부서를 지워 공란이 되었다고 가정하고
모델 학습을 진행한다.

그 후 공란으로 남겨진 데이터에 대하여 label을 붙인다.
"""
import random
from time import time
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM

'>>> 데이터 로드'

complaints = pd.read_csv('C:/Users/story/Downloads/complaints.csv', encoding='UTF-8')

original_complaints = complaints
sample_num = 500
complaints = complaints.iloc[:sample_num, :]

'>>> x y 설정'

words = complaints['_명사추출'].fillna('')  # 기준이 될 x값
tfidf_vectorizer = TfidfVectorizer()
x = tfidf_vectorizer.fit_transform(words).toarray()
x = x.reshape((x.shape[0], x.shape[1], 1))

label = complaints['처리부서*']  # 기준이 될 y값
one_hot_encoder = OneHotEncoder()
label_encoder = LabelEncoder()
label = label.to_numpy().reshape(-1, 1)
label_num = label_encoder.fit_transform(label).reshape(-1, 1)
y = one_hot_encoder.fit_transform(label_num).toarray()

print('X shape:', x.shape)
print('Y shape:', y.shape)

train_id = random.sample(range(sample_num), int(sample_num * 0.8))
test_id = [i for i in range(sample_num) if i not in train_id]

x_train = x

'>>> 모델 생성'

model = Sequential()
model.add(LSTM(200, activation='relu', return_sequences=True, input_shape=(x.shape[1], 1)))
model.add(LSTM(100, activation='relu', input_shape=(200, 1)))  # 리턴시퀀스는 마지막꺼는 빼고
# DENSE와 사용법 동일하나 input_shape=(열, 몇개씩잘라작업)
# model.add(Dense(5))
model.add(Dense(y.shape[1], activation='softmax'))

model.summary()

'>>> 분류 시작'

start = datetime.now()
print(f'시작시간: {start}')
adam_opt = keras.optimizers.Adam(lr=0.001, decay=0.9)
model.compile(optimizer=adam_opt, loss='mse')
model.fit(x_train, y_train, epochs=3, batch_size=250)

yhat = model.predict(x_test)
yhat_argmax = np.argmax(yhat, axis=1)
complaints['분류부서'] = label_encoder.inverse_transform(yhat_argmax)

end = datetime.now()
print(f'종료시간: {end}')
print(f'소요시간: {end - start}')

'>>> 분류 결과'

correct = 0
miss = 0
for idx in complaints.index:
    if complaints.loc[idx, '처리부서*'] == complaints.loc[idx, '분류부서']:
        correct += 1
    else:
        miss += 1
print('correct', correct)
print('miss', miss)

'>>> 파일 저장'

now = datetime.strftime(datetime.now(), '%m%d_%H%M')
complaints.to_csv(f'C:/Users/story/Downloads/complaints_model_{now}.csv', index=False, encoding='UTF-8')
print('저장완료')