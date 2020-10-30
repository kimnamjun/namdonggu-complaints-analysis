from time import time
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM

print('텐서플로우 로드 완료')
complaints = pd.read_csv('D:/온라인민원상담/outputs/3분기/민원_1023_1330.csv', encoding='cp949')

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

# 파일 write
now = datetime.strftime(datetime.now(), '%m%d_%H%M')
complaints.to_csv(f'D:/온라인민원상담/outputs/3분기/민원_tf_{now}.csv', index=False, encoding='cp949')
