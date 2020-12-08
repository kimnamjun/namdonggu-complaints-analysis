"""
머신러닝 / 딥러닝을 통한 제목 분류
실행은 되는데 사용하려면 조금 더 다듬어야 될수도?
"""
from datetime import datetime
import pandas as pd
import my_package as my

# 원본 데이터 불러오기
complaints = pd.read_csv('D:/온라인민원상담/outputs/3분기/민원_1023_1330.csv', encoding='cp949')

complaints = my.predict_nb(complaints)

complaints = my.word2vec(complaints)

my.similarity_check_n_gram(complaints)

now = datetime.strftime(datetime.now(), '%m%d_%H%M')
complaints.to_csv(f'D:/온라인민원상담/outputs/3분기/민원_predict_{now}.csv', index=False, encoding='cp949')

# 파라미터에 따라 며칠씩 걸릴 수도 있음
complaints = my.predict_lstm(complaints)

# 파일 write
now = datetime.strftime(datetime.now(), '%m%d_%H%M')
complaints.to_csv(f'D:/온라인민원상담/outputs/3분기/민원_tf_{now}.csv', index=False, encoding='cp949')