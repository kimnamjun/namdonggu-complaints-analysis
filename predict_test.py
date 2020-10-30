"""
목적 : 민원 결과 양식에 맞게 csv 파일을 생성한다.

민원 데이터에서 두 번째 도메인 행을 지우고 '민원0_csv' 파일로 저장
민원 데이터 기준 : '국민신문고 민원현황목록(7월~9월).xlsx'
"""
# tensorflow를 위해서 conda 3.7로 실행

from time import time
from datetime import datetime
import pandas as pd
import my_package as my

print('가즈아~!!')
# 원본 데이터 불러오기
complaints = pd.read_csv('D:/온라인민원상담/outputs/3분기/민원_1023_1330.csv', encoding='cp949')

# complaints = my.predict_svm(complaints)
complaints = my.word2vec(complaints)

now = datetime.strftime(datetime.now(), '%m%d_%H%M')
complaints.to_csv(f'D:/온라인민원상담/outputs/3분기/민원_predict_{now}.csv', index=False, encoding='cp949')