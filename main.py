"""
목적 : 민원 결과 양식에 맞게 csv 파일을 생성한다.

민원 데이터에서 두 번째 도메인 행을 지우고 '민원0_csv' 파일로 저장
민원 데이터 기준 : '국민신문고 민원현황목록(7월~9월).xlsx'
"""
from time import time
from datetime import datetime
import pandas as pd
import my_package as my

start = time()

# 원본 데이터 불러오기
original_complaints = pd.read_csv('D:/온라인민원상담/outputs/3분기/민원0.csv', encoding='cp949')
print(f'{round(time() - start,2)}초 경과')

# 양식에 맞게 DataFrame 생성
complaints = my.create_formatted_data(original_complaints)
print(f'{round(time() - start,2)}초 경과')

# 민원유입경로 생성
complaints = my.add_type(complaints)
print(f'{round(time() - start,2)}초 경과')

# 민원제목 생성
complaints = my.add_title(complaints)
print(f'{round(time() - start,2)}초 경과')

# 민원내용 생성 (유입경로에 종속)
complaints = my.add_text(complaints)
print(f'{round(time() - start,2)}초 경과')

# 임시컬럼 삭제
# complaints = my.delete_temp_columns(complaints)
# print(f'{round(time() - start,2)}초 경과')

# 명사 추출
complaints = my.add_nouns(complaints)

# 엑셀로 열면 '='로 시작하는 필드값이 '#NAME?'이 되는 경우가 있음
now = datetime.strftime(datetime.now(), '%m%d_%H%M')
complaints.to_csv(f'D:/온라인민원상담/outputs/3분기/민원_{now}.csv', index=False, encoding='cp949')
print(f'{round(time() - start,2)}초 경과 완료')