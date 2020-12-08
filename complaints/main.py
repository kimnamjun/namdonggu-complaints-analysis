from time import time
from datetime import datetime
import pandas as pd
import my_package as my

start = time()

# 원본 데이터 불러오기
original_complaints = pd.read_csv('D:/온라인민원상담/outputs/3분기/민원0.csv', encoding='cp949')

# 양식에 맞게 DataFrame 생성
complaints = my.create_formatted_data(original_complaints)

# 민원유입경로 생성
complaints = my.add_type(complaints)

# 민원제목 생성
complaints = my.add_title(complaints)

# 민원내용 생성 (유입경로에 종속)
complaints = my.add_text(complaints)

# 명사 추출 (민원내용에 종속)
complaints = my.add_nouns(complaints, deduplication=False)
# deduplication : 중복 제거
# True : 불법,광고물,제거,불법,광고물
# False: 불법,광고물,제거

# 부서없음 처리 (민원내용에 종속)
complaints = my.set_dept_for_na(complaints)

# 엑셀로 열면 '='로 시작하는 필드값이 '#NAME?'이 되는 경우가 있음
now = datetime.strftime(datetime.now(), '%m%d_%H%M')
complaints.to_csv(f'D:/온라인민원상담/outputs/3분기/민원_{now}_full.csv', index=False, encoding='cp949')

# 임시컬럼 삭제
complaints = my.delete_temp_columns(complaints)

complaints.to_csv(f'D:/온라인민원상담/outputs/3분기/민원_{now}_summ.csv', index=False, encoding='cp949')
print(f'{round(time() - start,2)}초 경과 완료')