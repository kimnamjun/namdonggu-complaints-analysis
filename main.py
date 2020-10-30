"""
목적 : 민원 결과 양식에 맞게 csv 파일을 생성한다.

민원 데이터에서 두 번째 도메인 행을 지우고 '민원0_csv' 파일로 저장
민원 데이터 기준 : '국민신문고 민원현황목록(7월~9월).xlsx'

주소 전처리
 6335행 인천광역시 남동구 예술로 172
        인천광역시 남동구 구월동 1335-4
 7172행 경기도 안양시 동안구 경수대로 884번길
        경기도 안양시 동안구 비산동 466-7
 9179행 인천광역시 남동구 간석로 72번길
        인천광역시 남동구 간석동 129-12
 9186행 인천광역시 남동구 간석로 72번길
        인천광역시 남동구 간석동 129-12
12824행 인천광역시 남동구 포구로 40
        인천광역시 남동구 논현동 810
14997행 경기도 시흥시 목감중앙로 19
        경기도 시흥시 조남동 663
18793행 인천광역시 남동구 구월로 99
        인천광역시 남동구 간석동 493-27
18796행 인천광역시 남동구 구월로 372번길 90
        인천광역시 남동구 만수동 1123
19154행 경기도 안산시 단원구 원포공원1로
        경기도 안산시 단원구 초지동 745-2
21180행 인천광역시 남동구 소래로 633
"""
from time import time
from datetime import datetime
import pandas as pd
import my_package as my

start = time()

# 원본 데이터 불러오기
original_complaints = pd.read_csv('D:/온라인민원상담/outputs/2분기/민원0.csv', encoding='cp949')

# 양식에 맞게 DataFrame 생성
complaints = my.create_formatted_data(original_complaints)

# 민원유입경로 생성
complaints = my.add_type(complaints)

# 민원제목 생성
complaints = my.add_title(complaints)

# 민원내용 생성 (유입경로에 종속)
complaints = my.add_text(complaints)

# 명사 추출 (민원내용에 종속)
complaints = my.add_nouns(complaints)

# 엑셀로 열면 '='로 시작하는 필드값이 '#NAME?'이 되는 경우가 있음
now = datetime.strftime(datetime.now(), '%m%d_%H%M')
complaints.to_csv(f'D:/온라인민원상담/outputs/2분기/민원_{now}_full.csv', index=False, encoding='cp949')

# 임시컬럼 삭제
# '_단어추출' 컬럼을 '민원내용*'으로 변경하는 내용도 여기 들어가 있음
complaints = my.delete_temp_columns(complaints)

complaints.to_csv(f'D:/온라인민원상담/outputs/2분기/민원_{now}_summ.csv', index=False, encoding='cp949')
print(f'{round(time() - start,2)}초 경과 완료')