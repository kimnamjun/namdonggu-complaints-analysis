"""
목적 : 민원 결과 양식에 맞게 csv 파일을 생성한다.

민원 데이터에서 두 번째 도메인 행을 지우고 '민원0_csv' 파일로 저장
민원 데이터 기준 : '국민신문고 민원현황목록(7월~9월).xlsx'

3분기 주소 전처리
"""
from collections import defaultdict
import pandas as pd

quarter = 1
file_name = [1746, 1723, 1655]
dic = defaultdict(int)
dic1 = defaultdict(int)
dic2 = defaultdict(int)
complaints = pd.read_csv(f'D:/온라인민원상담/outputs/{quarter}분기/민원_1103_{file_name[quarter-1]}_full.csv', encoding='cp949')

for x in complaints.loc[:,['_민원경로', '민원제목*']].values:
    dic[x[0]] += 1
    # if x[0] == '안전신문고':
    #     dic1[x[1]] += 1
    # if x[0] == '생활불편신고':
    #     dic2[x[1]] += 1

# print(dic)
# print(dic1)
# print(dic2)
for k, v in sorted(dic.items(), key=lambda x: -x[1]):
    print(k, v)