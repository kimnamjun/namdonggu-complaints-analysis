import re
import time
from collections import defaultdict, OrderedDict, Counter

import numpy as np
import pandas as pd
from konlpy.tag import Kkma

# 정규표현식으로 확실하게 잡을 수 있는게 위쪽
# 대충해도 얻어걸릴 수 있을만한게 아래쪽
# 같은 분류라도 단어에 따라 여러번 검사할수도 있음
patterns = OrderedDict({
    # 확정의 영역
    '비부과': re.compile('비부과'),
    '장애인주차구역': re.compile('장애인.{,20}차'),  # 장애인 스티커 미부착 차량...
    '소방시설불법주정차': re.compile('(소화|소방).{,20}(주차|정차|주정)'),
    '불법주정차': re.compile('(불법.{,20}(주차|정차|주정))|((주차|정차|주정)|위반)|(차량.{,20}과태료)'),
    '기타주정차': re.compile('주차|정차|주정'),
    '불법광고물': re.compile('불법.{,20}(광고|현수막|간판)'),
    '기타광고물': re.compile('(광고|현수막|간판)'),
    '불법건축물': re.compile('불법.{,20}건축물'),
    '불법개조차량': re.compile('튜닝|(제동|점멸|미점|브레이크)등|불법.{,10}개조.{,10}차|번호판|반사판|리플렉터'),
    'CCTV설치요청': re.compile('(CCTV|시시티비|씨씨티비|감시.?카메라).{,20}(설치|요청)'),
    '마스크미착용': re.compile('(마스크.{,20}(안|않|착용))|턱스크'),
    '강남급행': re.compile('강남.?급행'),
    '위험물관련': re.compile('위험물'),
    '코로나': re.compile('코로나|자가.*격리'),

    # 높은 가능성
    '쓰레기': re.compile('폐기물|쓰레기|투기'),
    '도로보수_1': re.compile('노면|기울|싱크홀|씽크홀|포트홀'),
    '도로보수_2': re.compile('도로.{,20}(시설|파손|수리|훼손|보수|정비|침하|파임|파인|메꾸|메꿔)'),
    '파손수리': re.compile('파손|수리|훼손|보수|정비'),
    '흡연': re.compile('흡연|금연|담배|꽁초'),
    '가로등': re.compile('가로등|조명'),
    '신호등': re.compile('신호등'),
    '아파트': re.compile('아파트'),
    '가로수': re.compile('가로수|나무'),
    '자전거': re.compile('자전거'),
    '방역': re.compile('방역'),
    '제설': re.compile('제설'),
    '오염': re.compile('대기.{,10}오염|황사|미세.?먼지|수질|소각'),
    '냄새': re.compile('냄새|악취'),
    '불친절': re.compile('불친절|불쾌|감정|고압적|자존심'),
    # '아동급식카드': re.compile('아동.?급식'), # 2만 여건 중 1건

    # 얻어걸릴 가능성
    '소음': re.compile('소음|시끄|방음'),
    '행정복지': re.compile('행정.?[^복지]|복지.?[^센터]'), # '행정'과 '복지'는 되지만 붙어있으면 안됨 '복지센터'도 안됨
    '대중교통': re.compile('버스|택시|노선'),
    '불법영업노점': re.compile('인도'),
    '위생': re.compile('위생|부패|변질|상한|상했|상하|상함'),
    '도시가스': re.compile('가스'),
    '수도': re.compile('수도|배수|맨홀'),
    '동물': re.compile('벌레|개|강아지|고양이|사체|시체|로드킬'),
    '사고': re.compile('사고')
})


def add_type(complaints: pd.DataFrame) -> pd.DataFrame:
    print('민원유입경로 추측 중')
    index_length = len(complaints.index)
    for idx in range(index_length):
        text = complaints.loc[idx, '_민원제목']
        if text.startswith('(SPP'):
            complaints.loc[idx, '_민원경로'] = '안전신문고'
        elif text.startswith('(안전'):
            complaints.loc[idx, '_민원경로'] = '안전신문고'
        elif text.startswith('[생활'):
            complaints.loc[idx, '_민원경로'] = '생활불편신고'
        elif text.startswith('스마트국민제보'):
            complaints.loc[idx, '_민원경로'] = '스마트'
        else:
            complaints.loc[idx, '_민원경로'] = '-'
    print('민원유입경로 추측 완료')
    return complaints


def add_title(complaints: pd.DataFrame) -> pd.DataFrame:
    hit_dict = defaultdict(int)
    no_hit = 0

    index_length = len(complaints.index)
    print('민원 요지 + 제목으로 1차 분류 중')
    for idx in range(index_length):
        text = complaints.loc[idx, '_민원요지'] + '\n' + complaints.loc[idx, '_민원제목']
        text = text.replace('\n', ' ')
        # 지금은 대표적인거 하나로 제목을 지정했는데 여러 범주에 포함되는 것도 있을지도?
        for key, pattern in patterns.items():
            if re.search(pattern, text):
                complaints.loc[idx, '민원제목*'] = key
                hit_dict[key] += 1
                break
        else:
            complaints.loc[idx, '민원제목*'] = '_미분류'

    print('민원 내용으로 2차 분류 중')
    for idx in range(index_length):
        text = complaints.loc[idx, '_민원요지']
        text = text.replace('\n', ' ')
        if complaints.loc[idx, '민원제목*'] == '_미분류':
            for key, pattern in patterns.items():
                if re.search(pattern, text):
                    complaints.loc[idx, '민원제목*'] = key
                    hit_dict[key] += 1
                    break
            else:
                complaints.loc[idx, '민원제목*'] = '_미분류_'
                no_hit += 1

    print('민원 제목 설정 완료')
    # 분류 결과 출력
    print('미분류:', no_hit)
    hit = sorted(hit_dict.items(), key=lambda x: (-x[1], x[0]))
    for i in range(len(hit)):
        # 한글은 2글자 차지하는데 1글자로 인식되서 살짝 밀림
        print(f'{hit[i][0]}: {hit[i][1]}'.ljust(25), end=' ')
        if i % 7 == 6:
            print()
    print()
    return complaints


macro_prefix = [
    re.compile(r' ?\*.*'),
    re.compile(r'( ?\(SPP[^)]*\))+'),
    re.compile(r'( ?\(안전[^)]*\))+'),
    re.compile(r'( ?\[.*])+'),
    re.compile(r'죄송합니다'),
    re.compile(r'&lt;|&gt;|[:punct]')
]
smart_prefix = [
    re.compile(r'\[이첩사유]\s+(.*)'),
    re.compile(r'\[민원내용]\s+(.*)')
]
pattern_space = re.compile(r'\s+')

def add_text(complaints: pd.DataFrame) -> pd.DataFrame:
    index_length = len(complaints.index)

    print('민원 내용 생성 중')
    for idx in range(index_length):
        if complaints.loc[idx, '_민원경로'] in ('안전신문고', '생활불편신고'):
            text = complaints.loc[idx, '_민원제목']
            text = re.sub(macro_prefix[0], ' ', text)
            text += ' ' + complaints.loc[idx, '_민원요지']
            for prefix in macro_prefix:
                text = re.sub(prefix, ' ', text)
            text = re.sub(pattern_space, ' ', text)
            text = text.strip()
            complaints.loc[idx, '민원내용*'] = text

        elif complaints.loc[idx, '_민원경로'] == '스마트':
            text = ''
            for prefix in smart_prefix:
                search = re.search(prefix, complaints.loc[idx, '_민원내용'])
                if search:
                    text += search.group(1) + ' '
            complaints.loc[idx, '민원내용*'] = text

        else:
            text = complaints.loc[idx, '_민원요지']
            if text.startswith('귀하의'):
                text = ''
            if not text:
                text = complaints.loc[idx, '_민원제목']
            if not text:
                text = '#NJ : ' + complaints.loc[idx, '_민원내용'][:100]
            complaints.loc[idx, '민원내용*'] = text

    print('민원 내용 생성 완료')
    return complaints


def delete_temp_columns(complaints: pd.DataFrame) -> pd.DataFrame:
    """
    임시 컬럼 삭제
    """
    select = list()
    for column in complaints.columns:
        if not column.startswith('_'):
            select.append(column)
    select = ['민원접수번호*','민원내용*','_명사추출']
    complaints = complaints.loc[:, select]

    print('임시 컬럼 삭제 완료')
    return complaints


def add_nouns(complaints: pd.DataFrame) -> pd.DataFrame:
    kkma = Kkma()
    index_length = len(complaints['민원내용*'])
    div = 1
    start_time = time.time()
    for idx in range(index_length):
        text = complaints.loc[idx, '민원내용*']
        ex_pos = kkma.pos(text)

        nouns = list()
        for term, wclass in ex_pos:
            if wclass == 'NNG' or wclass == 'NNP' or wclass == 'NP' or wclass == 'NNM':
                nouns.append(term)
        if idx % div == 0:
            print(f'{idx} / {index_length} : {round(time.time() - start_time, 2)}초')
        if idx // div == 10:
            div *= 10
        complaints.loc[idx, '_명사추출'] = ' '.join(nouns)

    return complaints


def set_dept_for_na():
    pass