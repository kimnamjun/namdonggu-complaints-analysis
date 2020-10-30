import math
import datetime

import numpy as np
import pandas as pd

# 원본 데이터 컬럼명 재정의
column_names = ['신청번호','신청일자','접수번호','접수일자','민원신청경로','신청경로상세','신청메뉴상세',
                '메뉴분류상세','신청인구분','민원인','주소','갑질피해여부','민원제목','민원요지','민원내용',
                '기타참고사항','기피구분','기피사유','처리유형','민원종류','업무분야','법정/내부처리기한',
                '공익신고여부','재신청여부','재신청차수','이전신청번호추가','처리기관','담당부서1','담당부서2',
                '담당부서3','담당부서4','blank','담당자','처리기간','E-MAIL','전화번호','관련법령','소속기관',
                '사후관리담당부서','사후관리담당자','사후관리담당자연락처','사후관리담당자E-MAIL','처리일',
                '민원요약','처리결과','검토확인','검토유무','온나라연계','처리연장횟수','1차연장사유',
                '2차연장사유','2차연장동의여부','2차연장동의방법','2차연장동의자','2차연장동의일자',
                '불산입기간처리횟수','내용보완요청횟수','내용보완응답횟수','1차추가답변','1차추가답변일',
                '2차추가답변','2차추가답변일','3차추가답변','3차추가답변일','추가답변횟수','1차만족도불만유형',
                '1차만족불만족사유','1차만족도등록일','1차만족도','2차만족도불만유형','2차만족불만족사유',
                '2차만족도등록일','2차만족도','3차만족도조사','3차만족불만족사유','3차만족도등록일','3차만족도',
                '4차만족도불만유형','4차만족불만족사유','4차만족도등록일','4차만족도']

# 2020년 1분기 기준 부서별 부서코드
dept = {'간석1동':3530024,'간석2동':3530025,'간석3동':3530026,'간석4동':3530027,'감사실':3530090,'건강증진과':3530082,
        '건설과':3530224,'건축과':3530226,'공동주택과':3530227,'공영개발과':3530223,'공원녹지과':3530219,
        '교통행정과':3530220,'구월1동':3530020,'구월2동':3530021,'구월3동':3530022,'구월4동':3530023,
        '기업지원과':3530178,'기획예산과':3530197,'남동산단지원사업소':3530209,'남촌도림동':3530035,
        '노인장애인과':3530212,'논현1동':3530096,'논현2동':3530097,'논현고잔동':3530119,'농축수산과':3530180,
        '대변인':3530195,'도시경관과':3530218,'도시재생과':3530222,'만수1동':3530028,'만수2동':3530029,
        '만수3동':3530030,'만수4동':3530031,'만수5동':3530032,'만수6동':3530033,'문화관광과':3530202,
        '민원봉사과':3530162,'방재하수과':3530225,'보건행정과':3530062,'보육정책과':3530215,'복지정책과':3530210,
        '사회보장과':3530211,'생활경제과':3530179,'서창2동':3530194,'세무과':3530186,'세입징수과':3530187,
        '소통협력담당관':3530188,'식품위생과':3530128,'아동복지과':3530214,'안전총괄과':3530199,'여성가족과':3530213,
        '의회사무국':3530018,'일자리정책과':3530198,'자동차관리과':3530221,'장수서창동':3530034,'재무과':3530174,
        '청소행정과':3530216,'체육진흥과':3530203,'총무과':3530157,'치매정신건강과':3530207,'토지정보과':3530191,
        '평생교육과':3530201,'환경보전과':3530217}

# 필수 컬럼에 대해 NA 처리
fillna_table = {
    '민원접수번호*': '번호없음',
    '민원등록일시*': '2020-07-01 00:00',
    '처리기한일시*': '2020-07-08 00:00',
    '담당자지정일시*': '2020-07-01 00:00',
    '답변일자*': '2020-07-01 00:00',
    '부서코드*': 0,
    '처리부서*': '부서없음',
    '처리담당자*': '담당자없음',
    '처리상태*': '완료',
    '답변일시*': '2020-07-01 00:00',
    '민원인주소*': '인천광역시 남동구 소래로 633'
}


def create_formatted_data(original_complaints: pd.DataFrame) -> pd.DataFrame:
    # 원본데이터 컬럼명 재정의, 필수 컬럼 아닌 것은 제외
    original_complaints.columns = column_names

    complaints = pd.DataFrame()

    complaints['민원접수번호*'] = original_complaints['신청번호']
    complaints['민원등록일자'] = pd.Series(tuple(map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M').date(), original_complaints['신청일자'])))
    complaints['민원등록일시*'] = pd.Series(tuple(map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M'), original_complaints['신청일자'])))
    complaints['민원제목*'] = None
    complaints['민원내용*'] = None
    complaints['처리기한일시*'] = pd.Series(tuple(map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M') + datetime.timedelta(days=7), original_complaints['신청일자'])))
    complaints['담당자지정일시*'] = original_complaints['접수일자']
    complaints['답변일자*'] = pd.Series(tuple(map(lambda x: x if isinstance(x, float) else datetime.datetime.strptime(x, '%Y-%m-%d %H:%M').date(), original_complaints['처리일'])))
    complaints['부서코드*'] = pd.Series(tuple(map(lambda x: dept.get(x, 0), original_complaints['담당부서4'])))
    complaints['처리부서*'] = original_complaints['담당부서4']
    complaints['처리담당자*'] = original_complaints['담당자']
    complaints['처리상태*'] = '완료'
    complaints['진행상태'] = None
    complaints['답변내용'] = None
    complaints['답변일시*'] = pd.Series(tuple(map(lambda x: x if isinstance(x, float) else datetime.datetime.strptime(x, '%Y-%m-%d %H:%M'), original_complaints['처리일'])))
    complaints['만족도평가'] = None
    complaints['만족도내용'] = None

    # 2020년 3분기 기준 주소 제대로 되지 않은 것 3개 있음 ('.' 하나, NA 둘)
    complaints['민원인주소*'] = pd.Series(tuple(map(lambda x: x.replace('[민원발생위치]', '') if x is not np.nan else x, original_complaints['주소'])))
    complaints['연장처리일수'] = None
    complaints['연장처리횟수'] = None
    complaints['연장처리기한'] = None
    complaints['실처리일수'] = None

    # 임시 컬럼
    complaints['_민원제목'] = original_complaints['민원제목'].fillna('')
    complaints['_민원내용'] = original_complaints['민원내용'].fillna('')
    complaints['_민원요지'] = original_complaints['민원요지'].fillna('')

    # 민원접수번호 순 정렬
    complaints.sort_values(by=['민원접수번호*'], inplace=True)

    # 필수 컬럼에 대해 NA 처리, fillna만으로 못잡는 것 같아서 이중으로 처리
    for column, substitute_value in fillna_table.items():
        complaints[column].fillna(substitute_value)  # np.nan만 하는건가?
        for idx in complaints.index:
            cell = complaints.loc[idx, column]
            # float('nan'), pd._libs.tslibs.nattype.NaTType 제외
            if isinstance(cell, float) and not cell >= 0 or isinstance(cell, pd._libs.tslibs.nattype.NaTType):
                complaints.loc[idx, column] = substitute_value


    print('양식에 맞는 DataFrame 생성 완료')
    return complaints