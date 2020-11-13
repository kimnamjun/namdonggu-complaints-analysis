import numpy as np
import pandas as pd
from my_package import func_create_formatted_data
from my_package import func_update_columns
from time import time
# from my_package import func_predict
# from my_package import func_predict_tensorflow
from my_package import func_visualize


def create_formatted_data(original_complaints: pd.DataFrame, quarter=3) -> pd.DataFrame:
    """
    양식에 맞는 DataFrame 생성
    :param original_complaints: 민원 원본 데이터
    :param quarter: 분기 변경
    :return: 양식에 맞는 데이터
    """
    return func_create_formatted_data.create_formatted_data(original_complaints, quarter)


def add_type(complaints: pd.DataFrame) -> pd.DataFrame:
    """안전신문고 및 생활불편신고 분리"""
    return func_update_columns.add_type(complaints)


def add_title(complaints: pd.DataFrame) -> pd.DataFrame:
    """
    정규표현식으로 민원제목 채워넣기
    제목 + 요지로 1차 분류
    내용으로 2차 분류
    """
    return func_update_columns.add_title(complaints)


def add_text(complaints: pd.DataFrame) -> pd.DataFrame:
    """
    정규표현식으로 민원제목 채워넣기
    제목 + 요지로 1차 분류
    내용으로 2차 분류
    """
    return func_update_columns.add_text(complaints)


def set_dept_for_na(complaints: pd.DataFrame) -> pd.DataFrame:
    """빈칸으로 남겨진 부서 채우기"""
    return func_update_columns.set_dept_for_na(complaints)


def delete_temp_columns(complaints: pd.DataFrame) -> pd.DataFrame:
    """임시 컬럼 삭제"""
    return func_update_columns.delete_temp_columns(complaints)


def add_nouns(complaints: pd.DataFrame, deduplication=True) -> pd.DataFrame:
    """민원 내용을 바탕으로 명사 추출"""
    return func_update_columns.add_nouns(complaints, deduplication=deduplication)

def load_nouns(complaints: pd.DataFrame, ref_complaints: pd.DataFrame) -> pd.DataFrame:
    """다른 파일에서 불러오기"""
    return func_update_columns.load_nouns(complaints, ref_complaints)

def wide_to_long(complaints):
    start = time()
    prev = time()
    delay = list()

    for idx, val in enumerate(complaints['민원내용*']):
        if idx % 100 == 0:
            new_complaints = pd.DataFrame(complaints).iloc[:0, :]
            print('진행상황:', idx, round(time()-start), round(time()-prev))
            prev = time()
        try:
            words = val.split(',')
            for word in words:
                complaints.loc[idx, '민원내용*'] = word
                new_complaints = pd.concat([new_complaints, pd.DataFrame(complaints.loc[idx, :]).T])
        except:
            pass
        if idx % 100 == 99 or idx == len(complaints['민원내용*']) - 1:
            delay.append(new_complaints)
    new2_com = pd.concat(delay)
    return new2_com

def predict_svm(complaints: pd.DataFrame) -> pd.DataFrame:
    """
    navie_bayes
    """
    return func_predict.predict_svm(complaints)

def word2vec(complaints: pd.DataFrame) -> pd.DataFrame:
    return func_predict_tensorflow.word2vec(complaints)

def show_wordcloud(complaints: pd.DataFrame, mask: np.array):
    return func_visualize.show_wordcloud(complaints, mask)