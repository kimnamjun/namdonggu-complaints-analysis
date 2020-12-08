import numpy as np
import pandas as pd
from my_package import func_create_formatted_data
from my_package import func_update_columns
from my_package import func_visualize
# from my_package import func_predict

def create_formatted_data(original_complaints: pd.DataFrame) -> pd.DataFrame:
    """
    양식에 맞는 DataFrame 생성
    :param original_complaints: 민원 원본 데이터
    :param quarter: 분기 변경
    :return: 양식에 맞는 데이터
    """
    return func_create_formatted_data.create_formatted_data(original_complaints)


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


def predict_nb(complaints: pd.DataFrame) -> pd.DataFrame:
    return func_predict.predict_nb(complaints)

def word2vec(complaints: pd.DataFrame) -> pd.DataFrame:
    return func_predict.word2vec(complaints)

def predict_lstm(complaints: pd.DataFrame) -> pd.DataFrame:
    return func_predict.predict_lstm(complaints)

def similarity_check_n_gram(complaints: pd.DataFrame, ngram_n=4, similarity=0.6, min_num=10):
    return func_predict.similarity_check_n_gram(complaints, ngram_n=ngram_n, similarity=similarity, min_num=min_num)


def show_wordcloud(complaints: pd.DataFrame, mask: np.array):
    """워드클라우드 생성"""
    return func_visualize.show_wordcloud(complaints, mask)