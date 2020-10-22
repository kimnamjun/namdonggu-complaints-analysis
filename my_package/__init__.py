import pandas as pd
from my_package import func_create_formatted_data
from my_package import func_update_columns
from my_package import func_predict

def create_formatted_data(original_complaints: pd.DataFrame) -> pd.DataFrame:
    """
    양식에 맞는 DataFrame 생성
    :param original_complaints: 민원 원본 데이터
    :return: 양식에 맞는 데이터
    """
    return func_create_formatted_data.create_formatted_data(original_complaints)


def add_type(complaints: pd.DataFrame) -> pd.DataFrame:
    """
    안전신문고 및 생활불편신고 분리
    """
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


def delete_temp_columns(complaints: pd.DataFrame) -> pd.DataFrame:
    """
    임시 컬럼 삭제
    """
    return func_update_columns.delete_temp_columns(complaints)


def add_nouns(complaints: pd.DataFrame) -> pd.DataFrame:
    return func_update_columns.add_nouns(complaints)

def predict_naive_bayes(complaints: pd.DataFrame) -> pd.DataFrame:
    return func_predict.predict_naive_bayes(complaints)