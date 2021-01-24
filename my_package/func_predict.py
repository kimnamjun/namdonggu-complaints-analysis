from time import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB  # NB model
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
# from xgboost import


def predict_svm(complaints: pd.DataFrame) -> pd.DataFrame:
    print('2만 2천여건 돌리는 데 10분 정도?')
    words = complaints['_명사추출']
    complaints['_명사추출'].fillna('x', inplace=True)
    x = TfidfVectorizer().fit_transform(words).toarray()
    print(x)

    complaints['_예측'] = None

    # 숫자 방식
    titles = complaints['민원제목*']
    complaints['y'] = None
    y_dic = dict()
    for idx, title in enumerate(sorted(set(titles))):
        y_dic[title] = idx
    print(y_dic)
    y_dic_inv = {v: k for k, v in y_dic.items()}
    # print(y_dic_inv)

    for idx, val in enumerate(complaints['민원제목*']):
        complaints.loc[idx, 'y'] = y_dic[val]
    y2 = np.array(complaints['y'], dtype='int')
    complaints.drop(['y'], axis='columns', inplace=True)
    print(y2)

    x_train, x_test, y_train, y_test = train_test_split(x, y2)

    svm = SVC(kernel='linear', gamma='auto')
    start = time()
    svm_model = svm.fit(X=x_train, y=y_train)
    print('fit 시간 :', time() - start)

    # y_pred = svm_model.predict(X=x_test)
    # print('predict 시간1 :', time() - start)

    y_pred = svm_model.predict(X=x)
    # acc = accuracy_score(y2, y_pred)
    # print('svm acc', acc)
    print('predict 시간2 :', time() - start)

    for idx, y_val in enumerate(y_pred):
        complaints.loc[idx, '_예측'] = y_dic_inv[y_val]

    return complaints
