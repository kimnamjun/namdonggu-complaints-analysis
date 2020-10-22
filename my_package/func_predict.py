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


def predict_naive_bayes(complaints: pd.DataFrame) -> pd.DataFrame:
    words = complaints['_명사추출']
    complaints['_명사추출'].fillna('치약', inplace=True)
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

    result4 = list()
    result5 = list()

    max_acc = 0

    for i in range(10):
        x_train, x_test, y_train, y_test = train_test_split(x, y2)

        svm = SVC(kernel='linear', gamma='auto')
        svm_model = svm.fit(X=x_train, y=y_train)
        y_pred = svm_model.predict(X=x_test)
        acc = accuracy_score(y_test, y_pred)
        result4.append(acc)

        if acc > max_acc:
            max_acc = acc
            print(i, 'svm', acc)
            y_pred = svm_model.predict(X=x)
            for idx, y_val in enumerate(y_pred):
                complaints.loc[idx, '_예측'] = y_dic_inv[y_val]

    print(sum(result4))  # 얘 뺴고는 고만고만한데 얘도 다른 애들보다 조금 높은 수준
    print(sum(result5))
    return predict_naive_bayes(complaints)
