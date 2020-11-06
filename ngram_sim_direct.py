from time import time
from datetime import datetime
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


class Node:
    """ Disjoint Set Node """
    def __init__(self, num):
        self.num = num
        self.parent = self

    def union(self, other):
        a = self.find()
        b = other.find()
        if a != b:
            b.parent = a

    def find(self):
        if self != self.parent:
            self.parent = self.parent.find()
        return self.parent


def diff_ngram(sa, sb, n=2):
    def ngram(s, num):
        res = list()
        slen = len(s) - num + 1
        for i in range(slen):
            ss = s[i:i + num]
            res.append(ss)
        return res

    # nltk ngrams보다 빠른 것 같음
    a = ngram(sa, n)
    b = ngram(sb, n)

    cnt = 0
    for i in a:
        for j in b:
            if i == j:
                cnt += 1
    maxi = max(len(a), len(b), 1)
    return cnt / maxi


# colab용
"""
import pandas as pd
from google.colab import auth, drive
auth.authenticate_user()
drive.mount('/content/gdrive')
"""
complaints = pd.read_csv('D:/온라인민원상담/outputs/3분기/민원_1102_0918_summ.csv.csv', encoding='cp949')
print(complaints.head())
print('데이터 로드 완료')


ngram_n = 4
similarity = 0.6
min_num = 10


words = complaints['_단어추출'].fillna('')
tfidf_vectorizer = TfidfVectorizer()
x = tfidf_vectorizer.fit_transform(words)

nodes = [Node(i) for i in range(len(complaints.index))]

# ngram_list 에 words 리스트로 만들기
ngram_list = list()
for idx, words in enumerate(complaints['_단어추출']):
    ngram_list.append(words.split(',') if isinstance(words, str) else list())

start = time()
# 유사도 검사 (22182개 num=2 기준 874초, num=5 기준 536초)
# colab num=5 700초 정도 (115개였던가?), num=3 942초 165개 (5 >=)
# num=3에 유사도 0.9면 시간은 비슷하고 170개...? 가 나올수 있나? / 묶일게 안묶이니까 늘어난듯
# 유사도 0.8이나 1이나 차이 별로 없는 듯 (36개 36개 동일)
for idx1, val1 in enumerate(ngram_list):
    if idx1 % 100 == 0:
        print(idx1, round(time() - start, 2))
    for idx2, val2 in enumerate(ngram_list[idx1+1:], start=idx1+1):
        sim = diff_ngram(val1, val2, ngram_n)  # ngram_n = 2
        if sim == similarity:  # similarity = 0.8
            nodes[idx1].union(nodes[idx2])

# 출력 순서대로 (대장은 그냥 앞에 있는 게 대장)
# 대장 번호, 부하 개수, 대장 내용, 부하 번호
dic = defaultdict(list)
for node in nodes:
    dic[node.parent.num].append(node.num)

cnt = 0
abc = list()
for key, value in dic.items():
    if len(value) >= min_num:  # min_num = 5
        cnt += 1
        abc.append((key, len(value), ngram_list[key], value))
abc.sort(key=lambda x: -x[1])
for a in abc:
    print(a)

print('time :', time() - start, cnt)