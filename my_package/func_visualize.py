from collections import Counter

import pandas as pd
from wordcloud import WordCloud

def show_wordcloud(complaints: pd.DataFrame):
    word_count = Counter()
    for words in complaints['민원내용*']:
        word_list = words.split(',')
        word_count.update(word_list)
    print(Counter)