from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from wordcloud import WordCloud


def show_wordcloud(complaints: pd.DataFrame, mask=None):
    """
    폰트 없으면 아쉬운대로 맑은고딕: 'C:/Windows/Fonts/malgun.ttf'
    """
    word_count = Counter()
    for words in complaints['민원내용*']:
        if isinstance(words, str):
            word_list = words.split(',')
            word_count.update(word_list)
    word_cloud = WordCloud(font_path='C:\\Users\\user\\AppData\\Local\\Microsoft\\Windows\\Fonts\\NanumSquareB.ttf', width=500, height=400, max_words=100,
                           min_font_size=4, background_color='white', prefer_horizontal=1, mask=mask)
    result = word_cloud.generate_from_frequencies(word_count)
    plt.imshow(result)
    plt.axis('off')
    plt.show()
