from PIL import Image
import numpy as np
import pandas as pd
import my_package as my

complaints = pd.read_csv('D:/온라인민원상담/outputs/3분기/민원_1109_1141_summ.csv', encoding='cp949')
mask = np.array(Image.open('D:/온라인민원상담/이미지/남동구 마스크.png'))
my.show_wordcloud(complaints, mask)
