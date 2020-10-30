from time import time
from datetime import datetime
import pandas as pd
import my_package as my


complaints = pd.read_csv('D:/온라인민원상담/outputs/3분기/민원_1029_1645_summ.csv', encoding='cp949')
start = time()

complaints = my.wide_to_long(complaints)
print('시간경과(초) :', time() - start)

now = datetime.strftime(datetime.now(), '%m%d_%H%M')
complaints.to_csv(f'D:/온라인민원상담/outputs/3분기/민원_{now}_long.csv', index=False, encoding='cp949')
