# 참고 : http://corazzon.github.io/matplotlib_font_setting

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

a = [0,1,2,3]
b = [1,2,4,8]

# [방법1] 각 컴포넌트마다 글꼴 설정

# 가능한 폰트 확인
font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')
print(font_list)
font1 = 'C:/Windows/Fonts/malgun.ttf'  # 기본 폰트
font2 = 'C:/Users/user/AppData/Local/Microsoft/Windows/Fonts/NanumSquareB.ttf'  # 설치 폰트

fontprop = fm.FontProperties(fname=font2)

plt.bar(a, height=b)
plt.xlabel('입력값', fontproperties=fontprop)
plt.ylabel('출력값', fontproperties=fontprop)
plt.show()


# [방법2] plt 전역 글꼴 설정
# 맑은 고딕은 되는데 나눔 글꼴은 안됨 (경로문제인듯)

'''
print(plt.rcParams['font.family'])  # 글꼴
print(plt.rcParams['font.size'])  # 글자 크기
print(plt.rcParams['font.serif'])  # serif 폰트 (바탕체 비슷)
print(plt.rcParams['font.sans-serif'])  # sans 폰트 (돋움체 비슷)
print(plt.rcParams['font.monospace'])  # 고정폭 폰트
'''

# [방법2-1]
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['font.size'] = 12

# [방법2-2]
# fontprop2 = fm.FontProperties(fname=font1).get_name()
# plt.rc('font', family=fontprop2)

plt.bar(a, height=b)
plt.xlabel('입력값')
plt.ylabel('출력값')
plt.show()


# [방법3] rcParams 설정 파일에 직접 설정
# 기본 글꼴 변경
# (생략)
