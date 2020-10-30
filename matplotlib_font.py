# 참고 : http://corazzon.github.io/matplotlib_font_setting

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as pm

# 가능한 폰트 확인
font_list = pm.findSystemFonts(fontpaths=None, fontext='ttf')
print(font_list)

# 기본 폰트
font1 = 'C:/Windows/Fonts/malgun.ttf'
# 설치 폰트
font2 = 'C:/Users/user/AppData/Local/Microsoft/Windows/Fonts/NanumSquareB.ttf'

# 미완성
#matplotlib.rc('font', family=font1)
# plt.rc('font', family=font1)

a = [0,1,2,3]
b = [1,2,4,8]

plt.bar(a, height=b)
plt.xlabel('입력값')
plt.ylabel('출력값')
plt.show()