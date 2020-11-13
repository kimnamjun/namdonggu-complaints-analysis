import time
x = 50
for i in range(x + 1):
    time.sleep(0.3)
    print(f'\r남은시간 : {i} / {x}', end='')