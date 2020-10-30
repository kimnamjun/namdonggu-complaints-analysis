def solution(string):
    def sub_function(n, s):
        return s

    string = '(' + string + ')'
    unpair = list()
    pair = list()
    number = list()
    for i, s in enumerate(string):
        if s == '(':
            if number:
                num = 0
                for e, n in enumerate(reversed(number)):
                    num += n * 10 ** e
            else:
                num = 1
            unpair.append([num, i])
            number = list()
        elif s == ')':
            pair.append([unpair[-1][1], i, unpair[-1][0]])
            unpair.pop()
        elif '0' <= s <= '9':
            number.append(int(s))
    pair = sorted(pair)
    print(pair)
    return sub_function(1, string)

print('정답:', solution('ab2(x3(y)z)'))


