def solution(arr):
    def sub_task(row_s, col_s, length):
        exist_zero, exist_one = 0, 0

        for row in arr[row_s:row_s + length]:
            exist_zero = 1 if 0 in row[col_s:col_s + length] else exist_zero
            exist_one = 1 if 1 in row[col_s:col_s + length] else exist_one

        if exist_zero & exist_one:
            length_div2 = length // 2
            zero, one = 0, 0
            a, b = sub_task(row_s, col_s, length_div2)
            zero, one = zero + a, one + b
            a, b = sub_task(row_s, col_s + length_div2, length_div2)
            zero, one = zero + a, one + b
            a, b = sub_task(row_s + length_div2, col_s, length_div2)
            zero, one = zero + a, one + b
            a, b = sub_task(row_s + length_div2, col_s + length_div2, length_div2)
            zero, one = zero + a, one + b
            return [zero, one]
        elif exist_zero:
            return [1, 0]
        else:
            return [0, 1]

    return sub_task(0, 0, len(arr))



arr0 = [[1,1,1,1,1,1,1,1],[0,1,1,1,1,1,1,1],[0,0,0,0,1,1,1,1],[0,1,0,0,1,1,1,1],[0,0,0,0,0,0,1,1],[0,0,0,0,0,0,0,1],[0,0,0,0,1,0,0,1],[0,0,0,0,1,1,1,1]]
print(solution(arr0))