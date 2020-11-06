# 두 노드 합하기
def union(a, b):
    fa = find(a)
    fb = find(b)
    if fa != fb:
        nodes[fb] = fa


# 최상위 부모노드 반환
def find(a):
    if a != nodes[a]:
        nodes[a] = find(nodes[a])
    return nodes[a]


# index: num, value: parent
nodes = [i for i in range(5)]

# 노드 연결
links = [[0,1],[3,4],[2,3],[0,4]]
for a, b in links:
    union(a, b)

# 연산 마무리
for i in nodes:
    find(i)

# 출력
for idx, val in enumerate(nodes):
    print(idx, val)
