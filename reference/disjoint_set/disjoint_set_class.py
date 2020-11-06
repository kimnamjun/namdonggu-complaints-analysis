class Node:
    """ Disjoint Set Node """
    def __init__(self, num):
        self.num = num
        self.parent = self

    # 두 노드 합하기
    def union(self, other):
        a = self.find()
        b = other.find()
        if a != b:
            b.parent = a

    # 최상위 부모노드 반환
    def find(self):
        if self != self.parent:
            self.parent = self.parent.find()
        return self.parent


# 노드 생성
nodes = list()
for i in range(5):
    nodes.append(Node(i))

# 노드 연결
links = [[0,1],[3,4],[2,3],[0,4]]
for a, b in links:
    nodes[a].union(nodes[b])

# 연산 마무리
for i in nodes:
    i.find()

# 출력
for i in nodes:
    print(i.num, i.parent.num)
