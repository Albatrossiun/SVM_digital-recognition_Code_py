class MyStack:
    def __init__(self):
        self.s = []

    def push(self, num):
        tmp = []
        tmp.append(num)
        # tmp.append(self.s)  [1,2,3,4]  [5]    [5, [1,2,3,4]]
        tmp += self.s
        self.s = tmp

    def pop(self):
        self.s = self.s[1:]

    def top(self):
        return self.s[0]

    def empty(self):
        return len(self.s) == 0

if __name__ == "__main__":
    s = MyStack()
    s.push(1)  # 1
    s.push(2)  # 2 1
    s.push(3)  # 3 2 1
    s.push(4)  # 4 3 2 1
    while(not s.empty()):
        print(s.top())
        s.pop()