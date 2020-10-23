'''
#增删查
def push(queue, num):
    queue.append(num)
    return queue

def pop(queue):
    queue = queue[1:]
    return queue

def front(queue):
    return queue[0]
'''

# 类
class MyQueue:
    # 构造函数
    def __init__(self):
        self.q = []
        #print("这是构造函数")

    def push(self,num):
        self.q.append(num)

    def pop(self):
        self.q = self.q[1:]

    def front(self):
        return self.q[0]

    def empty(self):
        return len(self.q) == 0

'''
class Student:
    def __init__(self):
        self.name = "默认姓名"
        self.age = 0
        self.sex = "默认性别"

    def Show(self):
        print(self.name)
        print(self.age)
        print(self.sex)
'''

if __name__ == "__main__":
    que = MyQueue()
    que.push(1)
    que.push(2)
    que.push(3)
    que.push(4)
    while(not que.empty()):
        print(que.front())
        que.pop()