import my_queue

class TreeNode:
    def __init__(self):
        self.data = None
        self.left = -1
        self.right = -1
        self.index = -1

    def SetData(self, data):
        self.data = data

    def SetLeft(self, index):
        self.left = index

    def SetRight(self, index):
        self.right = index

    def SetIndex(self, index):
        self.index = index

    def GetIndex(self):
        return self.index

    def Show(self):
        print("data:{},left:{},right:{},index:{}".format(self.data, self.left, self.right, self.index))

class BinaryTree:
    def __init__(self):
        self.list = []

    def insert(self, data):
        if len(self.list) == 0:
            node = TreeNode()
            node.SetData(data)
            node.SetIndex(0)
            self.list.append(node)
            return
        else:
            curNode = self.list[0]
            while(True):
                if data > curNode.data:
                    if curNode.right == -1:
                        node = TreeNode()
                        node.SetData(data)
                        node.SetIndex(len(self.list))
                        self.list.append(node)
                        self.list[curNode.GetIndex()].right = node.GetIndex()
                        return
                    else:
                        rightIndex = curNode.right
                        curNode = self.list[rightIndex]
                elif data <= curNode.data:
                    if curNode.left == -1:
                        node = TreeNode()
                        node.SetData(data)
                        node.SetIndex(len(self.list))
                        self.list.append(node)
                        self.list[curNode.GetIndex()].left = node.GetIndex()
                        return
                    else:
                        leftIndex = curNode.left
                        curNode = self.list[leftIndex]
                else:
                    return

    def Print(self):
        for n in self.list:
            n.Show()

    def FrontPrint(self):
        if len(self.list) == 0:
            return
        self.frontPrint(0)

    def frontPrint(self, index):
        if index == -1:
            return
        node = self.list[index]
        self.frontPrint(node.left)
        print(node.data)
        self.frontPrint(node.right)

        # 左 中 右
        # 中 左 右
        # 左 右 中

    def levelPrint(self):
        que = my_queue.MyQueue()
        que.push(0)
        while(not que.empty()):
            index = que.front()
            node = self.list[index]
            if node.left != -1:
                que.push(node.left)
            if node.right != -1:
                que.push(node.right)
            print(node.data)
            que.pop()

if __name__ == "__main__":
    bt = BinaryTree()
    nums = [9,7,5,0,11,3,5,6,8,3,13]
    for n in nums:
        bt.insert(n)
    bt.Print()
    bt.levelPrint()
    bt.FrontPrint()