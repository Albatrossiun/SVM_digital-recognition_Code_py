# 哈希表   字典
# 字典   key:value
# {你好-hello  香蕉-banana  明天-tomorrow ...}   新华字典
'''
MyMap = []
MyMap.append(["你好","hello"])
MyMap.append(["香蕉","banana"])
MyMap.append(["明天","tomorrow"])
# O(n)
# O(1)

def lookUpDictionary(key):
    for i in range(len(MyMap)):
        if MyMap[i][0] == key:
            return MyMap[i][1]
    return "找不到"

if __name__ == "__main__":
    print(MyMap)
    key = input("请输入要查找的词")
    value = lookUpDictionary(key)
    print(value)
'''
'''
MyMap = []
MyMap.append([0,"hello"])
MyMap.append([1,"banana"])
MyMap.append([2,"tomorrow"])

def lookUpDictionary(key):
    return MyMap[key][1]

if __name__ == "__main__":
    print(MyMap)
    key = input("请输入要查找的词")
    value = lookUpDictionary(key)
    print(value)
'''

class MyHashList:
    def __init__(self):
        self.list = []   # 0
        for i in range(2):
            self.list.append(None)
        self.capacity = len(self.list)
        self.used = 0

    def __resize(self):
        newList = []
        newCapacity = self.capacity * 2
        for i in range(newCapacity):
            newList.append(None)
        for n in self.list:
            if(n == None):
                continue
            key = n[0]
            value = n[1]
            num = hash(key)
            index = num % len(newList)
            while(newList[index] != None):
                index = (index + 1) % newCapacity
            newList[index] = [key, value]
        self.list = newList
        self.capacity = newCapacity
        print("扩容结束")

    def Insert(self, key, value):
        if ((self.used * 1.0 / self.capacity) > 0.7):
            # 没有空间啦 要扩容
            self.__resize()
        num = hash(key)
        index = num % len(self.list)
        while(self.list[index] != None):
            if(self.list[index][0] == key):
                self.list[index][1] = value
                return
            index = (index + 1) % len(self.list)
            print("正在找空间")

        self.list[index] = [key, value]
        self.used += 1

    def Find(self, key):
        num = hash(key)
        index = num % len(self.list)
        times = 0
        while(times < len(self.list)):
            if(self.list[index] == None):
                return None
            if(self.list[index][0] == key):
                return self.list[index][1]
            index = (index + 1) % len(self.list)
            times += 1
        return None

    def Show(self):
        print(self.list)
        print("当前容量{}，当前使用{}".format(self.capacity, self.used))

if __name__ == "__main__":
    '''
    mh = MyHashList()
    mh.Insert("hello", "你好")
    mh.Show()
    mh.Insert("weather", "天气")
    mh.Show()
    mh.Insert("monkey", "泥猴")
    mh.Show()
    mh.Insert("aaa", "啊啊啊")
    mh.Show()
    mh.Insert("hhh", "哈哈哈")
    mh.Show()
    print(mh.Find("hello"))
    print(mh.Find("food"))
    
    '''

    d = {"hello":"你好","aaa":"啊啊啊","hhh":"哈哈哈"}
    print(d['hello'])
    d['asdasd'] = 'ABC'
    print(d['asdasd'])
