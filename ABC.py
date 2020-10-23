import numpy as np
import random, math, copy
import matplotlib.pyplot as plt
import fuzzy_c_neans as fcm
import numpy as np
import struct
from skimage import morphology,draw
from sklearn import svm
import time
import pickle
from sklearn.datasets import make_blobs
import random
import model
 
def GrieFunc(data, train_data, train_labels, test_data, test_labels):             #目标函数
    clf = svm.SVC(C=data[0], kernel='rbf', gamma=data[1], verbose=1)
    clf.fit(train_data, train_labels)
    right = 0
    test_num = len(test_labels)

    for i in range(test_num):
        tmp = clf.predict([test_data[i,:]])
        if int(tmp[0] == int(test_labels[i])):
            right += 1
    print("训练了一次SVM, 正确率{}/{}, C{}, lambda{}".format(right, test_num, data[0], data[1]))
    return right*1.0/test_num
 
class ABSIndividual:
    def __init__(self,  bound, train_data, train_labels, test_data, test_labels):
        self.score = 0.
        self.invalidCount = 0                      #无效次数（成绩没有更新的累积次数）
        self.chrom = [random.uniform(a,b) for a,b in zip(bound[0,:],bound[1,:])]   #随机初始化
        self._train_data = train_data
        self._train_labels = train_labels
        self._test_data = test_data
        self._test_labels = test_labels
        self.calculateFitness()    
    def calculateFitness(self):
        self.score = GrieFunc(self.chrom, self._train_data, self._train_labels, self._test_data, self._test_labels)          #计算当前成绩
        
class ArtificialBeeSwarm:
    def __init__(self, foodCount, onlookerCount, bound, maxIterCount, maxInvalidCount, train_data, train_labels, test_data, test_labels): 
        self._train_data = train_data
        self._train_labels = train_labels
        self._test_data = test_data
        self._test_labels = test_labels
        self.foodCount = foodCount                  #蜜源个数，等同于雇佣蜂数目
        self.onlookerCount = onlookerCount          #观察蜂个数 
        self.bound = bound                          #各参数上下界
        self.maxIterCount = maxIterCount            #迭代次数
        self.maxInvalidCount = maxInvalidCount      #最大无效次数
        self.foodList = [ABSIndividual(self.bound, train_data, train_labels, test_data, test_labels) for k in range(self.foodCount)]   #初始化各蜜源
        self.foodScore = [d.score for d in self.foodList]                             #各蜜源最佳成绩
        self.bestFood = self.foodList[np.argmax(self.foodScore)]                      #全局最佳蜜源
 
    def updateFood(self, i):                                                  #更新第i个蜜源
        k = random.randint(0, self.bound.shape[1] - 1)                         #随机选择调整参数的维度
        j = random.choice([d for d in range(self.bound.shape[1]) if d !=i])   #随机选择另一蜜源作参考,j是其索引号
        vi = copy.deepcopy(self.foodList[i])
        vi.chrom[k] += random.uniform(-1.0, 1.0) * (vi.chrom[k] - self.foodList[j].chrom[k]) #调整参数
        vi.chrom[k] = np.clip(vi.chrom[k], self.bound[0, k], self.bound[1, k])               #参数不能越界
        vi.calculateFitness()
        if vi.score > self.foodList[i].score:           #如果成绩比当前蜜源好
            self.foodList[i] = vi
            if vi.score > self.foodScore[i]:            #如果成绩比历史成绩好（如重新初始化，当前成绩可能低于历史成绩）
                self.foodScore[i] = vi.score
                if vi.score > self.bestFood.score:      #如果成绩全局最优
                    self.bestFood = vi
            self.foodList[i].invalidCount = 0
        else:
            self.foodList[i].invalidCount += 1
            
    def employedBeePhase(self):
        for i in range(0, self.foodCount):              #各蜜源依次更新
            self.updateFood(i)            
 
    def onlookerBeePhase(self):
        foodScore = [d.score for d in self.foodList]  
        maxScore = np.max(foodScore)        
        accuFitness = [(0.9*d/maxScore+0.1, k) for k,d in enumerate(foodScore)]        #得到各蜜源的 相对分数和索引号
        if(len(accuFitness) == 0):
            return False
        for k in range(0, self.onlookerCount):
            i = random.choice([d[1] for d in accuFitness if d[0] >= random.random()])  #随机从相对分数大于随机门限的蜜源中选择跟随
            self.updateFood(i)
        return True
    def scoutBeePhase(self):
        for i in range(0, self.foodCount):
            if self.foodList[i].invalidCount > self.maxInvalidCount:                    #如果该蜜源没有更新的次数超过指定门限，则重新初始化
                self.foodList[i] = ABSIndividual(self.bound, self._train_data, self._train_labels, self._test_data, self._test_labels)
                self.foodScore[i] = max(self.foodScore[i], self.foodList[i].score)
 
    def solve(self):
        trace = []
        trace.append((self.bestFood.score, np.mean(self.foodScore)))
        for k in range(self.maxIterCount):
            self.employedBeePhase()
            ok = self.onlookerBeePhase()
            if (not ok):
                print("不能有效更新 直接退出")
                break
            self.scoutBeePhase()
            trace.append((self.bestFood.score, np.mean(self.foodScore)))
        print(self.bestFood.score)
        print(self.bestFood.chrom)
        self.printResult(np.array(trace))
        return self.bestFood.chrom
 
    def printResult(self, trace):
        x = np.arange(0, trace.shape[0])
        plt.plot(x, [(1-d)/d for d in trace[:, 0]], 'r', label='optimal value')
        plt.plot(x, [(1-d)/d for d in trace[:, 1]], 'g', label='average value')
        plt.xlabel("Iteration")
        plt.ylabel("function value")
        plt.title("Artificial Bee Swarm algorithm for function optimization")
        plt.legend()
        plt.show()
 
if __name__ == "__main__":
    random.seed()
    vardim = 2

    n_samples = 10
    n_test_samples = 1
    n_bins = 2  # use 3 bins for calibration_curve as we have 3 clusters here
    centers = [(-5, -5), (5, 5)]

    #X, Y = make_blobs(n_samples=n_samples, n_features=255, cluster_std=1.0,
    #                  centers=centers, shuffle=False, random_state=42)


    #Xt, Yt = make_blobs(n_samples=n_test_samples, n_features=255, cluster_std=1.0,
    #                  centers=centers, shuffle=False, random_state=42)

    train_images_idx3_ubyte_file = '.\\minist\\train-images-idx3-ubyte\\train-images.idx3-ubyte'
    train_label_idx1_ubyte_file = '.\\minist\\train-labels-idx1-ubyte\\train-labels.idx1-ubyte'
    test_images_idx3_ubyte_file = '.\\minist\\t10k-images-idx3-ubyte\\t10k-images.idx3-ubyte'
    test_label_idx1_ubyte_file = '.\\minist\\t10k-labels-idx1-ubyte\\t10k-labels.idx1-ubyte'
    train_data = model.normalize(model.decode_idx3_ubyte(train_images_idx3_ubyte_file))
    train_label = model.decode_idx1_ubyte(train_label_idx1_ubyte_file)
    test_data = model.normalize(np.array(model.decode_idx3_ubyte(test_images_idx3_ubyte_file)))
    test_label = np.array(model.decode_idx1_ubyte(test_label_idx1_ubyte_file))

    tmp_data = []
    tmp_label = []
    tmp_test_data = []
    tmp_test_label = []
    for i in range(60000):
        if(random.random() < 0.9):
            continue
        if(train_label[i] == 0 or train_label[i] == 1):# or train_label[i] == 5):
            tmp_data.append(train_data[i])
            tmp_label.append(train_label[i])
    for i in range(10000):
        if(test_label[i] == 0 or test_label[i] == 1):# or test_label[i] == 5):
            tmp_test_data.append(test_data[i])
            tmp_test_label.append(test_label[i])

    bound = np.tile([[0.01], [500]], vardim)
    bound[1][1] = 1
    
    abs = ArtificialBeeSwarm(5, 5, bound, 100, 3, np.array(tmp_data), np.array(tmp_label), np.array(tmp_test_data), np.array(tmp_test_label))
    ret = abs.solve()