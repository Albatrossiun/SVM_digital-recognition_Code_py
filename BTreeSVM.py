import fuzzy_c_neans as fcm
import numpy as np
import struct
import matplotlib.pyplot as plt
from skimage import morphology,draw
from sklearn import svm
import time
import pickle
import model
import math
import random

BTFILENAME = "btree.txt"

class SVMNode:
    def __init__(self, file_path = "",type = 0):
        self._train_tyep = type
        self.file_path = file_path

    def SetData(self, train_data, train_labels, test_data, test_labels, name):
        self._train_data = train_data
        self._train_labels = train_labels
        self._test_data = test_data
        self._test_label = test_labels
        self._name = name
        self._left = "NULL"
        self._right = "NULL"

    def Train(self):
        clf = svm.SVC(C=100.0, kernel='rbf', gamma=0.03, verbose=1)
        clf.fit(self._train_data, self._train_labels)
        self.clf = clf
        
    def TrainWithABC(self):
        # TODO ABC获取最佳参数 但是太慢了
        self.Train()

    def Save(self, path):
        file_path = path + "\\" + self._name
        save = pickle.dumps(self.clf)
        f = open(file_path, 'wb')
        f.write(save)

    def Predict(self, data):
        tmp = self.clf.predict([data])
        return tmp[0]

    def Load(self, path):
        file_path = file_path = path + "\\" + self._name
        data = open(file_path, 'rb')
        tmp = data.read()
        model = pickle.loads(tmp)
        self.clf = model

    def SetLet(self, left_name):
        self._left = left_name

    def SetRight(self, right_name):
        self._right = right_name

class Node:
    def __init__(self, name):
        self._left = "NULL"
        self._right = "NULL"
        self._name = name
        self._left_label = []
        self._right_label = []

class BTreeSVM:
    def __init__(self, model_path = ""):
        self._tree_dict = {}
        self._rootNodeName = ""
        self._index = 0
        self._model_path = model_path
        self._svm_dict = {}

    def SetData(self, train_data, train_labels, test_data, test_labels, class_num):
        self._train_data = train_data
        self._test_data = test_data
        self._train_labels = train_labels
        self._test_labels = test_labels
        self._class_num = class_num
        self._tree_dict = {}

    def LoadModel(self, model_path):
        fb_file_path = model_path + "\\" + BTFILENAME
        fb_file = open(fb_file_path,"r")
        lines = fb_file.readlines()
        self._rootNodeName = lines[0].replace('\n', '')
        for i in range(1, len(lines)):
            tmp = lines[i].split("\t")
            node = Node(tmp[0])
            node._left = tmp[1]
            node._right = tmp[2]
            n = int(tmp[3])
            for j in range(n):
                node._left_label.append(int(float((tmp[4+j]))))
            index = 5 + n
            n = int(tmp[4 + n])
            for j in range(n):
                node._right_label.append(int(float(tmp[index + j])))
            self._tree_dict[node._name] = node
            svmNode = SVMNode()
            svmNode._name = node._name
            svmNode.Load(model_path)
            self._svm_dict[node._name] = svmNode

        self.PrintTree()

    def BuildTree(self):
        # 首先求得每种类型的聚类中心
        group_data = {}
        type_set = set()
        for i in range(len(self._train_data)):
            if not self._train_labels[i] in group_data:
                group_data[self._train_labels[i]] = []
                type_set.add(self._train_labels[i])
            group_data[self._train_labels[i]].append(self._train_data[i])
        self._group_train_data = group_data
        self._type_set = type_set

        fcm_centers = []
        fcm_types = []
        for key in type_set:
            fcm_center, _ = fcm.GetFCM(np.array(group_data[key]), 1)
            fcm_centers.append(fcm_center[0])
            fcm_types.append(key)
        fcm_centers = np.array(fcm_centers)
        print("开始构建二叉树")
        self._rootNodeName = self.buildTree(fcm_centers, fcm_types)

        # 训练二叉树上的每一个节点
        self.trainNodeSVM()

    def InArray(self, a, array):
        for i in range(len(array)):
            if array[i] == a:
                return True
        return False

    def trainNodeSVM(self):
        for nodeName, node in self._tree_dict.items():
            tmp_data = []
            tmp_label = []
            left_count = 0
            right_count = 0
            for i in range(self._train_data.shape[0]):
                if(self.InArray(self._train_labels[i], node._left_label)):
                    tmp_data.append(self._train_data[i])
                    tmp_label.append(-1)
                    left_count += 1
                elif(self.InArray(self._train_labels[i], node._right_label)):
                    tmp_data.append(self._train_data[i])
                    tmp_label.append(1)
                    right_count += 1
            #print("==================={},{}".format(left_count, right_count))
            #exit()
            tmp_data = np.array(tmp_data)
            tmp_label = np.array(tmp_label)
            print("开始训练节点{}".format(nodeName))
            svmNode = SVMNode()
            svmNode.SetData(tmp_data, tmp_label, tmp_data, tmp_label, nodeName)
            svmNode.Train()
            self._svm_dict[nodeName] = svmNode

    def buildTree(self, fcm_centers, type_array):
        print(type_array)
        if(len(fcm_centers) == 0 or len(fcm_centers) == 1):
            return "NULL"
    
        node_name = "node_{}".format(self._index)
        self._index += 1
        node = Node(node_name)

        if(len(fcm_centers) == 2):
            node._left_label.append(type_array[0])
            node._right_label.append(type_array[1])
            self._tree_dict[node_name] = node
            
            return node._name
        _, fcm_labels = fcm.GetFCM(np.array(fcm_centers), 2)        
        leftDataList = []
        rightDataList = []
        leftTypeArray = []
        rightTypeArray = []
        for i in range(len(fcm_labels)):
            if (fcm_labels[i] == 0):
                leftDataList.append(fcm_centers[i])
                leftTypeArray.append(type_array[i])
            else:
                rightDataList.append(fcm_centers[i])
                rightTypeArray.append(type_array[i])
        node._left = self.buildTree(np.array(leftDataList), leftTypeArray)
        node._left_label = leftTypeArray
        node._right = self.buildTree(np.array(rightDataList), rightTypeArray)
        node._right_label = rightTypeArray
        self._tree_dict[node_name] = node
        return node._name

    def PrintTree(self):
        self.printTree(self._rootNodeName)

    def printTree(self, name):
        if(name == "NULL"):
            return
    
        node = self._tree_dict[name]
        print("{}, left->{} [{}], right->{} [{}]".format(name, node._left, node._left_label, node._right, node._right_label))
        self.printTree(node._left)
        self.printTree(node._right)

    def Save(self, path):
        btPath = path + "\\" + BTFILENAME
        # rootNodeName
        # Node1 left right leftArraySize leftArray rightArraySize rightArray
        f_bt = open(btPath, "w")
        f_bt.write(self._rootNodeName + '\n')
        for nodeName, node in self._tree_dict.items():
            f_bt.write(nodeName + '\t')
            f_bt.write(node._left + '\t')
            f_bt.write(node._right + '\t')
            f_bt.write("{}\t".format(len(node._left_label)))
            for i in range(len(node._left_label)):
                f_bt.write("{}\t".format(node._left_label[i]))
            f_bt.write("{}\t".format(len(node._right_label)))
            for i in range(len(node._right_label)):
                f_bt.write("{}\t".format(node._right_label[i]))
            f_bt.write('\n')
            self._svm_dict[nodeName].Save(path)
    
    def Predict(self, data):
        svmNode = self._svm_dict[self._rootNodeName]
        while (1):
            ret = svmNode.Predict(data)
            if ret == -1:
                if(self._tree_dict[svmNode._name]._left == "NULL"):
                    return self._tree_dict[svmNode._name]._left_label[0]
                #print("to {}".format(self._tree_dict[svmNode._name]._left_label))
                svmNode = self._svm_dict[self._tree_dict[svmNode._name]._left]
            else:
                if(self._tree_dict[svmNode._name]._right == "NULL"):
                    return self._tree_dict[svmNode._name]._right_label[0]
                #print("to {}".format(self._tree_dict[svmNode._name]._right_label))
                svmNode = self._svm_dict[self._tree_dict[svmNode._name]._right]
        return -1

if __name__ == "__main__":
    train_images_idx3_ubyte_file = '.\\minist\\train-images-idx3-ubyte\\train-images.idx3-ubyte'
    train_label_idx1_ubyte_file = '.\\minist\\train-labels-idx1-ubyte\\train-labels.idx1-ubyte'
    test_images_idx3_ubyte_file = '.\\minist\\t10k-images-idx3-ubyte\\t10k-images.idx3-ubyte'
    test_label_idx1_ubyte_file = '.\\minist\\t10k-labels-idx1-ubyte\\t10k-labels.idx1-ubyte'

    #test_data = model.normalize(np.array(model.decode_idx3_ubyte(test_images_idx3_ubyte_file)))
    #test_label = np.array(model.decode_idx1_ubyte(test_label_idx1_ubyte_file))

    #btSVM = BTreeSVM()
    #btSVM.LoadModel("D:\\SVM\\BTSVM\\test")


#    train_data = model.normalize(model.decode_idx3_ubyte(train_images_idx3_ubyte_file))
#    train_label = model.decode_idx1_ubyte(train_label_idx1_ubyte_file)
    test_data = model.normalize(np.array(model.decode_idx3_ubyte(test_images_idx3_ubyte_file)))
    test_label = np.array(model.decode_idx1_ubyte(test_label_idx1_ubyte_file))
#    print(len(test_data))
#    print(len(test_label))
    btSVM = BTreeSVM()
#    btSVM.SetData(train_data, train_label, test_data, test_label, 10)
#    btSVM.BuildTree()
#    btSVM.PrintTree()
#    btSVM.Save("D:\\SVM\\BTSVM\\m1")
    btSVM.LoadModel("D:\\SVM\\BTSVM\\m1")
    right = 0
    for i in range(len(test_data)):
    #for i in range(10):
        p = btSVM.Predict(test_data[i])
        if p == int(test_label[i]):
            right += 1
    print("{}/{}".format(right, len(test_data)))
    exit()