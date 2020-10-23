import numpy as np
import struct
import matplotlib.pyplot as plt
from skimage import morphology,draw

from sklearn import svm
import time

import pickle
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

train_images_idx3_ubyte_file = '.\\minist\\train-images-idx3-ubyte\\train-images.idx3-ubyte'
train_label_idx1_ubyte_file = '.\\minist\\train-labels-idx1-ubyte\\train-labels.idx1-ubyte'
test_images_idx3_ubyte_file = '.\\minist\\t10k-images-idx3-ubyte\\t10k-images.idx3-ubyte'
test_label_idx1_ubyte_file = '.\\minist\\t10k-labels-idx1-ubyte\\t10k-labels.idx1-ubyte'

modelName = "model_formal.txt"

def decode_idx3_ubyte(inputFile):
    binaryData = open(inputFile, 'rb').read()
    offest = 0
    head = '>iiii'
    _, imageNum, imageRow, imageCol = struct.unpack_from(head, binaryData, offest)
    print("读取文件：{}， 图片数：{}，规格：{}*{}".format(inputFile, imageNum, imageRow, imageCol))
    iamgeSize = imageCol * imageRow
    offest += struct.calcsize(head)
    imageFmt = '>' + str(iamgeSize) + 'B'
    images = np.empty((imageNum, imageRow * imageCol))
    for i in range(imageNum):
        tmp = np.array(struct.unpack_from(imageFmt, binaryData, offest))
        #images[i] = np.array(tmp.reshape(imageRow, imageCol))
        images[i] = np.array(tmp)
        offest += struct.calcsize(imageFmt)
    print('Load image done')
    return images

def decode_idx1_ubyte(inputFile):
    binaryData = open(inputFile, 'rb').read()
    head = '>ii'
    offset = 0
    _, labelNum = struct.unpack_from(head, binaryData, offset)
    print('读取文件：{}， 标签数：{}'.format(inputFile, labelNum))
    labelHead = '>B'
    offset += struct.calcsize(head)
    labels = np.empty(labelNum)
    for i in range(labelNum):
        labels[i] = struct.unpack_from(labelHead, binaryData, offset)[0]
        offset += struct.calcsize(labelHead)
    return labels

def getFrame(data):
    imageNum = data.shape[0]
    for i in range(imageNum):
        tmp = data[i].reshape(28,28)
        tmp = morphology.skeletonize(tmp)
        #plt.imshow(tmp)
        #plt.show()
        data[i] = tmp.reshape(28*28)
    return data

def conv(data):
    mask = np.zeros((3, 3))
    mask[0][0] = mask[0][2] = mask[1][1] = mask[2][0] = mask[2][2] = 1
    images = np.empty((data.shape[0], 26*26))
    for i in range(data.shape[0]):
        tmp = data[i]
        #tmp.reshape(28, 28)
        res = np.zeros((26, 26))
        for j in range(26):
            for k in range(26):
                for ii in range(3):
                    for jj in range(3):
                        res[j][k] += tmp[(j+ii)*26 + k+jj]*mask[ii][jj]
        for j in range(26):
            for k in range(26):
                if res[j][k] != 0:
                    res[j][k] = 1
        images[i] = res.reshape((1, 26 * 26))
    return images

def normalize(data):
    for i in range(data.shape[0]):
        for j in range(data[i].shape[0]):
            if data[i][j]!=0:
                data[i][j]=1
    return data

''' TODO
    提取图片特征
'''

def getPartionFeature(img):
    features = np.zeros(49)
    for i in range(7):
        for j in range(7):
            tmp = 0 
            for n in range(4):
                for m in range(4):
                    if(img[i*4+n][j*4+m]!=0):
                        tmp += 1
            features[i*7 + j] = tmp
    return features

def getProjestionFeature(img):
    '''
         ---1-------2-----
        8|   A    |   B    |
         |       9|        |3
         --- 11---|-- -12--|
        7|      10|        |4
         |  C     |   D    |
         ------------------
             6         5

         --------------- 1
        |       |       |
        |       |       |
        |--- ---|-- ----|2
        |       |       |
        |       |       |
        ---------------- 3
        4       5       6
    '''
    lines = np.zeros((6,28))
    for i in range(14):
        for j in range(28):
            if(img[i][j] != 0):
                lines[0][j] = 1
                lines[1][j] = 1
            if(img[i+14][j]!=0):
                lines[1][j] = 1
                lines[2][j] = 1
    for i in range(28):
        for j in range(14):
            if(img[i][j] != 0):
                lines[3][i] = 1
                lines[4][i] = 1
            if(img[i][j+14] != 0):
                lines[4][i] = 1
                lines[5][i] = 1
    features = np.zeros(12)
    for i in range(6):
        partA = 0
        partB = 0
        for j in range(14):
            if(lines[i][j] == 1):
                partA += 1
            if(lines[i][j+14] == 1):
                partB += 1
            features[i*2] = partA
            features[i*2 + 1] = partB
    return features

def getFeature(data):
    size = data.shape[0]
    # 区域划分像素统计特征
    #   28 * 28 按照 4 * 4 分成 49 个区域 统计每个区域的像素个数
    # 投影特征
    #   将图片分成四块 每一块投影到相应的边 共12条边
    newData = np.zeros((size, 28*28 + 49 + 12))
    for i in range(size):
        print(i)
        img = data[0].reshape(28,28)
        features = np.zeros((49 + 12))
        partionFeatures = getPartionFeature(img)
        projestionFeatures = getProjestionFeature(img)
        for j in range(49):
            features[j] = partionFeatures[j]
        for j in range(12):
            features[j+49] = projestionFeatures[j]
        #newData[i] = features
        newData[i] = np.append(data[i], features)
    return newData

def reNormalize(data):
    for i in range(data.shape[0]):
        for j in range(data[i].shape[0]):
            if data[i][j]==0:
                data[i][j]=1
            else:
                data[i][j]=0
    return data

def svmTrain():
    clf = svm.SVC(C=1.0, kernel='rbf', gamma=0.03, verbose=1)
    trainImages = normalize(decode_idx3_ubyte(train_images_idx3_ubyte_file))
    #trainImages = conv(normalize(decode_idx3_ubyte(train_images_idx3_ubyte_file)))
    trainLabels = decode_idx1_ubyte(train_label_idx1_ubyte_file)
    print(trainImages.shape)
    print('开始训练')
    clf = svm.SVC(C=100.0, kernel='rbf', gamma=0.03, verbose=1)
    clf.fit(trainImages, trainLabels)
    save = pickle.dumps(clf)
    f = open(modelName, 'wb')
    f.write(save)
    f.close()
    print('训练结束')
    return clf

def showTrainImage():
    trainImages = decode_idx3_ubyte(train_images_idx3_ubyte_file)
    trainLabels = decode_idx1_ubyte(train_label_idx1_ubyte_file)
    for i in range(100):
        tmp = trainImages[i].reshape(28, 28)
        plt.imshow(tmp)
        #plt.show()
        plt.savefig(".\\images\\{}_{}.png".format(int(trainLabels[i]), i))

def svmEvaluator():
    clf = loadModel()
    testImages = normalize(decode_idx3_ubyte(test_images_idx3_ubyte_file))
    testLabels = decode_idx1_ubyte(test_label_idx1_ubyte_file)

    right = 0
    for i in range(10000):
        tmp = clf.predict([testImages[i,:]])
        #print((tmp, testLabels[i]))
        if int(tmp[0]) == int(testLabels[i]):
            right += 1
    print(right * 1.0 / 10000)

def loadModel():
    data = open(modelName, 'rb')
    tmp = data.read()
    model = pickle.loads(tmp)
    return model

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


    # 初始化偏置向量
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#def predictByCNNSVM(data):
#    x = tf.placeholder("float", [None, 256])
#    y_ = tf.placeholder("float", [None, 10])
#    # 设置输入层的W和b
#    W = tf.Variable(tf.zeros([256, 10]))
#    b = tf.Variable(tf.zeros([10]))
#    # 计算输出，采用的函数是softmax（输入的时候是one hot编码）
#    y = tf.nn.softmax(tf.matmul(x, W) + b)
#    # 第一个卷积层，5x5的卷积核，输出向量是32维
#    w_conv1 = weight_variable([5, 5, 1, 32])
#    b_conv1 = bias_variable([32])
#    x_image = tf.reshape(x, [-1, 16, 16, 1])
#    # 图片大小是16*16，,-1代表其他维数自适应
#    h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
#    h_pool1 = max_pool_2x2(h_conv1)
#    # 采用的最大池化，因为都是1和0，平均池化没有什么意义
#
#    # 第二层卷积层，输入向量是32维，输出64维，还是5x5的卷积核
#    w_conv2 = weight_variable([5, 5, 32, 64])
#    b_conv2 = bias_variable([64])
#    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
#    h_pool2 = max_pool_2x2(h_conv2)
#    w_fc1 = weight_variable([4 * 4 * 64, 256])
#    b_fc1 = bias_variable([256])
#    h_pool2_flat = tf.reshape(h_pool2, [-1, 4 * 4 * 64])
#    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

#def loadCNNSVM():
#    saver = tf.train.Saver()
#    sess = tf.InteractiveSession()
#    saver.restore(sess, ".\\CNN+SVM\\CNN-SVM\\model.ckpt")
#    data = open(".\\CNN+SVM\\CNN-SVM\\model.svm")
#    tmp = data.read()
#    model = pickle.loads(tmp)
#    return sess, model

if __name__ == '__main__':
    #showTrainImage()
    svmTrain()
    svmEvaluator()