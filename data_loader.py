import struct
import numpy as np      # 支持大量的维度数组与矩阵运算
import matplotlib.pyplot as plt     # Python的绘图库

train_images_path = "E:\\zxy\\svm_learn\\minist\\train-images-idx3-ubyte\\train-images.idx3-ubyte"
train_labels_path = "E:\\zxy\\svm_learn\\minist\\train-labels-idx1-ubyte\\train-labels.idx1-ubyte"
test_images_path = "E:\\zxy\\svm_learn\\minist\\t10k-images-idx3-ubyte\\t10k-images.idx3-ubyte"
test_labels_path = "E:\\zxy\\svm_learn\\minist\\t10k-labels-idx1-ubyte\\t10k-labels.idx1-ubyte"

def read_images(inputFile):
    binaryData = open(inputFile, 'rb').read()
    offest = 0
    head = '>iiii'
    _, imageNum, imageRow, imageCol = struct.unpack_from(head, binaryData, offest)
    print("读取文件：{}， 图片数：{}，规格：{}*{}".format(inputFile, imageNum, imageRow, imageCol))

    iamgeSize = imageCol * imageRow  # 28*28 = 784
    offest += struct.calcsize(head)
    imageFmt = '>' + str(iamgeSize) + 'B'  # >784B
    # imageFmt = ">784B"
    images = np.empty((imageNum, imageRow * imageCol))
    for i in range(imageNum):
        image = struct.unpack_from(imageFmt, binaryData, offest)
        tmp = np.array(image)  # 1*784
        #plt.imshow(tmp.reshape((28,28)), cmap='binary')
        #plt.show()
        #tmp = tmp.reshape((imageRow, imageCol)) # 28*28
        images[i] = tmp
        offest += struct.calcsize(imageFmt)
    print('Load image done')
    return images

def read_labels(inputFile):
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

def ReadTrainImages():
    return read_images(train_images_path)

def ReadTrainLabels():
    return read_labels(train_labels_path)

def ReadTestImages():
    return read_images(test_images_path)

def ReadTestLabels():
    return read_labels(test_labels_path)

if __name__ == "__main__":
    train_images = ReadTrainImages()
    train_labels = ReadTrainLabels()
    for i in range(60000):
        print("这个图片里的数字是{}".format(train_labels[i]))
        plt.imshow(np.array(train_images[i]).reshape((28,28)), cmap="binary")
        plt.show()