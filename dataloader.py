import numpy as np
import struct

train_images_idx3_ubyte_file = '.\\minist\\train-images-idx3-ubyte\\train-images.idx3-ubyte'
train_label_idx1_ubyte_file = '.\\minist\\train-labels-idx1-ubyte\\train-labels.idx1-ubyte'
test_images_idx3_ubyte_file = '.\\minist\\t10k-images-idx3-ubyte\\t10k-images.idx3-ubyte'
test_label_idx1_ubyte_file = '.\\minist\\t10k-labels-idx1-ubyte\\t10k-labels.idx1-ubyte'

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

def loadTrainData():
    images = decode_idx3_ubyte(train_images_idx3_ubyte_file)
    label = decode_idx1_ubyte(train_label_idx1_ubyte_file)
    return images, label

def loadTestData():
    images = decode_idx3_ubyte(test_images_idx3_ubyte_file)
    label = decode_idx1_ubyte(test_label_idx1_ubyte_file)
    return images, label