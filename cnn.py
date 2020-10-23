import random
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from sklearn import svm
import dataloader
train_images_idx3_ubyte_file = '.\\minist\\train-images-idx3-ubyte\\train-images.idx3-ubyte'
train_label_idx1_ubyte_file = '.\\minist\\train-labels-idx1-ubyte\\train-labels.idx1-ubyte'
test_images_idx3_ubyte_file = '.\\minist\\t10k-images-idx3-ubyte\\t10k-images.idx3-ubyte'
test_label_idx1_ubyte_file = '.\\minist\\t10k-labels-idx1-ubyte\\t10k-labels.idx1-ubyte'

TESTDATANAME = "test"
TRAINDATANAME = "train"
MODELPATHROOT = '.\\cnnModel'
MODELPATH = MODELPATHROOT + "\\svmModel"


def load_data(dataTyep = TRAINDATANAME):
    def normilzed(data):
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if(data[i][j]!=0):
                    data[i][j] = 1
                else:
                    data[i][j] = 0
        return data

    img = None
    label = None
    if(dataTyep == TRAINDATANAME):
        img, label = dataloader.loadTrainData()
    elif(dataTyep == TESTDATANAME):
        img, label = dataloader.loadTestData()
    else:
        return None
    # 将数据格式转换成CNN的输入格式
    # 图片：二值图，28 * 28 float
    # 标签：onehost
    img = normilzed(img)
    img = img.astype("float32")
    #img = img.reshape((-1,28,28,1))
    '''
    onehostLabel = np.zeros((label.shape[0], 10))
    for i in range(label.shape[0]):
        onehostLabel[label[i]-1] = 1
    onehostLabel = onehostLabel.astype("int64")
    '''
    return img, label.astype("int64")


def DefineModel(x):
    # x 是传入的数据 placeholder float [None, 28*28]
    x_image = tf.reshape(x,[-1, 28, 28, 1])
    # 初始化权重向量
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
    
    # 初始化偏置向量
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    # 二维卷积运算，步长为1，输出大小不变
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    # 池化运算，将卷积特征缩小为1/2
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 第一层卷积
    with tf.variable_scope("conv1"):
        weitghts = weight_variable([5, 5, 1 ,32]) #5x5大小，1通道，32个不同的卷积核
        biases = bias_variable([32])
        conv1 = tf.nn.relu(conv2d(x_image, weitghts) + biases)
        pool1 = max_pool_2x2(conv1) # b*28*28*32 -> b*14*14*32
    
    # 第二层卷积
    with tf.variable_scope("conv2"):
        weitghts = weight_variable([5, 5, 32, 64])
        biases = bias_variable([64])
        conv2 = tf.nn.relu(conv2d(pool1, weitghts) + biases)
        pool2 = max_pool_2x2(conv2) # b*7*7*64

    # 全连接层 full connect level
    with tf.variable_scope("fc1"):
        weitghts = weight_variable([7 * 7 * 64, 256])
        biases = bias_variable([256])
        fc1_flat = tf.reshape(pool2, [-1, 7*7*64])
        fc1 = tf.nn.relu(tf.matmul(fc1_flat, weitghts) + biases)
        fc1_drop = tf.nn.dropout(fc1, 0.5)  # 防止过拟合

    with tf.variable_scope("fc2"):
        weitghts = weight_variable([256, 10])
        biases = bias_variable([10])
        fc2 = tf.matmul(fc1_drop, weitghts) + biases

    return fc2

def TrainCNN():
    x = tf.placeholder(tf.float32, shape = [None, 28*28], name = "input")
    y_ = tf.placeholder("int64", shape = [None], name = "y_")
    initial_learning_rate = 0.001
    y_fc2 = DefineModel(x)
    y_label = tf.one_hot(y_, 10, name="y_labels")
    # 损失函数
    loss_temp = tf.losses.softmax_cross_entropy(onehot_labels=y_label, logits=y_fc2)
    cross_entropy_loss = tf.reduce_mean(loss_temp)

    # 训练时的优化器
    train_step = tf.train.AdamOptimizer(learning_rate=initial_learning_rate,
                                        beta1=0.9,
                                        beta2=0.999,
                                        epsilon=1e-08).minimize(cross_entropy_loss)
    correst_prediction = tf.equal(tf.argmax(y_fc2, 1), tf.argmax(y_label, 1))
    accuracy = tf.reduce_mean(tf.cast(correst_prediction, tf.float32))
    saver = tf.train.Saver()
    tf.add_to_collection("predict", y_fc2)
    tf.add_to_collection("acc", accuracy)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("开始训练")
        imgX, labelY = load_data(TRAINDATANAME)

        # 全量数据会OOM
        # 每次随机选择50个做训练
        for epoch in range(5000):
            randImgs = []
            randLabels = []
            p = random.sample(range(60000), 100)
            for k in p:
                randImgs.append(imgX[k])
                randLabels.append(labelY[k])
            print("迭代第{}次".format(epoch))
            if epoch % 10 == 0:
                train_accuracy = accuracy.eval(feed_dict={x:randImgs, y_:randLabels})
                train_loss = cross_entropy_loss.eval(feed_dict={x:randImgs, y_:randLabels})
                print("acc={}".format(train_accuracy))
                print("loss={}".format(train_loss))
            train_step.run(feed_dict={x: randImgs, y_ : randLabels})
        saver.save(sess, MODELPATH)
'''
def LoadModel():
    imgX, labelY = load_data(TESTDATANAME)
    for i in range(10):
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(MODELPATH+'.meta')
            saver.restore(sess, tf.train.latest_checkpoint(MODELPATHROOT))
            graph = tf.get_default_graph()
            x = graph.get_tensor_by_name("input:0")
            randImgs = []
            randLabels = []
            p = random.sample(range(10000), 1)
            for k in p:
                randImgs.append(imgX[k])
                randLabels.append(labelY[k])
            feed_dict = {"input:0":randImgs, "y_:0":randLabels}
            pred_y = tf.get_collection("predict")
            pred = sess.run(pred_y, feed_dict={"input:0":randImgs})[0]
            pred = sess.run(tf.argmax(pred,1))
            print("pred",pred,"\n")
            print(randLabels)
'''
def LoadModel():
    sess = tf.Session()
    saver = tf.train.import_meta_graph(MODELPATH+'.meta')
    saver.restore(sess, tf.train.latest_checkpoint(MODELPATHROOT))
    return sess

class CNNPredictor:
    def __init__(self):
        self.sess = LoadModel()
    def Predict(self, img):
        img.reshape(28*28)
        tmp = []
        tmp.append(img)
        pred_y = tf.get_collection("predict")
        pred = self.sess.run(pred_y, feed_dict={"input:0":tmp})[0]
        pred = self.sess.run(tf.argmax(pred,1))[0]
        return pred

if __name__ == "__main__":
    TrainCNN()
    #predictor = CNNPredictor()
    #imgX, labelY = load_data(TESTDATANAME)
    #print(predictor.Predict(imgX[0]))