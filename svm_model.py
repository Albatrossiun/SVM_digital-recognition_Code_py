import numpy as np
import data_loader as dl
import matplotlib.pyplot as plt
from sklearn import svm

def Normalize(data):
    for i in range(data.shape[0]):
        for j in range(data[i].shape[0]):       #   data[i][j] i指60000里面的第i个 j指第i个元素的第j个元素
            if data[i][j] != 0:                 #   (60000,784) 60000个数据 每个数据都是784这么大 j=4，4是784里面的784
                data[i][j] = 1
    return data

if __name__ == "__main__":
    mySVM = svm.SVC(C=1.0, kernel='rbf', gamma=0.03, verbose=1)
    train_images = Normalize(dl.ReadTrainImages())
    train_labels = dl.ReadTrainLabels()
    test_images = Normalize(dl.ReadTestImages())
    test_labels = dl.ReadTestLabels()

    small_data_set = np.empty((5000, 28*28))
    small_data_set_label = np.empty(5000)
    for i in range(5000):
        small_data_set[i] = train_images[i*10]
        small_data_set_label[i] = train_labels[i*10]

    #mySVM.fit(train_images, train_labels)
    mySVM.fit(small_data_set, small_data_set_label)

    for i in range(10000):
        print("当前的图片标签是{}".format(test_labels[i]))
        #plt.imshow(test_images[i].reshape((28,28)), cmap="binary")
        #plt.show()
        res = mySVM.predict(test_images[i].reshape((1,784)))
        print("模型识别的结果为{}".format(res))
        #_ = input("继续")