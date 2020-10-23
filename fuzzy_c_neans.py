from fcmeans import FCM
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
from seaborn import scatterplot as scatter

# 需要聚类的数据集 以及聚类中心数
# 返回值1：聚类中心  返回值2：
def GetFCM(data, classNum):
    fcm = FCM(n_clusters=classNum)
    fcm.fit(data)
    return fcm.centers, fcm.u.argmax(axis=1)
    

if __name__ == "__main__":
    # create artifitial dataset
    n_samples = 50000
    n_bins = 3  # use 3 bins for calibration_curve as we have 3 clusters here
    centers = [(-5, -5), (0, 0), (5, 5)]

    X,_ = make_blobs(n_samples=n_samples, n_features=3, cluster_std=1.0,
                      centers=centers, shuffle=False, random_state=42)

    # fit the fuzzy-c-means
    fcm = FCM(n_clusters=3)
    fcm.fit(X)

    # outputs
    fcm_centers = fcm.centers
    fcm_labels  = fcm.u.argmax(axis=1)
    print(len(X))
    print(len(fcm_labels))

    # plot result
    
    f, axes = plt.subplots(1, 2, figsize=(11,5))
    scatter(X[:,0], X[:,1], ax=axes[0])
    scatter(X[:,0], X[:,1], ax=axes[1], hue=fcm_labels)
    scatter(fcm_centers[:,0], fcm_centers[:,1], ax=axes[1],marker="s",s=200)
    plt.show()