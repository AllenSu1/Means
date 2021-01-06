from sklearn import cluster, datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

df = pd.read_excel('120_data.xlsx', header=None)
X = df.to_numpy()

#開始計算kmeans時間
start = time.time()

# KMeans 演算法
kmeans_fit = cluster.KMeans(n_clusters = 3).fit(X)

# 印出分群結果
cluster_labels = kmeans_fit.labels_
kmeans_centers = kmeans_fit.cluster_centers_
# print("分群結果：")
# print(cluster_labels)
# print("---")
print('kmeans_centers = \n', kmeans_centers)

# 結束測量
end = time.time()
print('執行時間：%f 秒 ' %(end-start))

# plot
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[:, 0], X[:, 1], X[:, 2] , c=cluster_labels, cmap='Set1')
ax.scatter(kmeans_centers[:, 0], kmeans_centers[:, 1], kmeans_centers[:, 2] , s=200 , c='rgb' , alpha=1,cmap='Set1', marker='*' )

plt.show()