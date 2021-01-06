import numpy as np
from fcmeans import FCM
from matplotlib import pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import time

n_samples = 3000

df = pd.read_excel("120_data.xlsx", header=None)
X = df.to_numpy()

#開始計算FCM時間
start = time.time()

#FCM分群
fcm = FCM(n_clusters=3)
fcm.fit(X)

# outputs
fcm_centers = fcm.centers
fcm_labels = fcm.predict(X)

#印群心
print('fcm_centers = \n', fcm_centers)

# 結束測量
end = time.time()
print("執行時間：%f 秒" %(end-start))

# plot result
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=fcm_labels, cmap='Set1')
ax.scatter(fcm_centers[:, 0], fcm_centers[:, 1], fcm_centers[:, 2],  s=200 , c='rgb' , alpha=1,cmap='Set1', marker='*')
plt.show()
