from sklearn import svm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# 获取鸢尾花数据集
def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, [0,1,-1]])
    return data[:,:2], data[:,-1]
x, y = create_data()

# svm设置，调用了Python中的SVM库，自带的优化算法为SMO优化算法
# 这里应用了线性核函数，误差项惩罚参数C取了很大的值，为了实现硬间隔
clf = svm.SVC(C=100000,kernel='linear')
clf.fit(x, y)

# 获取w
w = clf.coef_[0]
a = -w[0] / w[1]  # 斜率
# 画图划线
xx = np.linspace(4, 7.5)  # (-5,5)之间x的值
yy = a * xx - (clf.intercept_[0]) / w[1]  # xx带入y，截距

# 画出与点相切的线，也就是支持向量
b = clf.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
b = clf.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])

# 画出支持向量和
plt.figure(figsize=(8, 4))
plt.plot(xx, yy)           # 画出分类平面
plt.plot(xx, yy_down)      # 画出支持向量
plt.plot(xx, yy_up)
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100)  # 标记支持向量点
plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Paired) # 画出样本点
plt.show()