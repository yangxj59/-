# -import numpy as np
import matplotlib.pyplot as plt#导入matplotlib库
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
np.random.seed(123)#每次生成相同的随机数
X, y = make_blobs(n_samples=1000, centers=2)#生成两类样本数
fig = plt.figure(figsize=(8,6))#设置尺寸
plt.scatter(X[:,0], X[:,1], c=y)#画散点图
plt.title("Dataset")#设置标题
plt.xlabel("First feature")#设置x轴标签
plt.ylabel("Second feature")#设置y轴标签
plt.show()#显示所画的图
y_true = y[:, np.newaxis]


X_train, X_test, y_train, y_test = train_test_split(X, y_true)
print(f'Shape X_train: {X_train.shape}')#划分出的训练集数据（返回值）
print(f'Shape y_train: {y_train.shape})')#划分出的训练集标签（返回值）
print(f'Shape X_test: {X_test.shape}')#划分出的测试集数据（返回值）
print(f'Shape y_test: {y_test.shape}')#划分出的测试集标签（返回值）


class Perceptron():

    def __init__(self):
        pass

    def train(self, X, y, learning_rate=0.05, n_iters=1000):
        n_samples, n_features = X.shape

        # 初始化参数
        self.weights = np.zeros((n_features,1))#生成一列大小为X.shape的数据
        self.bias = 0

        for i in range(n_iters):
            # 计算分类函数
            a = np.dot(X, self.weights) + self.bias

            # 计算输出
            y_predict = self.step_function(a)

            # 计算权重更新
            delta_w = learning_rate * np.dot(X.T, (y - y_predict))
            delta_b = learning_rate * np.sum(y - y_predict)

            # 更新参数
            self.weights += delta_w
            self.bias += delta_b

        return self.weights, self.bias

    def step_function(self, x):
        return np.array([1 if elem >= 0 else 0 for elem in x])[:, np.newaxis]
        #返回一个列向量,x>0取1；x<0取0

    def predict(self, X):
        a = np.dot(X, self.weights) + self.bias
        return self.step_function(a)

p = Perceptron()
w_trained, b_trained = p.train(X_train, y_train,learning_rate=0.05, n_iters=1000)
#返回self.weights, self.bias


def plot_hyperplane(X, y, weights, bias):
  #画出决策超平面（在二维是一条直线）  
    slope = - weights[0]/weights[1]
    intercept = - bias/weights[1]
    x_hyperplane = np.linspace(-10,10,10)
    y_hyperplane = slope * x_hyperplane + intercept
    fig = plt.figure(figsize=(8,6))
    plt.scatter(X[:,0], X[:,1], c=y)
    plt.plot(x_hyperplane, y_hyperplane, '-')
    plt.title("Dataset and fitted decision hyperplane")
    plt.xlabel("First feature")
    plt.ylabel("Second feature")
    plt.show()

plot_hyperplane(X, y, w_trained, b_trained)
