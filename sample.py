import numpy as np

w = 3  # 初期の重み
alpha = 0.0002  # 学習率
epoch = 30  # データの学習回数

# 仮説関数
def hypothesis(x):
    return w * x


# 勾配を計算する
def gradient(X, Y):
    sum = 0
    for x, y in zip(X, Y):
        sum += (hypothesis(x) - y) * x
    return 1 / len(X) * sum


# 重み更新時の値を計算する
def calc_weight(X, Y, w, alpha):
    return w - alpha * gradient(X, Y)


# 年齢:年収のデータ
data = np.array([
    [20, 268],
    [21, 272],
    [22, 277],
    [23, 292],
    [24, 320],
    [25, 344],
    [26, 363],
    [27, 377],
    [28, 390],
    [29, 404],
    [30, 414],
    [31, 429],
    [32, 439],
    [33, 451],
    [34, 459],
    [35, 465],
    [36, 474],
    [37, 481],
    [38, 489],
    [39, 494],
    [40, 505],
    [20, 222],
    [21, 242],
    [22, 262],
    [23, 281],
    [24, 301],
    [25, 321],
    [26, 341],
    [27, 361],
    [28, 370],
    [29, 379],
    [30, 388],
    [31, 397],
    [32, 407],
    [33, 414],
    [34, 421],
    [35, 428],
    [36, 435],
    [37, 442],
    [38, 447],
    [39, 452],
    [40, 457]
])

# 学習させる
for i in range(epoch):
    w = calc_weight(data[:, 0], data[:, 1], w, alpha)
    print('epoch: %d, weight: %f' % (i+1, w))

# 42歳の年収を予測させる
age = 42
income = hypothesis(age)
print('age: %d, income: %d' % (age, income))

