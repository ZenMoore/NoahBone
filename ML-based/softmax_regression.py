import numpy as np
from sklearn.model_selection import train_test_split # it is not useful to define my own function for dataset splitting
import matplotlib.pyplot as plt

lr = 0.001 # learning rate
data_path = '../assets/data/iris-3.txt' # for multi-classification, take 3 as emample
param_path = '../assets/params/sof_reg.txt'
types = {b'Iris-setosa': 0, b'Iris-versicolor': 1, b'Iris-virginica': 2}  # labels, as byte
iter_num = 1000000
threshold = 0.5


def load_dataset(data_path):
    data_with_label = np.loadtxt(
        fname=data_path,
        delimiter=',',
        dtype=float,
        converters={4: lambda s: types[s]}) # convert the 5-th cols to numerical labels
    datas, labels = np.split(data_with_label, (4,), axis=1)
    # add an all-one column for bias, because decision boundary is w0*1.0+w1*x1+w2*x2+w3*x3+w4*x4=0
    datas = np.c_[np.ones(datas.shape[0], dtype=float), datas]

    labels = labels.ravel() # convert [ [0],...,[0],[1],...[1],[2],...,[2]] to [0,...,0,1,...,1,2,...,2]
    # shape : x_train(emp_train_num,5), x_test(emp_test_num, 5), y_train(emp_train_num,), y_test(emp_test_num,)
    x_train, x_test, y_train, y_test = train_test_split(datas, labels, test_size=0.3, random_state=None)
    return x_train, x_test, y_train, y_test

# logistic function, sigmoid function as default
def logistic(x, L=1, k=1, x0=0):
    return float(L)/(1+np.exp(-k*(x - x0)))


def sigmoid(x):
    return logistic(x, 1, 1, 0)


# all gradient descent
def grad_descent(x_train, y_train):
    weights = np.ones(shape=(x_train.shape[1]))  # shape: weigths(5,)
    for i in range(iter_num):
        y = sigmoid(np.matmul(x_train, weights))  # P(y=1|X;\theta) = g(\theta^T*x), shape: y(emp_train_num,)
        residual = y - y_train
        #  formula of gradient for cross entropy loss, proof see https://zhuanlan.zhihu.com/p/28415991
        gradient = np.matmul(x_train.T, residual)/x_train.shape[0] # shape: (5, emp_train_num) * (emp_train_num,) = (5,)
        weights = weights - lr * gradient
    np.savetxt(param_path, weights)
    return weights


# stochastic gradient descent
def stoc_grad_descent(x_train, y_train):
    weights = np.zeros(shape=(x_train.shape[1]))  # shape: weigths(5,)
    for i in range(iter_num):
        index = np.random.randint(0, x_train.shape[0])
        x = x_train[index] # shape: x(5,)
        y = sigmoid(np.matmul(weights.T, x))
        residual = y - y_train[index]
        #  formula of gradient for cross entropy loss, proof see https://zhuanlan.zhihu.com/p/28415991
        gradient = x * residual  # shape: (5,) * (5,) = (5,)
        weights = weights - lr * gradient
    np.savetxt(param_path, weights)
    return weights


def test(x_test, y_test, weights):
    y_sig = sigmoid(np.matmul(x_test, weights))  # P(y=1|X;\theta) = g(\theta^T*x), shape: y(emp_test_num,)
    y_pred = np.array([1 if y_sig[i] >= threshold else 0 for i in range(y_test.shape[0])])  # 1 if y_sig >= threshold else 0
    tpfn = (y_pred == y_test) # TP & FN = 1
    tpfn_num = np.linalg.norm(tpfn, ord=0) # 0-norm to count #non-zero-element
    return tpfn_num / y_test.shape[0]


def inference(x, weights):
    y_sig = sigmoid(np.matmul(x.T, weights))
    y_pred = 1 if y_sig > threshold else 0
    return y_pred


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = load_dataset(data_path)
    weights = grad_descent(x_train, y_train)
    # weights = stoc_grad_descent(x_train, y_train)
    accuracy = test(x_test, y_test, weights)
    print('2-classification accuracy is %f' % accuracy)
    print('one example : %s@%s' % (inference(x_test[0],weights), y_test[0]))
    # visualize(x_test, y_test, weights)


