import numpy as np
from sklearn.model_selection import train_test_split # it is not useful to define my own function for dataset splitting
import matplotlib.pyplot as plt

lr = 0.001 # learning rate
data_path = '../assets/data/iris-3.txt' # for multi-classification, take 3 as emample
param_path = '../assets/params/sof_reg.txt'
types = {b'Iris-setosa': 0, b'Iris-versicolor': 1, b'Iris-virginica': 2}  # labels, as byte
iter_num = 10000
threshold = 0.5


def load_dataset(data_path):
    data_with_label = np.loadtxt(
        fname=data_path,
        delimiter=',',
        dtype=float,
        converters={4: lambda s: types[s]}) # convert the 5-th cols to numerical labels
    datas, labels = np.split(data_with_label, (4,), axis=1)
    # add an all-one column for bias, because decision boundary is w0*1.0+w1*x1+w2*x2+w3*x3+w4*x4=0 (for 2-classification)
    datas = np.c_[np.ones(datas.shape[0], dtype=float), datas]

    labels = labels.ravel() # convert [ [0],...,[0],[1],...[1],[2],...,[2]] to [0,...,0,1,...,1,2,...,2]
    # shape : x_train(emp_train_num,5), x_test(emp_test_num, 5), y_train(emp_train_num,), y_test(emp_test_num,)
    x_train, x_test, y_train, y_test = train_test_split(datas, labels, test_size=0.3, random_state=None)
    return x_train, x_test, y_train, y_test


# return shape: h(3, emp_num)
def softmax(x): # shape: x(3,emp_num)
    return np.exp(x)/np.matmul(np.ones(len(types)).T, np.exp(x))


# return shape: y(emp_num, 3)
def convert_to_one_hot(x, k): # shape: x(emp_num,)
    return np.eye(k)[x]


# all gradient descent
def grad_descent(x_train, y_train): # shape: x_train(emp_train_num, 5)
    weights = np.ones(shape=(x_train.shape[1], len(types)))  # shape: weigths(5,3)
    for i in range(iter_num):
        # y = [P1; P2; P3] = exp(W^Tx)/\sum{exp(W^Tx)} where x is one instance
        y = softmax(np.matmul(x_train, weights).T).T # shape: y(emp_train_num, 3)
        residual = y - y_train # shape: residual(emp_train_num, 3)
        #  formula of gradient for cross entropy loss, proof see https://www.cnblogs.com/Luv-GEM/p/10674719.html
        # shape: (5, emp_train_num) * (emp_train_num, 3) = (5, 3)
        gradient = np.matmul(x_train.T, residual)/x_train.shape[0]
        weights = weights - lr * gradient
    np.savetxt(param_path, weights)
    return weights


# stochastic gradient descent
def stoc_grad_descent(x_train, y_train):
    weights = np.zeros(shape=(x_train.shape[1]))  # shape: weigths(5, 3)
    for i in range(iter_num):
        index = np.random.randint(0, x_train.shape[0])
        x = x_train[index] # shape: x(5,)
        y = softmax(np.matmul(weights.T, x))
        residual = y - y_train[index] #  shape: residual(3,)
        #  formula of gradient for cross entropy loss, proof see https://www.cnblogs.com/Luv-GEM/p/10674719.html
        gradient = x * residual.T  # shape: (5,) * (3,)T = (5,3)
        weights = weights - lr * gradient
    np.savetxt(param_path, weights)
    return weights


def test(x_test, y_test, weights):
    # y = [P1; P2; P3] = exp(W^Tx)/\sum{exp(W^Tx)} where x is one instance
    y_sig = softmax(np.matmul(x_test, weights).T).T  # shape: y_sig(emp_test_num, 3)
    y_pred = np.argmax(y_sig, axis=1) # shape: y_pred(emp_test_num,)
    tpfn = (y_pred == y_test) # TP & FN = 1
    tpfn_num = np.linalg.norm(tpfn, ord=0) # 0-norm to count #non-zero-element
    return tpfn_num / y_test.shape[0]


def inference(x, weights):
    y_sig = softmax(np.matmul(weights.T, x))
    y_pred = np.argmax(y_sig)
    return y_pred


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = load_dataset(data_path)
    y_train = convert_to_one_hot(y_train.astype(int), len(types))
    weights = grad_descent(x_train, y_train)
    # weights = stoc_grad_descent(x_train, y_train)
    # weights = np.loadtxt(param_path)
    accuracy = test(x_test, y_test, weights)
    print('3-classification accuracy is %f' % accuracy)
    print('one example : %s@%s' % (inference(x_test[0],weights), y_test[0]))