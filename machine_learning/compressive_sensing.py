import numpy as np
import math
from PIL import Image


'''
Configuration
'''
img_size = 256
'''
Fondamental math-objs
'''


def sense(source, mode='gaussian', sampling_rate=0.7):
	mat_mesuring = np.zeros((img_size, img_size))
	img_mesured = np.zeros((int(img_size * sampling_rate), img_size))
	if mode == 'gaussian':  # gaussian random sensing
		mat_mesuring = np.random.randn(img_size, img_size)  # gaussian random mesuring matrix
		u, s, vh = np.linalg.svd(mat_mesuring)
		mat_mesuring = u[:int(img_size*sampling_rate),] # orthogonalization
		img_mesured = np.dot(mat_mesuring, source)  # the output after mesuring
	return mat_mesuring, img_mesured


def gen_sparse_base(mode='DCT'):
	sparse_base = np.zeros((img_size, img_size))
	if mode == 'DCT':  # discrete cosine transform
		v = range(img_size)
		for k in range(img_size):
			dct = np.cos(np.dot(v, k * math.pi / img_size))
			if k > 0:
				dct = dct - np.mean(dct)
			sparse_base[:, k] = dct/np.linalg.norm(dct)
	return sparse_base

'''
Algorithms
	for compressive sensing, there are overall six groups of algos:
	- combinatorial : Random Fourier Sampling, HHS, chaining pursuits...
	- non-convex optim(v) : FOCUSS, IRLS(v)...
	- convex optim :GPSR, Bregman, Fix Point Continuity, Interior Point Method, Simplex Algorithm...
	- bayesian : MAP, MLE, BCS...
	- greedy
		o serial(v) : MP, OMP(v), GP...
		o parallel : CoSaMP, SP...
	- threshold(v) : IHT(v), IST, AMP...
So many, yah... therefore, I chose a few which are marked by v : IRLS, OMP, IHT
'''

# OMP : a serial greedy algorithm
# y : img_mesured; D : sensing matrix(= mat_mesuring*sparse_base)
# paper : https://ieeexplore.ieee.org/document/342465
# code : heavily borrowed from https://blog.csdn.net/hjxzb/article/details/50923158
def omp(y, D):
	K = math.floor(3*(y.shape[0])/4) # number of iterations = degree of sparsity
	res = y # initialize residual
	index = np.ones((img_size), dtype=int) * -1
	result = np.zeros((img_size))
	for j in range(K):
		product = np.fabs(np.dot(np.transpose(D), res)) # column-wise projection
		pos = np.argmax(product) # position of max projection coefficient
		index[j] = pos
		my = np.linalg.pinv(D[:, index>=0]) # least square method
		result_dense = np.dot(my, y) # least square method
		res = y - np.dot(D[:, index>=0], result_dense)
	result[index>=0] = result_dense
	return result


# IHT : a threshold method
# y : img_mesured; D : sensing matrix(= mat_mesuring*sparse_base)
# paper : https://arxiv.org/abs/0805.0510v1
# code : heavily borrowed from https://blog.csdn.net/hjxzb/article/details/50932996
def iht(y, D):
	K = math.floor(y.shape[0]/3) # degree of sparsity
	result = np.zeros((img_size))
	u = 0.5 # influence factor
	for i in range(K):
		result = result + np.dot(np.dot(np.transpose(D), y - np.dot(D, result)), u) # x(t+1) = Hs( x(t) + uD^T(y-Dx(t)))
		temp = np.fabs(result)
		pos = np.argsort(temp)
		pos = pos[::-1] # "small->large" -> "large->small"
		result[pos[K:]] = 0 # the smaller positions become zero to satisfy sparsity
	return result


# IRLS : non-convex optim
# y : img_mesured; D : sensing matrix(= mat_mesuring*sparse_base)
# paper : https://ieeexplore.ieee.org/document/4518498
# code : heavily borrowed from https://blog.csdn.net/hjxzb/article/details/51077309
def irls(y, D):
	K = math.floor((y.shape[0])/3) # degree of sparsity
	result_temp = np.dot(np.transpose(D), y)
	i = 0 # index of iteration
	p = 1 # p-norm
	epsilon = 1
	while (epsilon > 1e-9) and (i < K):
		weight = (result_temp**2 + epsilon)**(p/2 - 1) # weight = ||s_(i-1)||^(p-2) but with an epsilon
		Q = np.diag(1/weight) # Q is a diagonal matrix, Q_i=1/weight_i
		result = np.dot(Q, np.dot(np.transpose(D), np.dot(np.linalg.inv(np.dot(D, np.dot(Q, np.transpose(D)))), y))) # s_(i+1)=QD^T(DQD^T)^(-1)y
		if np.linalg.norm(result - result_temp, 2) < np.sqrt(epsilon)/100:
			epsilon = epsilon/10
		result_temp = result
		i = i + 1
	return result

'''
Input&Reconstruction
'''


if __name__ == '__main__':
	img = np.array(Image.open('../assets/img/lena512.bmp').resize((img_size, img_size), resample=Image.BILINEAR))
	mat_sparse_coef = np.zeros((img_size, img_size)) # initialize sparse coefficient matrix
	mat_mesuring, img_mesured = sense(img, mode='gaussian', sampling_rate=0.7) # mesurement
	sparse_base = gen_sparse_base(mode='DCT')
	mat_sensing = np.dot(mat_mesuring, sparse_base)
	for i in range(img_size):
		print("reconstructing %d-th column..." % i)
		col_sparse_coef = irls(img_mesured[:, i], mat_sensing)
		mat_sparse_coef[:, i] = col_sparse_coef
	img_rec = np.dot(sparse_base, mat_sparse_coef) # reconstructed image
	Image.fromarray(img_rec).show()