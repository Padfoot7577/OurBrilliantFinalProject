import numpy as np
import matplotlib.pyplot as plt


'''
WARNING: need changing 
	-- right now I'm just generating a normal distribution and making it a function. It's wrong. 
'''
def sample_data():
	mu, sigma = 10, 0.1
	temp = np.random.normal(mu, sigma, 10)
	temp = sorted(temp)
	s = [(i,temp[i]) for i in range(len(temp))]
	return s
print(sample_data())

'''
an af takes in a vector of polynomial coefficient and some data,
and outputs the delta_th that an update wants to subtract with
currently for sample_af I'm using gradient as af
th: d*1
x: d*n
y: 1*n
all in numpy
I don't have lambda regulation in it, which is related to the error function. 
We might consider adding it, otherwise it's hard to converge
'''
def sample_af(th, x, y):
	d,n = np.shape(x)
	s = None
	lin_alg = 2/n*(np.dot(th.T, x) - y)*x
	del_th = np.array([np.sum(lin_alg, axis = 1)]).T
	return del_th

def initialize(dimension):
	return np.zeros((dimension,1))

'''test for sample_af'''
# X = np.array([[1,2,3,4],[1,4,9,16]])
# y = np.array([[1,2,3,4]])
# th = np.array([[1],[1]])
# print(sample_af(th,X,y))


'''
raw_data: list of (x,y) pairs
dimension: the degree we want our polynomial to be
output: a tuple of (X,Y) where X,Y are both numpy matricies
'''
def make_dim_data(raw_data, dimension):
	n = len(raw_data)
	X = np.zeros((dimension,n))
	Y = np.zeros((1,n))
	for i in range(n):
		x,y = raw_data[i]
		Y[0][i] = y
		for d in range(dimension):
			X[dimension-d-1][i] = x**d
	return (X,Y)

'''test of make_dim_data'''
# data = [(1,2),(2,4),(3,2),(4,1)]
# print(make_dim_data(data,3))



'''
error function
th: coefficients of the polynomial
WARNING: this error is not penalizing norm(th), which makes it hard to converge!!
'''
def error(th,X,y):
	return np.sum((th.T@X - y)**2)


'''
for each iteration, test the error and make a graph

af: acceleration function e.g. gradient, heavy ball, Nesterov
data: data that we want to fit
iter: number of iteration that we want to test 
	  -- we could change this to a threshold for error
lr: learning rate
'''
def test_function(af, data, dimension, iter, lr):
	error_per_iter = []
	X,y = make_dim_data(data, dimension)
	th = initialize(dimension)

	for _ in range(iter):
		e = error(th,X,y)
		error_per_iter.append(e)
		th -= af(th,X,y)*lr
	return (error_per_iter)

'''overall testcase'''
data = sample_data()
print(test_function(sample_af,data,4, 50, 0.01))