import numpy as np
mu, sigma = 0, 0.1
s = np.random.normal(10, sigma, 10)




'''
for each iteration, test the error and make a graph
af: acceleration function e.g. gradient, heavy ball, Nesterov
data: data that we want to fit
iter: number of iteration that we want to test 
	  -- we could change this to a threshold for error
'''
def test_function(af, data, iter):
	def error(function,real):
		for d in real:

print(s)