__author__ = 'Wasi Uddin Ahmad, Md Masudur Rahman'

import pandas as pd

def loadDataSet(filename):
	data = pd.DataFrame.as_matrix(pd.read_csv(filename, sep=','))
	x = data[:,0]
	y = data[:,2]
	#printDataSet(x, y)
	return (y, x)

def printDataSet(xVal, yVal):
	for i in range(xVal.shape[0]):
		#print ('Sample No.' + str(i), end=" , ")
		print (xVal[i], ",", yVal[i])
	print ("Size of the data set: ", xVal.shape[0])

if __name__ == '__main__':
	pass