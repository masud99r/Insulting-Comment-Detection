__author__ = 'Wasi Uddin Ahmad, Md Masudur Rahman'

import numpy as np
from sklearn.cluster import KMeans

def generateBOW(comment, vocabulary):
	tokenList = comment.getTokensList()
	BOW = []
	for i in range(len(vocabulary)):
		if vocabulary[i] in tokenList:
			BOW += [tokenList[vocabulary[i]]]
		else:
			BOW += [0]
	return BOW

def runKMneas(listOfTrainComments, listOfTestComments, listOfUniqueTokens):
	xTrain = []
	yTrain = []
	for i in range(len(listOfTrainComments)):
		BOW = generateBOW(listOfTrainComments[i], listOfUniqueTokens)
		xTrain.append(BOW)
		yTrain.append(listOfTrainComments[i].getStatus())

	xTest = []
	yTest = []
	for i in range(len(listOfTestComments)):
		BOW = generateBOW(listOfTestComments[i], listOfUniqueTokens)
		xTest.append(BOW)
		yTest.append(listOfTestComments[i].getStatus())

	clf = KMeans(n_clusters=2, max_iter = 300)
	clf.fit(xTrain, yTrain)
	score = clf.score(xTest)
	prediction = clf.predict(xTest)
	print('K-means Clustering, Score - ' + str(score), '\n')

if __name__ == '__main__':
	pass
