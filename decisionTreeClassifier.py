__author__ = 'Wasi Uddin Ahmad, Md Masudur Rahman'

import numpy as np
from sklearn.tree import DecisionTreeClassifier

def generateBOW(comment, vocabulary):
	tokenList = comment.getTokensList()
	BOW = []
	for i in range(len(vocabulary)):
		if vocabulary[i] in tokenList:
			BOW += [tokenList[vocabulary[i]]]
		else:
			BOW += [0]
	return BOW

def runDecisionTree(trainData, trainLabel, testData, testLabel, crit, split, depth):
	clf = DecisionTreeClassifier(criterion=crit, splitter=split, max_depth=depth)
	clf.fit(trainData, trainLabel)
	accuracy = clf.score(testData, testLabel)
	#print("Selected Parameters: Criterion - " + crit + ", Splitter - " + split + ", Max Depth - " + str(depth))
	#print("Decision Tree Classifier, Accuracy - " + str(round(accuracy*100, 2)) + "%")
	prediction = clf.predict(testData)
	return (prediction, accuracy)

def decisionTreeClassifier(listOfTrainComments, listOfTestComments, listOfUniqueTokens):
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

	prediction, accuracy = runDecisionTree(xTrain, yTrain, xTest, yTest, "entropy", "best", None)
	return (prediction, accuracy)

if __name__ == '__main__':
	pass