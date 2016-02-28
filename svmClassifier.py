__author__ = 'Wasi Uddin Ahmad, Md Masudur Rahman'

from sklearn.svm import SVC

def generateBOW(comment, vocabulary):
	tokenList = comment.getTokensList()
	BOW = []
	for i in range(len(vocabulary)):
		if vocabulary[i] in tokenList:
			BOW += [tokenList[vocabulary[i]]]
		else:
			BOW += [0]
	return BOW

def testModelWithHyperParameter(train_xValues, train_yValues, test_xValues, test_yValues, cValue, kernel_name):
	clf = SVC(C=cValue,kernel=kernel_name)
	clf.fit(train_xValues, train_yValues)
	trainAcc = clf.score(train_xValues, train_yValues)
	testAcc = clf.score(test_xValues, test_yValues)
	prediction = clf.predict(test_xValues)
	#print("C: " + str(cValue), "Train Accuracy: " + str(round(trainAcc*100, 2)) + "%", "Test Accuracy: " + str(round(testAcc*100, 2)) + "%\n")
	#print("SVM Classifier, Accuracy - " + str(round(testAcc, 2)) + "%")
	return (prediction, testAcc)

def svmClassifier(listOfTrainComments, listOfTestComments, listOfUniqueTokens, numItr, baseValue, mulFactor):
	train_xValues = []
	train_yValues = []
	for i in range(len(listOfTrainComments)):
		BOW = generateBOW(listOfTrainComments[i], listOfUniqueTokens)
		train_xValues.append(BOW)
		train_yValues.append(listOfTrainComments[i].getStatus())

	test_xValues = []
	test_yValues = []
	for i in range(len(listOfTestComments)):
		BOW = generateBOW(listOfTestComments[i], listOfUniqueTokens)
		test_xValues.append(BOW)
		test_yValues.append(listOfTestComments[i].getStatus())

	print('Accuracy Statistics for Linear Kernel')
	penalty_param = baseValue
	for i in range(numItr):
		testModelWithHyperParameter(train_xValues, train_yValues, test_xValues, test_yValues, penalty_param, 'linear')
		penalty_param = penalty_param * mulFactor
	
	print('Statistics for Polynomial Kernel')
	penalty_param = baseValue
	for i in range(numItr):
		testModelWithHyperParameter(train_xValues, train_yValues, test_xValues, test_yValues, penalty_param, 'poly')
		penalty_param = penalty_param * mulFactor

	print('Statistics for RBF Kernel')
	penalty_param = baseValue
	for i in range(numItr):
		testModelWithHyperParameter(train_xValues, train_yValues, test_xValues, test_yValues, penalty_param, 'rbf')
		penalty_param = penalty_param * mulFactor

	print('Statistics for Sigmoid Kernel')
	penalty_param = baseValue
	for i in range(numItr):
		testModelWithHyperParameter(train_xValues, train_yValues, test_xValues, test_yValues, penalty_param, 'sigmoid')
		penalty_param = penalty_param * mulFactor

def svmClassifier(listOfTrainComments, listOfTestComments, listOfUniqueTokens, penalty, kernelType):
	train_xValues = []
	train_yValues = []
	for i in range(len(listOfTrainComments)):
		BOW = generateBOW(listOfTrainComments[i], listOfUniqueTokens)
		train_xValues.append(BOW)
		train_yValues.append(listOfTrainComments[i].getStatus())

	test_xValues = []
	test_yValues = []
	for i in range(len(listOfTestComments)):
		BOW = generateBOW(listOfTestComments[i], listOfUniqueTokens)
		test_xValues.append(BOW)
		test_yValues.append(listOfTestComments[i].getStatus())

	prediction, testAcc = testModelWithHyperParameter(train_xValues, train_yValues, test_xValues, test_yValues, penalty, kernelType)
	return (prediction, testAcc)

if __name__ == '__main__':
	pass