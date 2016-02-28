__author__ = 'Wasi Uddin Ahmad, Md Masudur Rahman'

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB

def generateBOW(comment, vocabulary):
	tokenList = comment.getTokensList()
	BOW = []
	for i in range(len(vocabulary)):
		if vocabulary[i] in tokenList:
			BOW += [tokenList[vocabulary[i]]]
		else:
			BOW += [0]
	return BOW

def MultinomialNaiveBayes(listOfTrainComments, listOfTestComments, listOfUniqueTokens):
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

	clf = MultinomialNB()
	clf.fit(xTrain, yTrain)
	accUsingSklearn = clf.score(xTest, yTest)
	retValue = clf.predict(xTest)
	#print('Multinomial Naive Bayes Classifier, Accuracy - ' + str(round(accUsingSklearn*100, 2)) + '%')
	return (yTest, retValue, accUsingSklearn)

def GaussianNaiveBayes(listOfTrainComments, listOfTestComments, listOfUniqueTokens):
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

	clf = GaussianNB()
	clf.fit(xTrain, yTrain)
	accUsingSklearn = clf.score(xTest, yTest)
	print('Gaussian Naive Bayes Classifier, Accuracy - ' + str(round(accUsingSklearn*100, 2)) + '%', '\n')
	
def BernoulliNaiveBayes(listOfTrainComments, listOfTestComments, listOfUniqueTokens):
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

	clf = BernoulliNB()
	clf.fit(xTrain, yTrain)
	accUsingSklearn = clf.score(xTest, yTest)
	print('Bernoulli Naive Bayes Classifier, Accuracy - ' + str(round(accUsingSklearn*100, 2)) + '%', '\n')

if __name__ == '__main__':
	pass