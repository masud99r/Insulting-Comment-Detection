__author__ = 'Wasi Uddin Ahmad, Md Masudur Rahman'

from sklearn.linear_model import LogisticRegression

def generateBOW(comment, vocabulary):
	tokenList = comment.getTokensList()
	BOW = []
	for i in range(len(vocabulary)):
		if vocabulary[i] in tokenList:
			BOW += [tokenList[vocabulary[i]]]
		else:
			BOW += [0]
	return BOW

def LRClassifier(listOfTrainComments, listOfTestComments, listOfUniqueTokens):
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

	logreg = LogisticRegression()
	logreg.fit(xTrain, yTrain)
	accuracy = logreg.score(xTest,yTest)
	accTrain = logreg.score(xTrain,yTrain)
	#print('Logistic Regression Classifier, Training Accuracy - ' + str(round(accTrain*100, 2)) + '%', '\n')
	#print('Logistic Regression Classifier, Accuracy - ' + str(round(accuracy*100, 2)) + '%')
	prediction = logreg.predict(xTest)
	return (prediction, accuracy)

if __name__ == '__main__':
	pass
