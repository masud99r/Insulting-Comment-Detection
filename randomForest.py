__author__ = 'Wasi Uddin Ahmad, Md Masudur Rahman'

from sklearn.ensemble import RandomForestClassifier

def generateBOW(comment, vocabulary):
	tokenList = comment.getTokensList()
	BOW = []
	for i in range(len(vocabulary)):
		if vocabulary[i] in tokenList:
			BOW += [tokenList[vocabulary[i]]]
		else:
			BOW += [0]
	return BOW

def randomForestClassification(listOfTrainComments, listOfTestComments, listOfUniqueTokens):
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

	clf = RandomForestClassifier(n_estimators=4, criterion='entropy')
	clf.fit(xTrain, yTrain)
	accuracy = clf.score(xTest, yTest)
	# print('Random Classifier Classifier, Accuracy - ' + str(round(accuracy*100, 2)) + '%', '\n')
	prediction = clf.predict(xTest)
	return (prediction, accuracy)

if __name__ == '__main__':
	pass