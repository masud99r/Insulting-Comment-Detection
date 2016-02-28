__author__ = 'Wasi Uddin Ahmad, Md Masudur Rahman'

from sklearn.metrics.pairwise import cosine_similarity

def generateBOW(comment, invertedDocumentFrequencyOfTokens, vocabulary):
	tokenList = comment.getTokensList()
	BOW = []
	for i in range(len(vocabulary)):
		if vocabulary[i] in tokenList:
			BOW += [tokenList[vocabulary[i]] * invertedDocumentFrequencyOfTokens[vocabulary[i]]]
		else:
			BOW += [0]
	return BOW

def kNearestNeighbor(listOfTrainComments, listOfTestComments, listOfUniqueTokens, invertedDocumentFrequencyOfTokens, k):
	xTrain = []
	yTrain = []
	for i in range(len(listOfTrainComments)):
		BOW = generateBOW(listOfTrainComments[i], invertedDocumentFrequencyOfTokens, listOfUniqueTokens)
		xTrain.append(BOW)
		yTrain.append(listOfTrainComments[i].getStatus())

	xTest = []
	yTest = []
	for i in range(len(listOfTestComments)):
		BOW = generateBOW(listOfTestComments[i], invertedDocumentFrequencyOfTokens, listOfUniqueTokens)
		xTest.append(BOW)
		yTest.append(listOfTestComments[i].getStatus())

	predictedLabels = []
	for i in range(len(xTest)):
		cosValues = []
		for j in range(len(xTrain)):
			cosValues.append(cosine_similarity(xTrain[j], xTest[i]))
		sortedResult = [val[0] for val in sorted(enumerate(cosValues), key=lambda x:x[1], reverse=True)]
		voteCountingMap = {}
		for j in range(k):
			result = yTrain[sortedResult[j]]
			if result in voteCountingMap:
				voteCountingMap[result] += 1
			else:
				voteCountingMap[result] = 1

		maximumVote = 0
		classLabel = 0
		for key, value in voteCountingMap.items():
			if value > maximumVote:
				maximumVote = value
				classLabel = key
		predictedLabels.append(classLabel)
		print(str(i+1) + " : " + str(classLabel))

	missClassification = 0
	for i in range(len(predictedLabels)):
		if predictedLabels[i] != yTest[i]:
			missClassification += 1
	accuracy = missClassification / len(predictedLabels)
	accuracy = 1 - accuracy
	print('K-Nearest Neighbor Classifier, Accuracy - ' + str(round(accuracy*100, 2)) + '%', '\n')

if __name__ == '__main__':
	pass
