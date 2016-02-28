__author__ = 'Wasi Uddin Ahmad, Md Masudur Rahman'

import readData as rd
import numpy as np
import featureExtract as fext
from commentClass import Comment
from naiveBayesClassifier import MultinomialNaiveBayes
from naiveBayesClassifier import GaussianNaiveBayes
from naiveBayesClassifier import BernoulliNaiveBayes
from logisticRegressionClassifier import LRClassifier
from decisionTreeClassifier import decisionTreeClassifier
from randomForest import randomForestClassification
from adaBoost import adaBoostClassifier
from kMeans import runKMneas
from svmClassifier import svmClassifier
from kNearestNeighbor import kNearestNeighbor
from chiSquareTest import ChiTest
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

def calculateTermFrequency(tokenList):
	dictionary = {}
	for i in range(len(tokenList)):
		if tokenList[i] in dictionary:
			dictionary[tokenList[i]] += 1
		else:
		 	dictionary[tokenList[i]] = 1
	return dictionary

def processTrainData(filename):
	listOfTrainComments = []
	listOfUniqueTokens = [] # unique tokens of the entire corpus
	documentFrequencyOfTokens = {}
	xVal, yVal = rd.loadDataSet(filename)
	for i in range(xVal.shape[0]):
		tempVal = Comment(i)
		tempVal.setContent(xVal[i])
		tempVal.setStatus(yVal[i])
		listOfTrainComments.append(tempVal)

	for i in range(len(listOfTrainComments)):
		content = listOfTrainComments[i].getContent()
		status = listOfTrainComments[i].getStatus()
		content = fext.commentNormalizer(content)
		tokenList = fext.commentTokenizer(content)
		tokenList = fext.removeStopWords(tokenList)
		tokenList = fext.commentStemmer(tokenList)
		# bigramList = fext.convertToBigrams(tokenList)
		# tokenList = tokenList + bigramList
		listOfUniqueTokens = listOfUniqueTokens + tokenList  # list of unique tokens

		dicTokens = calculateTermFrequency(tokenList)
		listOfTrainComments[i].setTokenList(dicTokens)
		for key, value in dicTokens.items():
			if key in documentFrequencyOfTokens:
				documentFrequencyOfTokens[key] += 1
			else:
				documentFrequencyOfTokens[key] = 1

	#documentFrequencyOfTokens = sorted(documentFrequencyOfTokens.items(), key=lambda x: x[1], reverse=True)
	for key, val in documentFrequencyOfTokens.items():
		if val >= 5:
			listOfUniqueTokens.append(key)

	invertedDocumentFrequencyOfTokens = {}
	totalNumberOfDoc = len(listOfTrainComments)
	for key, val in documentFrequencyOfTokens.items():
		invertedDocumentFrequencyOfTokens[key] = 1 + np.log2(totalNumberOfDoc / val)

	'''
	for i in range(len(listOfTrainComments)):
		cmnt = listOfTrainComments[i]
		tokenList = cmnt.getTokensList()
		for key, val in tokenList.items():
			tokenList[key] = val * invertedDocumentFrequencyOfTokens[key]
	'''
	#print(len(listOfUniqueTokens))
	return (listOfTrainComments, listOfUniqueTokens, invertedDocumentFrequencyOfTokens, documentFrequencyOfTokens)

def processTestData(filename):
	listOfTestComments = []
	xVal, yVal = rd.loadDataSet(filename)
	for i in range(xVal.shape[0]):
		tempVal = Comment(i)
		tempVal.setContent(xVal[i])
		tempVal.setStatus(yVal[i])
		listOfTestComments.append(tempVal)

	for i in range(len(listOfTestComments)):
		content = listOfTestComments[i].getContent()
		status = listOfTestComments[i].getStatus()
		content = fext.commentNormalizer(content)
		tokenList = fext.commentTokenizer(content)
		tokenList = fext.removeStopWords(tokenList)
		tokenList = fext.commentStemmer(tokenList)
		# bigramList = fext.convertToBigrams(tokenList)
		# tokenList = tokenList + bigramList
		dicTokens = calculateTermFrequency(tokenList)
		listOfTestComments[i].setTokenList(dicTokens)

	return listOfTestComments

def calculateAccuracy(weightList, actualLabel, predictedLabel1, predictedLabel2, predictedLabel3, predictedLabel4):
	finalPredictedLabel = []
	for i in range(len(predictedLabel1)):
		val1 = predictedLabel1[i]
		val2 = predictedLabel2[i]
		val3 = predictedLabel3[i]
		val4 = predictedLabel4[i]
		countOnes = 0
		countZeros = 0
		if val1 == 1:
			countOnes += weightList[0]
		else:
			countZeros += weightList[0]
		if val2 == 1:
			countOnes += weightList[1]
		else:
			countZeros += weightList[1]
		if val3 == 1:
			countOnes += weightList[2]
		else:
			countZeros += weightList[2]
		if val4 == 1:
			countOnes += weightList[3]
		else:
			countZeros += weightList[3]
		if countOnes >= countZeros:
			finalPredictedLabel.append(1)
		else:
			finalPredictedLabel.append(0)

	f1score = f1_score(actualLabel, finalPredictedLabel)
	precision = precision_score(actualLabel, finalPredictedLabel)
	recall = recall_score(actualLabel, finalPredictedLabel)
	# print('Precision of the Ensemble Learner - ' + str(round(precision*100, 2)) + '%')
	# print('Recall of the Ensemble Learner - ' + str(round(recall*100, 2)) + '%')
	# print('F score of the Ensemble Learner - ' + str(round(f1score*100, 2)) + '%')
	missClassification = 0
	for i in range(len(finalPredictedLabel)):
		if finalPredictedLabel[i] != actualLabel[i]:
			missClassification += 1

	accuracy = 1 - (missClassification / len(finalPredictedLabel))
	#print('\nAccuracy of the Ensemble Learner - ' + str(round(accuracy*100, 2)) + '%', '\n')
	return (accuracy, precision, recall, f1score)

def generateWeight(accNB, accSVM, accLR, accDT):
	tempList = [accNB, accSVM, accLR, accDT]
	indices = [i[0] for i in sorted(enumerate(tempList), key=lambda x:x[1], reverse=True)]
	weightList = [0, 0, 0, 0]
	for i in range(len(tempList)):
		weightList[indices[i]] = len(tempList) - i
	return weightList

if __name__ == '__main__':
	listOfTrainComments, listOfUniqueTokens, invertedDocumentFrequencyOfTokens, documentFrequencyOfTokens = processTrainData('train.csv')
	listOfTestComments = processTestData('test_with_solutions.csv')
	numberOfFeatures = 100
	for i in range(30):
		vocab = ChiTest(listOfTrainComments, listOfTestComments, listOfUniqueTokens, numberOfFeatures)
		# print(str(numberOfFeatures) + ':' + str(round(acc*100, 2)))
		actualLabel, predictedLabel1, accNB = MultinomialNaiveBayes(listOfTrainComments, listOfTestComments, vocab)
		predictedLabel2, accSVM = svmClassifier(listOfTrainComments, listOfTestComments, vocab, 1.0, 'linear')
		predictedLabel3, accLR = LRClassifier(listOfTrainComments, listOfTestComments, vocab)
		predictedLabel4, accDT = decisionTreeClassifier(listOfTrainComments, listOfTestComments, vocab)
		# weightList = [1, 1, 1, 1]
		weightList = generateWeight(accNB, accSVM, accLR, accDT)
		# print(weightList)
		accEN, precision, recall, f1score = calculateAccuracy(weightList, actualLabel, predictedLabel1, predictedLabel2, predictedLabel3, predictedLabel4)
		prediction, accRF = randomForestClassification(listOfTrainComments, listOfTestComments, vocab)
		prediction, accAda = adaBoostClassifier(listOfTrainComments, listOfTestComments, vocab)
		accNB = str(round(accNB*100, 2))
		accSVM = str(round(accSVM*100, 2))
		accLR = str(round(accLR*100, 2))
		accDT = str(round(accDT*100, 2))
		accEN = str(round(accEN*100, 2))
		accRF = str(round(accRF*100, 2))
		accAda = str(round(accAda*100, 2))
		precision = str(round(precision*100, 2))
		recall = str(round(recall*100, 2))
		f1score = str(round(f1score*100, 2))
		print(str(numberOfFeatures) + '\t' + accNB + '\t' + accSVM + '\t' + accLR + '\t' + accDT + '\t' + accEN + '\t' + accRF + '\t' + accAda + '\t' + precision + '\t' + recall + '\t' + f1score)
		#print(str(numberOfFeatures) + '\t' + accNB + '\t' + accSVM + '\t' + accLR + '\t' + accDT + '\t' + accEN)
		numberOfFeatures += 100
	# IDFtokens = {}
	# totalNumberOfDoc = len(listOfTrainComments)
	# for i in range(len(vocab)):
	# 	IDFtokens[vocab[i]] = 1 + np.log2(totalNumberOfDoc / documentFrequencyOfTokens.get(vocab[i]))
	# kNearestNeighbor(listOfTrainComments, listOfTestComments, vocab, IDFtokens, 3)
	# LRClassifier(listOfTrainComments, listOfTestComments, vocab)
	# predictedLabel5 = randomForestClassification(listOfTrainComments, listOfTestComments, vocab)
	# runKMneas(listOfTrainComments, listOfTestComments, vocab)
	# BernoulliNaiveBayes(listOfTrainComments, listOfTestComments, listOfUniqueTokens)
	# GaussianNaiveBayes(listOfTrainComments, listOfTestComments, listOfUniqueTokens)
	# svmClassifier(listOfTrainComments, listOfTestComments, vocab, 5, 0.01, 10)
