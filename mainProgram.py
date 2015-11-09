__author__ = 'Wasi Uddin Ahmad, Md Masudur Rahman'

import readData as rd
import numpy as np
import featureExtract as fext
from commentClass import Comment
from naiveBayesClassifier import MultinomialNaiveBayes
from naiveBayesClassifier import GaussianNaiveBayes
from naiveBayesClassifier import BernoulliNaiveBayes
from svmClassifier import svmClassifier

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
		#listOfUniqueTokens = listOfUniqueTokens + tokenList  # list of unique tokens

		dicTokens = calculateTermFrequency(tokenList)
		listOfTrainComments[i].setTokenList(dicTokens)
		for key, value in dicTokens.items():
			if key in documentFrequencyOfTokens:
				documentFrequencyOfTokens[key] += 1
			else:
				documentFrequencyOfTokens[key] = 1

	#listOfUniqueTokens = list(set(listOfUniqueTokens)) #vocabulary
	#documentFrequencyOfTokens = sorted(documentFrequencyOfTokens.items(), key=lambda x: x[1], reverse=True)
	for key, val in documentFrequencyOfTokens.items():
		if val >= 10:
			listOfUniqueTokens.append(key)

	'''
	invertedDocumentFrequencyOfTokens = {}
	totalNumberOfDoc = len(listOfTrainComments)
	for key, val in documentFrequencyOfTokens.items():
		invertedDocumentFrequencyOfTokens[key] = 1 + np.log2(totalNumberOfDoc / val)

	for i in range(len(listOfTrainComments)):
		cmnt = listOfTrainComments[i]
		tokenList = cmnt.getTokensList()
		for key, val in tokenList.items():
			tokenList[key] = val * invertedDocumentFrequencyOfTokens[key]
	'''

	print(len(listOfUniqueTokens))
	return (listOfTrainComments, listOfUniqueTokens)

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
		dicTokens = calculateTermFrequency(tokenList)
		listOfTestComments[i].setTokenList(dicTokens)

	return listOfTestComments

if __name__ == '__main__':
	listOfTrainComments, listOfUniqueTokens = processTrainData('train.csv')
	listOfTestComments = processTestData('test_with_solutions.csv')
	MultinomialNaiveBayes(listOfTrainComments, listOfTestComments, listOfUniqueTokens)
	#BernoulliNaiveBayes(listOfTrainComments, listOfTestComments, listOfUniqueTokens)
	#GaussianNaiveBayes(listOfTrainComments, listOfTestComments, listOfUniqueTokens)
	#svmClassifier(listOfTrainComments, listOfTestComments, listOfUniqueTokens, 5, 0.01, 10)
