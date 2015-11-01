__author__ = 'Wasi Uddin Ahmad, Md Masudur Rahman'

import readData as rd
import featureExtract as fext
from commentClass import Comment

listOfComments = []
listOfUniqueTokens = [] # unique tokens of the entire corpus
documentFrequencyOfTokens = {}

def calculateTermFrequency(tokenList):
	dictionary = {}
	for i in range(len(tokenList)):
		if tokenList[i] in dictionary:
			dictionary[tokenList[i]] += 1
		else:
		 	dictionary[tokenList[i]] = 1
	return dictionary

if __name__ == '__main__':
	xVal, yVal = rd.loadDataSet('train.csv')
	for i in range(xVal.shape[0]):
		tempVal = Comment(i)
		tempVal.setContent(xVal[i])
		tempVal.setStatus(yVal[i])
		listOfComments.append(tempVal)

	for i in range(len(listOfComments)):
		content = listOfComments[i].getContent()
		status = listOfComments[i].getStatus()
		content = fext.commentNormalizer(content)
		tokenList = fext.commentTokenizer(content)
		tokenList = fext.removeStopWords(tokenList)
		tokenList = fext.commentStemmer(tokenList)
		listOfUniqueTokens = listOfUniqueTokens + tokenList  # dictionary containing unique tokens

		dicTokens = calculateTermFrequency(tokenList)
		listOfComments[i].setTokenList(dicTokens)

		for key, value in dicTokens.items():
			if key in documentFrequencyOfTokens:
				documentFrequencyOfTokens[key] += 1
			else:
				documentFrequencyOfTokens[key] = 1
		
		'''
		print(listOfComments[i].getContent())
		print('***************************')
		print(dicTokens)
		print('***************************')
		'''

	listOfUniqueTokens = list(set(listOfUniqueTokens))
	print(len(listOfUniqueTokens))
	documentFrequencyOfTokens = sorted(documentFrequencyOfTokens.items(), key=lambda x: x[1], reverse=True)
	for i in range(len(documentFrequencyOfTokens)):
		var = documentFrequencyOfTokens[i]
		if var[1] > 5:
			print(var[0], ' : ', var[1])