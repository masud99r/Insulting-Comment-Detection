__author__ = 'Wasi Uddin Ahmad, Md Masudur Rahman'

import re
import string
import numpy as np
from nltk import wordpunct_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk import bigrams
from nltk.stem.snowball import EnglishStemmer # loading the stemmer module from NLTK

def removeHTMLTages(comment):
	p = re.compile(r'<.*?>')
	comment = p.sub(' ', comment)
	comment = comment.replace('&nbsp;',' ')
	return comment

def removeBadTokens(comment):
	rep = {'\\xa0': ' ', '\\xc2': ' ', '\n': ' ', '\r': ''}
	rep = dict((re.escape(k), v) for k, v in rep.items())
	pattern = re.compile("|".join(rep.keys()))
	comment = pattern.sub(lambda m: rep[re.escape(m.group(0))], comment)
	return comment

def commentTokenizer(comment):
	comment = comment.lower()
	comment = removeHTMLTages(comment)
	comment = removeBadTokens(comment)
	tokenList = wordpunct_tokenize(comment)
	return tokenList # returns list of tokens

def convertToBigrams(tokenList):
	bigramList = []
	for i in range(len(tokenList)):
		if i == 0:
			continue
		bigram = tokenList[i-1] + '_' + tokenList[i]
		bigramList.append(bigram)
	return bigramList # returns list of tokens

def removeStopWords(tokenList):
	stopwordList = np.genfromtxt('stopwords.txt',dtype='str')
	filteredWords = [w for w in tokenList if not w in stopwordList]
	return filteredWords

def commentNormalizer(comment):
	comment = "".join([ch for ch in comment if ch not in string.punctuation])
	comment = re.sub("\\d+(\\.\\d+)?", "NUM", comment)
	return comment
	'''
	for i in range(len(tokenList)):
		tokenList[i] = "".join([ch for ch in tokenList[i] if ch not in string.punctuation])
		tokenList[i] = re.sub("\\d+(\\.\\d+)?", "NUM", tokenList[i])
	
	tokenList = list(filter(None, tokenList)) # filters all empty list
	return tokenList # returns list of tokens after normalization
	'''

def commentStemmer(tokenList):
	stemmer = EnglishStemmer()
	for i in range(len(tokenList)):
		tokenList[i] = stemmer.stem(tokenList[i])
	return tokenList # returns list of tokens after stemming

if __name__ == '__main__':
	pass
