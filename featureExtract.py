__author__ = 'Wasi Uddin Ahmad, Md Masudur Rahman'

import string
from nltk import wordpunct_tokenize
from nltk import bigrams
from nltk.stem.snowball import EnglishStemmer # loading the stemmer module from NLTK

def commentTokenizer(comment):
	tokenList = wordpunct_tokenize(comment)
	#bi_tokens = [ " ".join(pair) for pair in bigrams(tokenList)]
	#tokenList = tokenList + bi_tokens
	return tokenList # returns list of tokens

def commentNormalizer(tokenList):
	for i in range(len(tokenList)):
		tokenList[i] = "".join([ch for ch in tokenList[i] if ch not in string.punctuation])
	
	tokenList = list(filter(None, tokenList)) # filters all empty list
	return tokenList # returns list of tokens after normalization

def commentStemmer(tokenList):
	stemmer = EnglishStemmer()
	for i in range(len(tokenList)):
		tokenList[i] = stemmer.stem(tokenList[i])
	return tokenList # returns list of tokens after stemming

if __name__ == '__main__':
	pass