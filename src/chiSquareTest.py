__author__ = 'Wasi Uddin Ahmad, Md Masudur Rahman'

from sklearn.feature_selection import chi2

def generateBOW(comment, vocabulary):
	tokenList = comment.getTokensList()
	BOW = []
	for i in range(len(vocabulary)):
		if vocabulary[i] in tokenList:
			BOW += [tokenList[vocabulary[i]]]
		else:
			BOW += [0]
	return BOW

def ChiTest(listOfTrainComments, listOfTestComments, listOfUniqueTokens, count):
	xTrain = []
	yTrain = []
	vocab = []
	for i in range(len(listOfTrainComments)):
		BOW = generateBOW(listOfTrainComments[i], listOfUniqueTokens)
		xTrain.append(BOW)
		yTrain.append(listOfTrainComments[i].getStatus())

	chi2v, pval = chi2(xTrain, yTrain)
	indices = [i[0] for i in sorted(enumerate(chi2v), key=lambda x:x[1], reverse=True)]
	for i in range(len(indices)):
		#print(listOfUniqueTokens[indices[i]] + "\t" + str(chi2v[indices[i]]))
		vocab.append(listOfUniqueTokens[indices[i]])
		if (i+1) == count:
			break
	
	return vocab

if __name__ == '__main__':
	pass
