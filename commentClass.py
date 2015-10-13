__author__ = 'Wasi Uddin Ahmad, Md Masudur Rahman'

class Comment:

	list_of_tokens = None
	content = None
	status = -1

	def __init__(self, id):
		self.id = id

	def setContent(self, val):
		self.content = val

	def getContent(self):
		return self.content

	def setStatus(self, status):
		self.status = status

	def getStatus(self):
		return self.status

	def setTokenList(self, tokenList): # sets token dictionary
		self.list_of_tokens = tokenList

	def getTokensList(): # returns token dictionary
		return list_of_tokens