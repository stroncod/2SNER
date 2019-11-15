from pycrfsuite import ItemSequence
import sklearn
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from pprint import pprint 
import numpy as np


class CRFSuite(object):

	def __init__(self,algorithm):
		self.trainer = sklearn_crfsuite.CRF(
    		algorithm='lbfgs',
    		c1=0.1,
    		c2=0.1,
		)

	def train(self,X,y):
	
		xseq,yseq = self.getItemSeq(X,y)
		#pprint(xseq)
		self.trainer = self.trainer.fit(xseq,yseq)

	def f1_score(self,y_pred, y_test):
		y_test = self.getItemSeqY(y_test)
		#y_test = y_test.reshape((len(y_pred)),1)
		return metrics.flat_f1_score(y_test, y_pred, average='weighted')


	def label(self,x_test):
		x_test	= self.getItemSeqX(x_test)
		y_pred = self.trainer.predict(x_test)
		return y_pred 

	def getItemSeq(self,X,y):
	
		xseq = []
		yseq = []
		featDict = {}
		for vector in X:
			for index,item in enumerate(vector):
				featDict[str(index)] = item
			feat_arr = [featDict]
			featDict = {}
			xseq.append(feat_arr)
			
		for item in y:
			token_arr = [item]	
			yseq.append(token_arr)

		#pprint(yseq)
	#xseq = pycrfsuite.ItemSequence(xseq)
	#yseq = pycrfsuite.ItemSequence(yseq)#modificar a dict {iterador: entidad}
		return xseq,yseq

	def getItemSeqX(self,X):
		xseq = []
		featDict = {}
		for vector in X:
			for index,item in enumerate(vector):
				featDict[str(index)] = item
			feat_arr = [featDict]
			featDict = {}
			xseq.append(feat_arr)
		return xseq

	def getItemSeqY(self,y):
		yseq = []

		for item in y:
			token_arr = [item]	
			yseq.append(token_arr)
		return yseq