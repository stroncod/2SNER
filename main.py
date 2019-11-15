import sklearn_crfsuite
from crfsuite import CRFSuite
import utils
from sffs import SFFS
#import classifiers

train_dataset = open('datasets/dev_feats_3.5-train','r')
test_dataset = open('datasets/dev_feats_3.5-test','r')


X_train,y_train = utils.extractFeatures(train_dataset)
dim_train = utils.extractDim(X_train)

X_test, y_test = utils.extractFeatures(test_dataset)
#Binary token in one step

#y_train = utils.binary_label_transform(y_train)
#y_test = utils.binary_label_transform(y_test)



crf = CRFSuite('l2sgd')
sffs = SFFS(crf,X_train,y_train,X_test,y_test,dim_train)
sffs = sffs.fit()
'''
#save last model
crf = sffs.classifier
pred = sffs.y_pred
del sffs 
#add first step input 
for prediction,vector in zip(y_pred,X_test):
	for t,y in enumerate(prediction):
		vector.append(crf.tagger_.marginal(y,t))
#change sffs 
sffs = SFFS(crf,X_test,y_test,X_train,y_test,dim_train)
sffs = sffs.fit()
'''
#utils.getMetrics()