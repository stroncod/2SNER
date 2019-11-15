import numpy as np
from pprint import pprint 
#from classifier import crfsuite
import random
class SFFS(object):

  def __init__ (self,classifier,X_train,y_train, X_test, y_test,dim):
    self.classifier = classifier
    #self.k_features = k_features
    self.X_train = X_train
    self.y_train = y_train
    self.X_test = X_test
    self.y_test = y_test
    self.subset = set()
    self.indices=tuple()
    self.dim = dim

  def fit(self):
    best_score = 0
    is_improving = True
    is_back_improving = True
    dimension = self.dim
    i_features = set(i for i in range(dimension))
    j= 0
    while is_improving:
      aux_score = 0
      for i in i_features:
        if not self.indices:
          self.indices = (i,)
        else:
          self.indices = tuple(self.subset)+(i,)
        print(self.indices)
        score = self.calc_score(self.indices)
        if score>aux_score:
          aux_score = score
          aux_i = i
        print(j,self.subset,self.indices,score,best_score,'\n', sep=' ', end='',flush=True)
        
      if aux_score >= best_score:
        best_score = aux_score
        self.subset.add(aux_i)
        i_features = i_features - self.subset

        
        if len(self.subset) >= 2:
          is_back_improving = True
          while is_back_improving:
            backwards_score = best_score
            for indice in self.subset.copy():
              arr_aux = set(self.subset)
              arr_aux.remove(indice)
              print(self.subset,arr_aux)
              tuple_aux = tuple(arr_aux)
              new_score = self.calc_score(tuple_aux)
              print('try=',indice,'new_score=',new_score,'indices=',tuple_aux,'\n', sep=' ',end='',flush=True)
              if new_score>backwards_score:
                backwards_score = new_score
                backward_tuple = tuple_aux
                backward_indice = indice

            if backwards_score > best_score:
              print('Removing feature...')    
              self.indices = backward_tuple
              self.subset.remove(backward_indice)
              best_score = backwards_score
            else:
              is_back_improving = False
        
      else:
        is_improving = False
      j+=1

    return self
    
  def calc_score(self,indices):

    ##self.classifier.train(self.X_train[:, indices],self.y_train)
    xtrain = self.transform(indices,self.X_train)
    xtest = self.transform(indices,self.X_test)
    #pprint(xtest)
    self.classifier.train(xtrain,self.y_train)
    y_pred = self.classifier.label(xtest)
    ytest = self.y_test.reshape((len(self.y_test)),1)
    #pprint(ytest)
    #print("CON EL APAGON QUE COSAS SUCEDEN")
    #pprint(y_pred)
    #pprint(ytest)
    ##pprint(y_pred)
    score = self.classifier.f1_score(ytest,y_pred)
        #self.classifier.clean()
    return score
    #return random.uniform(0,1)


  def transform(self,indices, X):
    x_reduce = []
    for item in X:
      x_reduce.append([ element for index,element in enumerate(item) if index in indices])
    return np.array(x_reduce)

