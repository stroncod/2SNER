import numpy as np
from pprint import pprint 
def extractFeatures(dataset):
	vector= []
	X_set = []
	y_set = []
	for line in dataset:
		if line.strip():
			#print(line)
			token,entity,vector = line.split("\t")
			##print(line.split('\t'))
			featSet = vector.rstrip().split(',')
			
			#temporal fix 
			if len(featSet) < 46:
				#print(" se tiene que cambiar")
				fix_vector = vector.replace(',,',',___,___,')
				featSet = fix_vector.rstrip().split(',')
			else:
				fix_vector = vector.replace(',,',',___,')
				featSet = fix_vector.rstrip().split(',')
				
			y_set.append(entity)
			X_set.append(featSet)
	#inside_shape = max([len(i) for i in X_set])

	X_set= np.array(X_set)
	y_set= np.array(y_set)	
	#X_set = X_set.reshape(X_set.shape[0],inside_shape-1)
	return X_set, y_set

def extractDim(featSet):
	return max(len(i) for i in featSet)

def binary_label_transform(y):
	y_set = set(y)
	if 'O' in y_set:
		y_set.remove('O')
	elif 'token' in y_set:
		y_set.remove('token')
	else:
		print('Not token label found')

	for index,label in enumerate(y):
		if label in y_set:
			y[index] = 'entity'

	return y


if __name__ == '__main__':
	main()