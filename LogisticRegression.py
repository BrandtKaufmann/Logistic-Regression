import pandas as pd
import numpy as np

	

def StandardScaler(data):
	# Scaling data, transforming values into their Z-scores
	# 'data' = np.array, pd.DataFrame, pd.Series, native array
	
	mean = np.mean(data, axis=0)
	std = np.std(data, axis=0)
	ScaledData = (data - mean) / std
	
	return ScaledData

def TrainTestSplit(data, TestProportion, Label = None):
	# splitting data randomly into training and testing,
	# taking in pd.DataFrame, proportion of data for testing,
	# and index or name of labels if desired
	# returning train and test pd.DataFrames in that order, if given
	# Label, split into X_train, X_test, y_train, y_test
	
	TotalRows = data.shape[0]
	TotalCols = data.shape[1]

	NumberOfTestRows = int(TotalRows * TestProportion)
	TestRows = np.random.choice(TotalRows, size=NumberOfTestRows, replace=False)

	Train = data.drop(TestRows, axis=0)
	Test = data.iloc[TestRows]
	
	if not Label:
	
		return Train, Test
	
	elif (type(Label) == int and Label<TotalCols):
		
		X_train = Train.drop([data.columns[Label]], axis = 1)
		X_test = Test.drop([data.columns[Label]], axis = 1)
		
		y_train = Train.iloc[:,Label]
		y_test = Test.iloc[:,Label]
		
		return X_train, X_test, y_train, y_test

	elif(type(Label) == str and Label in data.columns):
		
		X_train = Train.drop([Label], axis = 1)
		X_test = Test.drop([Label], axis = 1)
		
		y_train = Train.loc[:,Label]
		y_test = Train.loc[:,Label]

		return X_train, X_test, y_train, y_test
	
	else:
		print('Bad Label: <', Label, '>, returned None')
	
	return None

def InitializeWeights(NumberOfWeights, ReLU = None):
	# Params are int number of input nodes, 1 bool for ReLU activation function, 
	# Sigmoid/Tanh activation both work well with Xavier Weight Initialization:
	# 	"Understanding the difficulty of training deep feedforward neural networks" (Gorlot, Bengio)
	# but Xavier intialization has problems with ReLU so we use He Weight Initialization:
	# 	“Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification.” (Kaiming He, et al.)
	
	if not ReLU:
		
		# Xavier Weight Initialization require the number of inputs, 
		# and generates random numbers in a uniform distribution between ±1/sqrt(numInputs), 
		
		UpperBound = 1.0/np.sqrt(NumberOfWeights) 
		Weights = np.random.uniform(-UpperBound, UpperBound, NumberOfWeights)
			
	else:
		
		# He Weight Initialization returns n random numbers from a Gaussian distribution
		# with mean 0 and std = sqrt(2/n)
		std = np.sqrt(2.0/NumberOfWeights)
		Weights = np.random.normal(0.0, std, NumberOfWeights)
	
	return Weights

def MakePredictions(Data, Weights, Bias)
	# Takes in an input feature(single row of DataFrame), 
	# weights(row with weight for each column), and bias(scaler/int)
	# Takes dot product of two 1 dim arrays, adds bias(y intercept)
	# returning (Weight1 * Input1) + (Weight2 * Input2) + ... + BiasValue 
	# Applies Sigmoid activation function to squeeze resulting value to the (0,1) range
	
	Predictions = Data.dot(Weights) + Bias	
	Predictions = 1/(1+np.exp(-Predictions))
	
def MakeBinary(Data, Threshold = 0.5):
	
	BinaryPredictions = (Data >= Threshold).astype(int)

def CalculateLoss(X_train, Labels, Weights, Bias):
	# Returns the loss of a training run, we compare our hypothesis with
	# the ground truth labels, applies a function to calculate the loss of our predictions 