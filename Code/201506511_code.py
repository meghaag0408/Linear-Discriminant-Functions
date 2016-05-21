import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
import math as math

def augment_vector(X):
	bias = np.ones((len(X), 1))
    	return np.hstack((X, bias))

def normalise_train_set(X, Y):
	X = np.asarray(X)
	Y = np.asarray(Y)
	train_set = augment_vector(X)
	labels_ =  np.unique(Y)
	idx = Y==labels_[1]					#Select one class as negative class
	train_set[idx] = -train_set[idx]	#Make the x coordinate of selected class as negative
	dim = train_set.shape[1]			#Return Dimensionality of feature space // train set
	return X, Y, train_set, dim

def single_sample_perceptron(X, Y, learning_rate, margin):
	#Obtained the values as required (Augmented and negated)
	X, Y, train_set, dim = normalise_train_set(X, Y)
	
	i=0
	n = len(train_set)
	k=0
	weights = [10, 10, 10]
	print weights
	count = 0
	while i!=n:
		k = (k+1)%n
		count+=1
		if np.dot(train_set[k],weights)<=margin:
			weights = weights + learning_rate * train_set[k]
			i=0
		else:
			i=i+1
	print "HELLO"
	print 'Converged after {} iterations'.format(count)	
	print weights
	plot_boundary(weights, train_set, X, Y)
	return weights

def relaxation_algo_with_margin(X, Y, learning_rate, margin):
	#Obtained the values as required (Augmented and negated)
	X, Y, train_set, dim = normalise_train_set(X, Y)
	weights = [0, 0, 1]
	k=-1
	i=0
	count=0
	while i!=len(train_set):
		k=(k+1)%len(train_set)
		if np.dot(train_set[k],weights)<=margin:
			i=0
	 		temp1=(margin - np.dot(train_set[k],weights))
	 		temp2=np.dot(train_set[k],train_set[k])
	 		temp3=float((float(temp1)/float(temp2))*2)
	 		temp=np.dot(temp3,train_set[k])
	 		weights=weights + (learning_rate * temp)
		else:
			i+=1
			count+=1

	print weights
	print 'Converged after {} iterations'.format(count)	
	plot_boundary(weights, train_set, X, Y)
	return weights

def least_mean_square(X, Y, learning_rate, margin):
	#Obtained the values as required (Augmented and negated)
	X, Y, train_set, dim = normalise_train_set(X, Y)
	
	a4=[0.5,0.5,-1]
	a4=[0.125,0.125,-1]
	k=-1
	n=len(train_set)
	cnt=0
	nk1=1
	count=0
	while 1:
		k=(k+1)%n
		flag=1
		count+=1
		nk=float(nk1)/2000.0
		temp1=nk* (margin - np.dot(train_set[k],a4))
		temp2=np.dot(temp1,train_set[k])
		temp3=math.sqrt(np.dot(temp2,temp2))
		if temp3<margin:
			flag=0
		a4=np.sum([a4,temp2],axis=0)
		if(flag==1 or count==1000):
			break;

	print a4
	plot_boundary(a4, train_set, X, Y)
	return a4

def ques_6_least_mean_square(X, Y, learning_rate, margin):
	#Obtained the values as required (Augmented and negated)
	X, Y, train_set, dim = normalise_train_set(X, Y)
	
	a4=[0.5, 0.5 ,-1]
	k=-1
	n=len(train_set)
	cnt=0
	nk1=1
	count=0
	while 1:
		k=(k+1)%n
		flag=1
		count+=1
		nk=float(nk1)/2000.0
		temp1=nk* (margin - np.dot(train_set[k],a4))
		temp2=np.dot(temp1,train_set[k])
		temp3=math.sqrt(np.dot(temp2,temp2))
		if temp3<margin:
			flag=0
		a4=np.sum([a4,temp2],axis=0)
		if(flag==1 or count==2000):
			break;

	print a4
	weights = single_sample_perceptron_ques_6(X, Y, learning_rate, margin)	
	plot_boundary_ques_6(a4, weights, train_set, X, Y)
	return a4

def plot_boundary_ques_6(weights1, weights2, train_set, X, Y):
	# plot data-points
	x_points = X[:,0]
	y_points = X[:,1]
	length = len(x_points)

	x_points_1 = x_points[0:length/2]
	x_points_2 = x_points[length/2:length]
	y_points_1 = y_points[0:length/2]
	y_points_2 = y_points[length/2:length]
	
	plt.plot(x_points_1,y_points_1,'ro');
	plt.axis([0,10,0,10])
	plt.plot(x_points_2,y_points_2,'bo');
	

	a,b,c = weights1
	xchord_1 = 0
	xchord_2 = -(float(c))/(float(a))
	ychord_2 = 0
	ychord_1 = -(float(c))/(float(b))

	Lms_Line,  =  plt.plot([xchord_1, xchord_2],[ychord_1, ychord_2],'black', label='Lms Line')
	a,b,c = weights2
	xchord_1 = 0
	xchord_2 = -(float(c))/(float(a))
	ychord_2 = 0
	ychord_1 = -(float(c))/(float(b))

	Perceptron_Line, = plt.plot([xchord_1, xchord_2],[ychord_1, ychord_2], 'Green', label='Perceptron Line')
	plt.legend([Lms_Line, Perceptron_Line],['Lms Line', 'Perceptron Line'], loc=2)
	plt.show()
def single_sample_perceptron_ques_6(X, Y, learning_rate, margin):
	#Obtained the values as required (Augmented and negated)
	X, Y, train_set, dim = normalise_train_set(X, Y)
	
	i=0
	n = len(train_set)
	k=0
	weights = [1, 1, 1]
	print weights
	count = 0
	while i!=n:
		k = (k+1)%n
		count+=1
		if np.dot(train_set[k],weights)<=margin:
			weights = weights + learning_rate * train_set[k]
			i=0
		else:
			i=i+1
	print weights
	return weights

def plot_boundary(weights, train_set, X, Y):
	# plot data-points
	x_points = X[:,0]
	y_points = X[:,1]
	length = len(x_points)

	x_points_1 = x_points[0:length/2]
	x_points_2 = x_points[length/2:length]
	y_points_1 = y_points[0:length/2]
	y_points_2 = y_points[length/2:length]
	
	plt.plot(x_points_1,y_points_1,'ro');
	plt.axis([0,10,0,10])
	plt.plot(x_points_2,y_points_2,'bo');
	

	a,b,c = weights
	xchord_1 = 0
	xchord_2 = -(float(c))/(float(a))
	ychord_2 = 0
	ychord_1 = -(float(c))/(float(b))

	plt.plot([xchord_1, xchord_2],[ychord_1, ychord_2],'black')
	plt.show()	

def predict(test_set, weights):
	test_set = augment_vector(test_set)
	print weights
	pred_list = []
	for i in range(len(test_set)):
		print test_set[i]
		if np.dot(test_set[i], weights)<0:
			pred_list.append(2)
		elif np.dot(test_set[i], weights)>0:
			pred_list.append(1)

	return pred_list



def compute_accuracy(pred_labels_, Y_test):
	count=0
	length = len(pred_labels_)
	for k in range(len(pred_labels_)):
		if pred_labels_[k]==Y_test[k]:
			count = count+1

	accuracy = (float(count)/float(length))*100
	return accuracy


if __name__ == '__main__':
	X= [(1, 6), (7, 2), (8, 9), (9, 9), (4, 8), (8, 5), (2, 1), (3, 3), (2, 4), (7, 1), (1, 3), (5, 2)]
	Y  = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
	testset = [(0, 1), (8, 1), (2, 6), (2, 4.5), (6, 1.5), (4, 3)]
	test = [(6 , 4) ,(6, 6), (9 , 4),  (0, 0),  (0 ,-2) , (1, 1) ]
	Y_test = [ 1, 1, 1, 2, 2, 2]
	Xnew= [(1, 6), (7, 6), (8, 9), (9, 9), (4, 8), (8, 5), (2, 1), (3, 3), (2, 4), (7, 1), (1, 3), (5, 2)]
	X_no_sep= [(2, 1), (7, 2), (2, 4), (9, 9), (4, 8), (5, 2),(1, 6), (3, 3), (8, 9), (7, 1), (1, 3), (8, 5)]
	
	print 'Enter the type of algo to follow: '
	print "Enter 1...........Single-sample perceptron"
	print "Enter 2...........Single-sample perceptron with margin"
	print "Enter 3...........Relaxation algorithm with margin"
	print "Enter 4...........Widrow-Hoff or Least Mean Squared (LMS) Rule"
	print "Enter 5...........Solutions Algin Case"
		
	choice = input("Enter Choice\n")

	if choice == 1:
		learning_rate = 1.0
		margin = 0.0
		weights = single_sample_perceptron(X, Y, learning_rate, margin)	
		pred_labels = predict(test, weights)
		print "Predicted Classes: ",
		print pred_labels
		accuracy = compute_accuracy(pred_labels, Y_test)
		print 'Accuracy for the test data set is  {}%'.format(accuracy)
		

	elif choice == 2:
		learning_rate = 1.0
		margin = 2.0
		weights = single_sample_perceptron(X, Y, learning_rate, margin)
		pred_labels = predict(test, weights)
		print "Predicted Classes: ",
		print pred_labels
		accuracy = compute_accuracy(pred_labels, Y_test)
		print 'Accuracy for the test data set is  {}%'.format(accuracy)
		
	
	elif choice == 3:
		learning_rate = 1.0
		margin = 1.0
		weights = relaxation_algo_with_margin(X, Y, learning_rate, margin)
		pred_labels = predict(test, weights)
		print "Predicted Classes: ",
		print pred_labels
		accuracy = compute_accuracy(pred_labels, Y_test)
		print 'Accuracy for the test data set is  {}%'.format(accuracy)
		
		print 'Non Separable Data Plotting is not possible!!'
		#weights = relaxation_algo_with_margin(X_no_sep, Y, learning_rate, margin)
		

	elif choice==4:
		learning_rate =0.01
		margin = 1.0
		print 'Original Plot'
		weights = least_mean_square(X, Y, learning_rate, margin)
		print 'Test Data Set'
		pred_labels = predict(test, weights)
		print "Predicted Classes: ",
		print pred_labels
		accuracy = compute_accuracy(pred_labels, Y_test)
		print 'Accuracy for the test data set is  {}%'.format(accuracy)
		print 'Non Separable Data Plotting'
		weights = least_mean_square(X_no_sep, Y, learning_rate, margin)

	elif choice==5:
		learning_rate =0.01
		margin = 1.0
		print 'Properly Classified : Solutions Align : Q-6'	
		X= [(1, 6), (7, 2), (8, 9), (9, 9), (4, 8), (8, 5), (2, 1), (3, 3), (2, 4), (7, 1), (1, 3), (5, 2)]
		Xnew= [(1, 5.6), (7, 4.3), (8, 9), (9, 9), (4, 8), (8, 4.4), (2, 1), (3, 3), (2, 4), (7, 0.5), (1, 3), (5, 2)]	
		weights = ques_6_least_mean_square(Xnew, Y, learning_rate, margin)
	
