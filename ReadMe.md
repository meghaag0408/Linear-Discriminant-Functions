********************************************************************************************************************************************
SMAI - Assignment1
Name - Megha Agarwal
Roll No - 201506511
Course - M.Tech CSIS ( PG1)
********************************************************************************************************************************************
____________________________________________________________________________________________________________________________________________
Preamble
____________________________________________________________________________________________________________________________________________
The idea of this assignment is to explore the material presented as part of Chapter 5 on constructing Linear Discriminant Functions based on various approaches such as Perceptron criterion function and Mean Squared Error.


____________________________________________________________________________________________________________________________________________
Aim/Requirement:
____________________________________________________________________________________________________________________________________________

Suppose now that we have a set of n samples y 1 , ..., y n , some labelled ω 1 and some labelled ω 2 . We want to use these samples to determine the weights a in a linear discriminant function g(x) = a t y. We tend  to look for a weight vector that classifies all of the samples correctly. If such a weight vector exists, the samples are said to be linearly separable.

Given there are two class w1 and w2. And there are n samples. We need to determine the weight vector a such that it properly classified the n samples in the two classes w1 and w2.

The algorithm that can be used to classify the n samples in two classes are :
	*Single-sample perceptron
	*Single-sample perceptron with margin
	*Relaxation algorithm with margin
	*Widrow-Hoff or Least Mean Squared (LMS) Rule


The detailed algorithms and the updation/relaxation rules are given in the project report attached.
____________________________________________________________________________________________________________________________________________
Implementation:
____________________________________________________________________________________________________________________________________________
-> Merge the two given classes into a training set which is used to determine the weight vector.
-> Convert the n samples in the form of array (numpy library is used for the same).
-> g(x) = w1x1 + w2x2 + w0 
-> Classification :
	if g(x) > 0 -> It belongs to Class1
	if g(x) <0 -> It belongs to class2
	if g(x) =0 -> It lies on the hyperplane separating the two classes.
-> The given samples are augmented by adding 1 at the adding making the g(x) value as:
	g(x) = ax
	
-> Negate one of the classes so that they are reflected on the same side of the hyperplane as the other class.
-> Now, if ay>0 then the sample is considered to be classified, otherwise missclassified.
-> The sample on the hyperplane is assumed to be classiified.
-> Starting with any arbitary weight vector and margin and learning rate value, the given algos updates the weight vector which classifies the two classes properly.
-> The weight vector is plotted using the corresponding straight line equation - ax + by + c ===> w1x1 + w2x2 + w0


____________________________________________________________________________________________________________________________________________
Instructions to use the code:
____________________________________________________________________________________________________________________________________________
-> The code is run using python filename.py without giving any command line arguments.
-> The window prompts with the data:

	Enter the type of algo to follow: 
	Enter 1...........Single-sample perceptron
	Enter 2...........Single-sample perceptron with margin
	Enter 3...........Relaxation algorithm with margin
	Enter 4...........Widrow-Hoff or Least Mean Squared (LMS) Rule
	Enter 5...........Solutions Algin Case
	Enter Choice
	1

-> The value of margin, learning rate and weight vector are to be changed before compiling/running the code so as to analyse different results. e.g. covnergence time etc.


____________________________________________________________________________________________________________________________________________
Libraries/Packages Needed
____________________________________________________________________________________________________________________________________________
These are all the packages that are required so as to the run the code:
	import matplotlib.pyplot as plt
	import numpy as np
	import pylab as pl
	import math as math

********************************************************************************************************************************************

