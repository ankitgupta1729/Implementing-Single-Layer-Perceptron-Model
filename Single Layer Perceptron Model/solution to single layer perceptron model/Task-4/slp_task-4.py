#!/usr/bin/env python
# coding: utf-8

# ## Task 4

# In[2]:


#Importing python libraries for plotting graphs and some mathematical computation.
import numpy as np
from matplotlib import pyplot as plt
import sys


# In[3]:


# Generating Linearly Separarble data and considering it as a training data

data_points = np.random.randn(50,2) #50 randomly generated data points with 2 features(dimensions) using 
# pseudorandom number generator

y=[] # creating class-label
for i in range(50):
    y.append(0) # initializing class-label as 0

# I have considered the line 10x+10y=1 for separating the randomly generated data points

for i in range(len(data_points)):
    if (10*data_points[i,0] - 1) < (10*data_points[i,1]):
        plt.scatter(data_points[i,0],data_points[i,1],c='r')
        y[i]=0 # red-points are labelled as 0
    else:
        plt.scatter(data_points[i,0],data_points[i,1],c='g')
        y[i]=1 # green-points are labelled as 1
        
plt.show()
data = [] #defining data matrix as input for features x0(bias),x1,x2

Bias=1 # assuming value of bias as 1  

for i in range(50):
    temp=[]
    temp.append(Bias) #appending bias
    temp.append(data_points[i,0]) #appending value of x1
    temp.append(data_points[i,1]) #appending value of x2
    temp.append(y[i])#appending trained class-label
    data.append(temp)   
for i in range(50):
    print("our dataset is:")
    print(data[i])


# In[4]:


# Implementing the single layer perceptron model


weights=[0.20,1.00,-1.00] # initializing weights

def predict(inputs,weights): # This function takes values of features (x0,x1,x2) 
    
    #one by one from dataset and weight vector as input and return the value according 
    #to the definition of standard sigmoid function  
    
    threshold = 0.0 # I have set the threshold as 0 
    v = 0.0
    for input,weight in zip(inputs,weights):
        v += input*weight
    return 1 if v >= threshold else 0.0    

def accuracy(matrix,weights): # This function gives the accuracy on the scale of 0 to 1 
    # in terms of how many points are correctly out of total number of data points   
    
    num_correct=0.0 #initialized total correct points as 0 
    preds=[]
    for i in range(len(matrix)):
        pred=predict(matrix[i][:-1],weights)
        preds.append(pred)
        if pred==matrix[i][-1]: num_correct += 1.0 #if data point is correctly classified then number of 
            #correctly classified points are added by 1
            
    print("Predictions:",preds)
    return num_correct/float(len(matrix))
    
def train_weights(matrix,weights,iterations=1000,l_rate=1.0): # This function is to train weights
    #According to perceptron convergence theorem, after number of iterations, maximum accuracy will be reached 
    # and after that weight vector will not be changed. Since, generated data is of random numbers.
    #So, each time when we run the cell, dataset will be different and so, number of iterations 
    #to get the estimated weight vector with accuracy 1.0 will be diferent and so accordingly you can change
    # the value of number of iterations.
    for epoch in range(iterations):
        cur_acc=accuracy(matrix,weights)
        print("\nIteration  %d \nWeights:  "%epoch,weights)
        print("Accuracy: ", cur_acc )
        if cur_acc==1.0 and True : break
            
        for i in range(len(matrix)):
            prediction = predict(matrix[i][:-1],weights)
            error = matrix[i][-1] - prediction #to get the difference between predicted and actual class-label
            
            if True:
                print("Training on data at index %d..."%i)
            for j in range(len(weights)):
                if True:
                    sys.stdout.write("\t Weight[%d]: %0.5f ---> "%(j,weights[j]))
                weights[j]= weights[j] + (l_rate*error*matrix[i][j]) # weight is modified 
                #according to the definition
                if True: sys.stdout.write("%0.5f\n"%weights[j])
    print("\n Final estimated weight vector with accuracy %0.5f is:"%cur_acc)
    return weights    


# In[5]:


train_weights(data, weights=weights,iterations=1000,l_rate=1.0)

