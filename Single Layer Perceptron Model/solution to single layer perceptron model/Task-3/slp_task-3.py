#!/usr/bin/env python
# coding: utf-8

# In[54]:


from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import sys


# In[55]:


# Real life data is generally non-linealrly separable but using internet I got sonar dataset which is 
#linearly separable
# I have used sonar dataset which is linearly separable and normalized 
# It can be downloaded from 
# https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+%28Sonar%2C+Mines+vs.+Rocks%29
# https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/

data = pd.read_csv("/home/ankit/Desktop/sonar.csv")
data.columns=['x'+str(x) for x in range(60)]+['Material']
#data
data.shape


# In[56]:


data1= data.to_numpy() #converting dataframe "data" to numpy array "data1"
dataset=[]
bias=1
for i in range(len(data1)):
    temp=[]
    temp.append(bias)
    for j in range(61):
        temp.append(data1[i][j])
    dataset.append(temp) # appending 1st column as bias=1 and all other columns of matrix data1 
    # and created a new matrix of size 207*62 for our purpose
#print(dataset)    


# In[57]:


# the original dataset has class-labels as "M" and "R", so we have to convert it into 1 and 0
for i in range(len(dataset)):
    if dataset[i][-1]=='M':
        dataset[i][-1]=1
    else:
        dataset[i][-1]=0


# In[58]:


# here dataset has 62 dimensions which is much harder to visualize. 
# if it is of 2 dimensions then I can use plt.scatter() to visualize it easily
# for data visualization purpose, we have to reduce the dimensions using PCA or t-SNE like techniques
# and then visualize it


# In[59]:


# Implementing perceptron model

weights=[] # initializing weights
for i in range(62):
    weights.append(np.random.randn())


# In[60]:


# I have commented the print statements because datset is of size 207*62 and 
# for each iteration printing weight values, accuracy etc will take much cell space and 
#   browser's tab will be crashed. 


# In[73]:


def predict(inputs,weights): # This function takes values of features (x0,x1,x2,...x61) 
    
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
            
    #print("Predictions:",preds)
    return num_correct/float(len(matrix))
    
def train_weights(matrix,weights,iterations=10000,l_rate=2.0): # This function is to train weights
    #According to perceptron convergence theorem, after number of iterations, maximum accuracy will be reached 
    # and after that weight vector will not be changed. Since, generated data is of random numbers.
    #So, each time when we run the cell, dataset will be different and so, number of iterations 
    #to get the estimated weight vector with accuracy 1.0 will be diferent and so accordingly you can change
    # the value of number of iterations.
    for epoch in range(iterations):
        cur_acc=accuracy(matrix,weights)
        #print("\nIteration  %d \nWeights:  "%epoch,weights)
        #print("Accuracy: ", cur_acc )
        if cur_acc==1.0 and True : break
            
        for i in range(len(matrix)):
            prediction = predict(matrix[i][:-1],weights)
            error = matrix[i][-1] - prediction #to get the difference between predicted and actual class-label
            
            #if True:
                #print("Training on data at index %d..."%i)
            for j in range(len(weights)):
                #if True:
                    #sys.stdout.write("\t Weight[%d]: %0.5f ---> "%(j,weights[j]))
                weights[j]= weights[j] + (l_rate*error*matrix[i][j]) # weight is modified 
                #according to the definition
                #if True: sys.stdout.write("%0.5f\n"%weights[j])
    print("\n Final estimated weight vector with accuracy %0.5f is:"%cur_acc)
    return weights


# In[74]:


train_weights(dataset, weights=weights,iterations=10000,l_rate=2.0)


# In[ ]:




