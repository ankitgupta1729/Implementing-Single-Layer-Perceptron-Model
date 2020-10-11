#!/usr/bin/env python
# coding: utf-8

# # Task 2

# In[85]:


#Importing required libraries
import numpy as np
import matplotlib.pyplot as plt
import sys


# In[86]:


#np.random.seed(42)

#Genrating normally distributed random numbers by pseudorandom generator
#each class has 50 data points and class-labels are 0,1 and 2  
#toal features(dimensions) are 2
# each class is linearly-separable
class_0 = np.random.randn(50, 2) + np.array([0, -3]) # this class is centred around point (0,-3)
class_1 = np.random.randn(50, 2) + np.array([3, 3])  # this class is centred around point (3,3)
class_2 = np.random.randn(50, 2) + np.array([-3, 3]) # this class is centred around point (-3,3)


dataset = np.vstack([class_0,class_1,class_2]) #combining 50 data points of class 0, 50 data points of class 1,

#50 data points of class 2 in a vertical fashion or think like I have put it in a vertical stack

labels = np.array([0]*50 + [1]*50 + [2]*50) #assigning class-labels, initial 50 data-points as class 0,then 
#class-1 and then clas-2

vect = np.zeros((150, 3)) # creating a 150*3 size matrix 

for i in range(150):
    vect[i, labels[i]] = 1 #here, 150*3 matrix vect has 3 columns in which column-0 has values as 0 for those data
#points which don't have class-label as 0 and column-1 has values as 0 for those data
#points which don't have class-label as 1 and column-2 has values as 0 for those data
#points which don't have class-label as 2 


# In[87]:


plt.scatter(dataset[:,0], dataset[:,1], c=labels)
plt.show() # it shows the linaerly-separable data points 


# In[88]:


bias=1 #assuming bias as 1
data=[]
for i in range(150):
    temp=[]
    temp.append(bias)
    temp.append(dataset[i][0])
    temp.append(dataset[i][1])
    temp.append(labels[i])
    data.append(temp) # creating 150*4 size matrix in which fetarures are x0,x1,x2 where x0 is bias and x1,x2 
    # are features, last column represents the class-label
data #display the dataset    


# In[89]:


vect   #display the above created 150*3 matrix


# ## Estimating input weight vector for class 0

# In[90]:


# For class 0

weights_0=[0.20,1.00,-1.00] # initializing weights

def predict(inputs,weights_0): # This function takes values of features (x0,x1,x2) 
    
    #one by one from dataset and weight vector as input and return the value according 
    #to the definition of standard sigmoid function  
    
    threshold = 0.0 # I have set the threshold as 0 
    v = 0.0
    for input,weight in zip(inputs,weights_0):
        v += input*weight
    return 1 if v >= threshold else 0.0    

def accuracy(matrix,weights_0): # This function gives the accuracy on the scale of 0 to 1 
    # in terms of how many points are correctly out of total number of data points   
    
    num_correct=0.0 #initialized total correct points as 0 
    preds=[]
    for i in range(len(matrix)):
        pred=predict(matrix[i][:-1],weights_0)
        preds.append(pred)
        if pred==vect[i][0]: num_correct += 1.0 #if data point is correctly classified then number of 
            #correctly classified points are added by 1
            
    print("Predictions:",preds)
    return num_correct/float(len(matrix))
    
def train_weights(matrix,weights_0,iterations=1000,l_rate=1.0): # This function is to train weights
    #According to perceptron convergence theorem, after number of iterations, maximum accuracy will be reached 
    # and after that weight vector will not be changed. Since, generated data is of random numbers.
    #So, each time when we run the cell, dataset will be different and so, number of iterations 
    #to get the estimated weight vector with accuracy 1.0 will be different and so accordingly you can change
    # the value of number of iterations.
    for epoch in range(iterations):
        cur_acc=accuracy(matrix,weights_0)
        print("\nIteration  %d \nWeights:  "%epoch,weights_0)
        print("Accuracy: ", cur_acc )
        if cur_acc==1.0 and True : break
            
        for i in range(len(matrix)):
            prediction = predict(matrix[i][:-1],weights_0)
            error = vect[i][0] - prediction #to get the difference between predicted and actual class-label
            
            if True:
                print("Training on data at index %d..."%i)
            for j in range(len(weights_0)):
                if True:
                    sys.stdout.write("\t Weight[%d]: %0.5f ---> "%(j,weights_0[j]))
                weights_0[j]= weights_0[j] + (l_rate*error*matrix[i][j]) # weight is modified 
                #according to the definition
                if True: sys.stdout.write("%0.5f\n"%weights_0[j])
    print("\n Final estimated weight vector with accuracy %0.5f is:"%cur_acc)
    return weights_0    

train_weights(data, weights_0=weights_0,iterations=1000,l_rate=1.0)


# ## Estimating input weight vector for class 1

# In[91]:


# For class 1

weights_1=[0.20,1.00,-1.00] # initializing weights

def predict(inputs,weights_1): # This function takes values of features (x0,x1,x2) 
    
    #one by one from dataset and weight vector as input and return the value according 
    #to the definition of standard sigmoid function  
    
    threshold = 0.0 # I have set the threshold as 0 
    v = 0.0
    for input,weight in zip(inputs,weights_1):
        v += input*weight
    return 1 if v >= threshold else 0.0    

def accuracy(matrix,weights_1): # This function gives the accuracy on the scale of 0 to 1 
    # in terms of how many points are correctly out of total number of data points   
    
    num_correct=0.0 #initialized total correct points as 0 
    preds=[]
    for i in range(len(matrix)):
        pred=predict(matrix[i][:-1],weights_1)
        preds.append(pred)
        if pred==vect[i][1]: num_correct += 1.0 #if data point is correctly classified then number of 
            #correctly classified points are added by 1
            
    print("Predictions:",preds)
    return num_correct/float(len(matrix))
    
def train_weights(matrix,weights_1,iterations=1000,l_rate=1.0): # This function is to train weights
    #According to perceptron convergence theorem, after number of iterations, maximum accuracy will be reached 
    # and after that weight vector will not be changed. Since, generated data is of random numbers.
    #So, each time when we run the cell, dataset will be different and so, number of iterations 
    #to get the estimated weight vector with accuracy 1.0 will be different and so accordingly you can change
    # the value of number of iterations.
    for epoch in range(iterations):
        cur_acc=accuracy(matrix,weights_1)
        print("\nIteration  %d \nWeights:  "%epoch,weights_1)
        print("Accuracy: ", cur_acc )
        if cur_acc==1.0 and True : break
            
        for i in range(len(matrix)):
            prediction = predict(matrix[i][:-1],weights_1)
            error = vect[i][1] - prediction #to get the difference between predicted and actual class-label
            
            if True:
                print("Training on data at index %d..."%i)
            for j in range(len(weights_1)):
                if True:
                    sys.stdout.write("\t Weight[%d]: %0.5f ---> "%(j,weights_1[j]))
                weights_1[j]= weights_1[j] + (l_rate*error*matrix[i][j]) # weight is modified 
                #according to the definition
                if True: sys.stdout.write("%0.5f\n"%weights_1[j])
    print("\n Final estimated weight vector with accuracy %0.5f is:"%cur_acc)
    return weights_1

train_weights(data, weights_1=weights_1,iterations=1000,l_rate=1.0)


# ## Estimating input weight vector for class 2

# In[92]:


# For class 2

weights_2=[0.20,1.00,-1.00] # initializing weights

def predict(inputs,weights_2): # This function takes values of features (x0,x1,x2) 
    
    #one by one from dataset and weight vector as input and return the value according 
    #to the definition of standard sigmoid function  
    
    threshold = 0.0 # I have set the threshold as 0 
    v = 0.0
    for input,weight in zip(inputs,weights_2):
        v += input*weight
    return 1 if v >= threshold else 0.0    

def accuracy(matrix,weights_2): # This function gives the accuracy on the scale of 0 to 1 
    # in terms of how many points are correctly out of total number of data points   
    
    num_correct=0.0 #initialized total correct points as 0 
    preds=[]
    for i in range(len(matrix)):
        pred=predict(matrix[i][:-1],weights_2)
        preds.append(pred)
        if pred==vect[i][2]: num_correct += 1.0 #if data point is correctly classified then number of 
            #correctly classified points are added by 1
            
    print("Predictions:",preds)
    return num_correct/float(len(matrix))
    
def train_weights(matrix,weights_2,iterations=1000,l_rate=1.0): # This function is to train weights
    #According to perceptron convergence theorem, after number of iterations, maximum accuracy will be reached 
    # and after that weight vector will not be changed. Since, generated data is of random numbers.
    #So, each time when we run the cell, dataset will be different and so, number of iterations 
    #to get the estimated weight vector with accuracy 1.0 will be different and so accordingly you can change
    # the value of number of iterations.
    for epoch in range(iterations):
        cur_acc=accuracy(matrix,weights_2)
        print("\nIteration  %d \nWeights:  "%epoch,weights_2)
        print("Accuracy: ", cur_acc )
        if cur_acc==1.0 and True : break
            
        for i in range(len(matrix)):
            prediction = predict(matrix[i][:-1],weights_2)
            error = vect[i][2] - prediction #to get the difference between predicted and actual class-label
            
            if True:
                print("Training on data at index %d..."%i)
            for j in range(len(weights_2)):
                if True:
                    sys.stdout.write("\t Weight[%d]: %0.5f ---> "%(j,weights_2[j]))
                weights_2[j]= weights_2[j] + (l_rate*error*matrix[i][j]) # weight is modified 
                #according to the definition
                if True: sys.stdout.write("%0.5f\n"%weights_2[j])
    print("\n Final estimated weight vector with accuracy %0.5f is:"%cur_acc)
    return weights_2

train_weights(data, weights_2=weights_2,iterations=1000,l_rate=1.0)

