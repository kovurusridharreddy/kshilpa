# -*- coding: utf-8 -*-
"""
Created on Tue May 15 17:26:48 2018

@author: Suprim-FC
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt





data=pd.read_csv("E:\Machine_Learning_AZ_Template_Folder\Machine Learning A-Z Template Folder\Part 2 - Regression\Section 4 - Simple Linear Regression\Salary_Data.csv")

x1=data.iloc[:, :-1].values
y=data.iloc[:, -1].values

y=y.reshape((len(y),1))

theta=np.zeros((1,2))

theta=np.transpose(theta)

x0=np.ones((len(x1),1))

x=np.concatenate((x0,x1),axis=1)

#x=np.transpose(x)

predict=gradient(x,y,theta,0.001)




y_pred2=theta[0]+theta[1]*1.1

def estimate_error(y,predict):
    m=len(y)
    return ((y-predict))/(2*m)
    

def update_theta(x,theta,error,m,x_var,alpha):
    theta[0]=theta[0]+(error*x_var[0]*alpha)
    theta[1]=theta[1]+(error*x_var[1]*alpha)
    return theta
    
        
def gradient(x,y,theta,alpha):
    m=len(y)
    for i in range(0,m-1):
        x_var=x[i]
        y_var=y[i]
        predict_y= np.dot(x_var,theta)
        error=estimate_error(predict_y,y_var)
        theta=update_theta(x,theta,error,m,x_var,alpha)
    return theta
        #return tehta
        

"""
def update_theta(x,theta,error,m,x_var,alpha):
   # for i in range(0,len(theta)-1):
        #    theta[i]=theta[i]+(error*x_var[i]*alpha)
    theta[0]=theta[0]+(error*x_var[0]*alpha)
    theta[1]=theta[1]+(error*x_var[0]*alpha)
    return theta


def estimate_error(predict,y_var):
    m=len(y)
    return ((predict-y_var))/(2*m)

def gradient(x,y,theta,alpha):
    m=len(y)
    x_var=x[0]
    y_var=y[0]
    predict_y= np.dot(x_var,theta)
    error=estimate_error(predict_y,y_var)
    theta=update_theta(x,theta,error,m,x_var,alpha)
    return theta,error,predict_y
        #return tehta

"""

