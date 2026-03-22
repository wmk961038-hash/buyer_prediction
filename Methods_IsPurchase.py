import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from numpy import ndarray
from scipy.constants import torr


#sigmoid函数定义
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict_model(data,w):
    if not isinstance(data,np.ndarray) or not isinstance(w,np.ndarray):
        raise TypeError("Parameter type error in function predict_model")
    return sigmoid(data@w)

def loss_function(df,w):
    if not isinstance(df,pd.DataFrame) or not isinstance(w,np.ndarray):
        raise TypeError("Parameter type error in function loss_function")
    to_return=0
    for i in df.index:
        temp_data=df.iloc[i,[0,1,4]].values
        to_return+=-(df.iloc[i,2]*math.log(predict_model(temp_data,w))+(1-df.iloc[i,2])*math.log(1-predict_model(temp_data,w)))
    return to_return

#返回损失函数对所有变量的偏导函数构成的向量
def loss_function_partial(df,w):
    if not isinstance(df, pd.DataFrame) or not isinstance(w, np.ndarray):
        raise TypeError("Parameter type error in function loss_function")
    to_return = np.zeros(len(w))
    for i in df.index:
        temp_x=df.iloc[i,[0,1,4]].values
        temp_y=df.iloc[i,2]
        temp_vector=-(temp_y-predict_model(temp_x,w))*temp_x
        to_return+=temp_vector
    return to_return/len(df.index)