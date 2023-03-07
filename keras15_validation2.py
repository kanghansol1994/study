import tensorflow as tf
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
import pandas as pd
import random
import matplotlib as mpl
import matplotlib.pyplot as plt

# 0. seed initialization
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. data
x_train = np.array(range(1,17))
y_train = np.array(range(1,17))

x_train,x_test,y_train,y_test=train_test_split(x_train,y_train,train_size=10/16+0.0001,random_state=seed)
x_val,x_test,y_val,y_test=train_test_split(x_test,y_test,train_size=0.5,random_state=seed)
print(x_train.shape,x_val.shape,x_test.shape)

# 2. model
model=Sequential()
model.add(Dense(1,input_dim=1))

# 3. compile, training
model.compile(loss='mse',optimizer='adam')