import tensorflow as tf
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,LeakyReLU
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing
import random

#0. seed initialization
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

#1. data prepare
datasets=fetch_california_housing()
# print(datasets.DESCR)

x=datasets.data
y=datasets.target

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.9,random_state=seed)

# 2. model build
model=Sequential()
model.add(Dense(16,input_dim=8,activation='sigmoid'))
model.add(Dense(16,activation=LeakyReLU()))
model.add(Dense(16,activation=LeakyReLU()))
model.add(Dense(16,activation=LeakyReLU()))
model.add(Dense(1))

# 3. compile,training
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,batch_size=1000,epochs=1000,validation_split=0.2)

# 4. evaluate,predict
loss = model.evaluate(x_test,y_test)
print(f'loss : {loss}')
y_predict = model.predict(x_test)
print(f'r2 : {r2_score(y_test,y_predict)}')