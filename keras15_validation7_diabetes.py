import tensorflow as tf
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import load_diabetes
import random

# 0. seed initialization
seed=20580
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. data prepare
datasets=load_diabetes()
x=datasets.data
y=datasets.target

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,random_state=seed)


 # 2. model build
model=Sequential()
model.add(Dense(1,input_dim=10,activation='linear'))
model.add(Dense(1,activation='linear'))
model.add(Dense(1,activation='linear'))
model.add(Dense(1,activation='linear'))

# 3. compile, training
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,batch_size=1000,epochs=10000,validation_split=0.2)


# 4. evaluate, predict
loss=model.evaluate(x_test,y_test)
print(f'loss : {loss}')
y_predict=model.predict(x_test)
r2=r2_score(y_test,y_predict)
print(f'r2 score : {r2}')
print(f'seed : {seed}')