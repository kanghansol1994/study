import tensorflow as tf
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error

# 1. data
x = np.array(range(1,17))
y = np.array(range(1,17))

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

# 2. model build
model=Sequential()
model.add(Dense(1,input_dim=1))

# 3. compile, build
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=10,batch_size=len(x),validation_split=0.2)

# 4. evaluate
loss=model.evaluate(x_test,y_test)
print(f'loss : {loss}')