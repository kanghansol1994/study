import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D,Flatten

#1. 데이터
#2. 모델

model = Sequential()
# model.add(LSTM(10, input_shape =(3,1))) #토탈파라미터:541
model.add(Conv1D(10, 2, input_shape=(3,1))) #토탈:141
model.add(Conv1D(10, 2))                   #토탈:301 conv1d는 conv2d와 파라미터 계산식 같음
model.add(Flatten())
model.add(Dense(5))
model.add(Dense(1))

model.summary()