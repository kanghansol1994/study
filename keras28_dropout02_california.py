from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 
from sklearn.metrics import r2_score
import pandas as pd
#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8,random_state=333,
)
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)#위의 두줄을 이 한줄로 줄일 수 있음
x_test = scaler.transform(x_test)




#2. 모델구성
# input1 = Input(shape=(13,))
# dense1 = Dense(30)(input1)
# drop1 = Dropout(0.3)(dense1)
# dense2 = Dense(20, activation='relu')(drop1)
# drop2 = Dropout(0.2)(dense2)
# dense3 = Dense(10)(drop2)
# drop3 = Dropout(0.5)(dense3)
# output1 = Dense(1)(drop3)
# model = Model(inputs = input1, outputs = output1)

model = Sequential()
model.add(Dense(4, activation='relu', input_shape=(8,)))
model.add(Dropout(0.1))
model.add(Dense(6, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

import datetime
date = datetime.datetime.now()
print(date) #2023-03-14 11:11:20.871391
date = date.strftime('%m%d_%H%M')
print(date)# 0314_1115

filepath = './_save/MCP/keras27_4/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'


from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor = 'val_loss', patience=20, mode='min',
                   verbose=1, 
                   restore_best_weights=True
                ) 
mcp = ModelCheckpoint(monitor='val_loss', mode='auto',
        verbose=1,
        save_best_only=True,
        filepath= "".join([filepath, 'k27_', date, '_', filename])      
) 


model.fit(x_train, y_train, epochs=10000, batch_size=128,
          callbacks=[es], #mcp
          validation_split=0.2)


#4. 평가, 예측
print("==================1. 기본 출력======================")
loss = model.evaluate(x_test,y_test, verbose=0)
print('loss : ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)


