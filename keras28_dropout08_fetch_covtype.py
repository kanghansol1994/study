#과적합 방지:데이터량을 늘리기 , 노드양을 일부 빼기(Dropout)

# 저장할때 평가결과값, 훈련시간등을 파일에 넣어줘

#save_model과 비교

from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 
from sklearn.metrics import r2_score
import pandas as pd
from keras.utils import to_categorical
#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets['target']

y = to_categorical(y)
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

input1 = Input(shape=(54,))
dense1 = Dense(5, activation='relu')(input1)
dense2 = Dense(5, activation='relu')(dense1)
dense3 = Dense(5, activation='relu')(dense2)
dense4 = Dense(10, activation='relu')(dense3)
output1 = Dense(8, activation='linear')(dense4)
model = Model(inputs = input1, outputs = output1)


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
es = EarlyStopping(monitor = 'val_loss', patience=10, mode='min',
                   verbose=1, 
                   restore_best_weights=True
                ) 
mcp = ModelCheckpoint(monitor='val_loss', mode='auto',
        verbose=1,
        save_best_only=True,
        filepath= "".join([filepath, 'k27_', date, '_', filename])      
) 


model.fit(x_train, y_train, epochs=1000, batch_size=5000,
          callbacks=[es], #mcp
          validation_split=0.2)


#4. 평가, 예측
print("==================1. 기본 출력======================")
loss = model.evaluate(x_test,y_test, verbose=0)
print('loss : ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)


