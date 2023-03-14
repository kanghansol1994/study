from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 
from sklearn.metrics import r2_score
import pandas as pd
#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets['target']

print(type(x)) # <class 'numpy.ndarray'>
print(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,random_state=333,
)
scaler = MinMaxScaler()
# scaler.fit(x) #x를 바꿀 준비하라
# x = scaler.transform(x) #x를 바꿔라
x_train = scaler.fit_transform(x_train)#위의 두줄을 이 한줄로 줄일 수 있음
x_test = scaler.transform(x_test)



'''
#2. 모델구성
# model = Sequential()
# model.add(Dense(30, input_shape=(13,)))
# model.add(Dense(20))
# model.add(Dense(10))
# model.add(Dense(1))

input1 = Input(shape=(13,))
dense1 = Dense(30)(input1)
dense2 = Dense(20, activation='relu')(dense1)
dense3 = Dense(10)(dense2)
output1 = Dense(1)(dense3)
model = Model(inputs = input1, outputs = output1)

# model.save('./_save/keras26_1_save_model.h5')


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor = 'val_loss', patience=10, mode='min',
                   verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto',
        verbose=1,
        save_best_only=True,
        filepath='./_save/MCP/keras27_ModelCheckPoint1.hdf5'
                      ) #shift+tap = 왼쪽으로 옮겨짐 tab=오른쪽으로 옮겨짐
#가장 좋은 지점 하나만 저장됨

model.fit(x_train, y_train, epochs=10000, 
          callbacks=[es, mcp],
          validation_split=0.2)
'''
model = load_model('./_save/MCP/keras27_ModelCheckPoint1.hdf5')


#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)
