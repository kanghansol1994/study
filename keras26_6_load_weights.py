from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model,load_model
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

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)


#2. 모델
input1 = Input(shape=(13,))
dense1 = Dense(30)(input1)
dense2 = Dense(20, activation='relu')(dense1)
dense3 = Dense(10)(dense2)
output1 = Dense(1)(dense3)
model = Model(inputs = input1, outputs = output1)
# model.save('./_save/keras26_1_save_model.h5')
# model = load_model('./_save/keras26_3_save_model.h5')
###############################################################
# model.load_weights('./_save/keras26_5_save_weights1.h5')
#얘는 초기 랜덤값의 웨이트만 저장되어 있음
################################################################

model.load_weights('./_save/keras26_5_save_weights2.h5')



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

# model.fit(x_train, y_train, epochs=200, batch_size=12,
#           validation_split=0.2,
#           verbose=1,)


#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)


#모델 구성 다음에 세이브하면 모델만 저장
#컴파일과 훈련 다음에 세이브하면 가중치까지 저장
#최적의 weight로 저장 되어 있기때문에 몇번을 돌려도 같은 값이 나옴

