import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, LSTM, GRU, Conv1D, Conv2D, SimpleRNN, Concatenate, concatenate, Dropout, Bidirectional, Flatten, MaxPooling2D, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import datetime
from tensorflow.keras.layers import concatenate, Concatenate
import random
import tensorflow as tf
#0.시드 초기화
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

#1.데이터
path = './_data/시험/'
path_save = './_save/samsung/'
samsung = pd.read_csv(path + '삼성전자 주가2.csv', encoding='cp949', index_col=0) 
hyundae = pd.read_csv(path + '현대자동차.csv', encoding='cp949', index_col=0)
# print(samsung) #(3260, 16)
# print(samsung.columns) 
# print(samsung.info())
# print(samsung.describe())
# print(hyundae) #(3140, 16)
# print(hyundae.columns)
# print(hyundae.info())
# print(hyundae.describe())
x1 = samsung[['종가','고가','저가','거래량','등락률','개인','기관','외국계']]
x2 = hyundae[['종가','고가','저가','거래량','등락률','개인','기관','외국계']]
y = samsung[['시가']]
# print(x1.shape) #(3260, 8)
# print(x2.shape) #(3140, 8)
# print(y.shape) #(3260, 1)
# print(x1.columns)
# print(x1.info())
x1 = x1.replace(',', '', regex=True).astype(float)
x2 = x2.replace(',', '', regex=True).astype(float)
y = y.replace(',', '', regex=True).astype(float)

x1 = x1.to_numpy()
x2 = x2.to_numpy()
y= y.to_numpy()

x1 = x1[:300][::-1]
y = y[:300][::-1]
x2 = x2[:300][::-1]
print(x1.shape) #(300, 8)
print(x2.shape) #(300, 8)
print(y.shape) #(300, 1)

timesteps = 20

def split_x(datasets, timesteps):
    x = []
    for i in range(len(datasets) - timesteps -1 ):
        subset = datasets[i : (i + timesteps)]
        x.append(subset)
    return np.array(x)
x1 = split_x(x1, timesteps)
x2 = split_x(x2, timesteps)
y = y[timesteps+1:]

print(x1)
print(x1.shape) #(279,20,8)
print(x2)
print(x2.shape) #(279,20,8)
print(y)
print(y.shape)  #(279,1)


x1_train, x1_test, x2_train, x2_test, y_train,y_test = train_test_split(x1, x2, y, train_size=0.8, shuffle=False)

print(x1_train.shape) #(223,20,8)
print(x1_test.shape) #(56,20,8)

x1_train = x1_train.reshape(223, 160)
x1_test = x1_test.reshape(56, 160)
x2_train = x2_train.reshape(223, 160)
x2_test = x2_test.reshape(56, 160)

scaler = MinMaxScaler()
x1_train = scaler.fit_transform(x1_train)
x1_test = scaler.transform(x1_test)
x2_train = scaler.transform(x2_train)
x2_test = scaler.transform(x2_test)

x1_train = x1_train.reshape(223, 20,8)
x1_test = x1_test.reshape(56, 20,8)
x2_train = x2_train.reshape(223, 20,8)
x2_test = x2_test.reshape(56, 20,8)

#2. 모델구성
input1 = Input(shape=(20,8))
dense1 = LSTM(10, return_sequences=True, activation='relu', name='samsung1')(input1)
dense2 = Dense(20, activation='relu', name='samsung2')(dense1)
dense3 = Conv1D(20, 4, activation='relu', name='conv1d')(dense2)
dense4 = Flatten(name='flatten')(dense3)
dense5 = Dense(128, activation='relu', name='samsung3')(dense4)
dense6 = Dense(64, activation='relu', name='samsung4')(dense5)
output1 = Dense(100, activation='relu', name='output1')(dense6)

#2. 모델2
input2 = Input(shape=(20, 8))
dense11 = LSTM(10, activation='relu', name='huyndae1')(input2)
dense12 = Dense(128, activation='relu', name='huyndae2')(dense11)
dense13 = Dense(64, activation='relu', name='huyndae3')(dense12)
dense14 = Dense(32, activation='relu', name='huyndae4')(dense13)
output2 = Dense(100, activation='relu', name='output2')(dense14)

#2.머지
merge1 = concatenate([output1, output2], name='mg1')
merge2 = Dense(128, activation='relu', name='mg2')(merge1)
merge3 = Dense(64, activation='relu', name='mg3')(merge2)
merge4 = Dense(32, activation='relu', name='mg4')(merge3)
merge5 = Dense(16, activation='relu', name='mg5')(merge4)
last_output = Dense(1, name='last')(merge5)

model = Model(inputs=[input1, input2], outputs=last_output)

model.summary()

#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', mode='min', patience=100, restore_best_weights=True)
hist = model.fit([x1_train, x2_train], y_train, epochs=300, batch_size=32, validation_split=0.2, callbacks=[es])

model.save(path_save + 'keras53_samsung3_khs.h5')

# 4. 평가, 예측
loss = model.evaluate([x1_test, x2_test], y_test)
print('loss : ', loss)
1
y_predict = model.predict([x1_test, x2_test])

r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

print('y_predict = ', y_predict[-1:])
import matplotlib.pyplot as plt
plt.plot(range(len(y_test)),y_test,label='data',c='orange')
plt.plot(range(len(y_test)),y_predict,label='model')
plt.legend()
plt.show()


# loss :  3254626.25
# r2스코어 :  -0.5136741212938492
# y_predict =  [[62402.36]]

#삼성전자와 현대자동차 주가로 삼성전자 주가 맞추기

#두개의 csv 각각 데이터에서 컬럼 7개 이상 추출(그 중 거래량은 반드시 들어갈 것)
#timesteps 와 feautre는 알아서 자르기

#제공된 데이터 외 추가 데이터 사용 금지

#1. 삼성전자 28일(화) 종가 맞추기 (점수 배점 0.3)
#2. 현대자동차 29일(수) 아침 시가 맞추기 (점수배점 0.7)

#앙상블 모델 사용해야만 함

#마감시간 : 27일(월) 23시59분59초 / 28일(화) 23시59분59초
#메일 제목 : 윤영선 [삼성 1차] 60,350.07원
#                  [삼성 2차] 60,350.07원
#첨부파일 : keras53_samsung2_khs_submit.py
#          keras53_samsung4_khs_submit.py
#가중치     _save/samsung/keras53_samsung2_khs.h5 / hdf5
#          _save/samsung/keras53_samsaung4_yys.h5 / hdf5
