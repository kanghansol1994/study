import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Concatenate, concatenate, Dropout, Bidirectional, Flatten, MaxPooling1D, Input,LeakyReLU
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
hyundae = pd.read_csv(path + '현대자동차2.csv', encoding='cp949', index_col=0)
samsung = pd.read_csv(path + '삼성전자 주가3.csv', encoding='cp949', index_col=0) 
# print(samsung) #(3260, 16)
# print(samsung.columns) 
# print(samsung.info())
# print(samsung.describe())
# print(hyundae) #(3140, 16)
# print(hyundae.columns)
# print(hyundae.info())
# print(hyundae.describe())
hyundae = hyundae[['시가','종가','고가','저가','거래량','등락률','개인','기관','외국계']]
samsung = samsung[['시가','종가','고가','저가','거래량','등락률','개인','기관','외국계']]

# print(x1.shape) #(2040, 8)
# print(x2.shape) #(2100, 8)
# print(y.shape) #(2040, 1)
# print(x1.columns)
# print(x1.info())
x1 = hyundae[['종가','고가','저가','거래량','등락률','개인','기관','외국계']]
x2 = samsung[['종가','고가','저가','거래량','등락률','개인','기관','외국계']]
y = hyundae[['시가']]
x1 = x1.replace(',', '', regex=True).astype(float)
x2 = x2.replace(',', '', regex=True).astype(float)
y = y.replace(',', '', regex=True).astype(float)
# print(x1)

x1 = x1.to_numpy()
x2 = x2.to_numpy()
y= y.to_numpy()


x1 = x1[:200][::-1]
y = y[:200][::-1]
x2 = x2[:200][::-1]
# print(x1.shape) #(200, 8)
# print(x2.shape) #(200, 8)
# print(samsung_y.shape) #(200, 1)

timesteps = 10

def split_x(datasets, timesteps):
    x = []
    for i in range(len(datasets) - timesteps -1 ):
        subset = datasets[i : (i + timesteps)]
        x.append(subset)
    return np.array(x)
x1 = split_x(x1, timesteps)
x2 = split_x(x2, timesteps)
y = y[timesteps+1:]

# print(x1)
# print(x1.shape) #(280,20,8)
# print(x2)
# print(x2.shape) #(1460,5,8)
# print(y)
# print(y.shape)  #(1460,1)


x1_train, x1_test, x2_train, x2_test, y_train,y_test = train_test_split(x1, x2, y, train_size=0.8, shuffle=False)

print(x1_train.shape) #(1168,30,8)
print(x1_test.shape) #(292,30,8)


x1_train = x1_train.reshape(151, 80)
x1_test = x1_test.reshape(38, 80)
x2_train = x2_train.reshape(151, 80)
x2_test = x2_test.reshape(38, 80)


scaler = MinMaxScaler()
x1_train = scaler.fit_transform(x1_train)
x1_test = scaler.transform(x1_test)
x2_train = scaler.transform(x2_train)
x2_test = scaler.transform(x2_test)


x1_train = x1_train.reshape(151, 10,8)
x1_test = x1_test.reshape(38, 10,8)
x2_train = x2_train.reshape(151, 10,8)
x2_test = x2_test.reshape(38, 10,8)



#2. 모델구성
input1 = Input(shape=(10,8))
dense1 = Conv1D(30, 2, activation='relu', name='hyundae1')(input1)
dense2 = Conv1D(30, 2, activation='relu', name='hyundae2')(dense1)
dense3 = Flatten(name = 'flatten1')(dense2)
dense4 = Dense(1000, activation='relu', name='hyundae3')(dense3)
output1 = Dense(1000, activation='relu', name='output1')(dense4)

#2. 모델2
input2 = Input(shape=(10, 8))
dense11 = Conv1D(30, 2 , activation='relu', name='samsung1')(input2)
dense12 = Conv1D(30, 2 , activation='relu', name='samsung2')(dense11)
dense13 = Flatten(name = 'flatten2')(dense12)
dense14 = Dense(1000, activation='relu', name='samsung3')(dense13)
output2 = Dense(1000, activation='relu', name='output2')(dense14)

#2.머지
merge1 = concatenate([output1, output2], name='mg1')
merge2 = Dense(1000, activation='relu', name='mg2')(merge1)
last_output = Dense(1, activation='linear', name='last')(merge2)

model = Model(inputs=[input1, input2], outputs=last_output)

model.summary()

#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', mode='min', patience=200, restore_best_weights=True)
hist = model.fit([x1_train, x2_train], y_train, epochs=1000, batch_size=32, validation_split=0.2, callbacks=[es])

model.save(path_save + 'keras53_samsung5_khs.h5')

# 4. 평가, 예측
loss = model.evaluate([x1_test, x2_test], y_test)
print('loss : ', loss)

y_predict = model.predict([x1_test, x2_test])

r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

print('y_predict = ', np.round(y_predict[-1:],2))
# import matplotlib.pyplot as plt
# plt.plot(range(len(y_test)),y_test,label='data',c='orange')
# plt.plot(range(len(y_test)),y_predict,label='model')
# plt.legend()
# plt.show()
