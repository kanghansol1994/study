#save_model과 비교

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

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8,random_state=333,
)
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)#위의 두줄을 이 한줄로 줄일 수 있음
x_test = scaler.transform(x_test)




#2. 모델구성
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
                   verbose=1, 
                #    restore_best_weights=True
                ) 
mcp = ModelCheckpoint(monitor='val_loss', mode='auto',
        verbose=1,
        save_best_only=True,
        filepath='./_save/MCP/keras27_3_MCP.hdf5'
                      ) #shift+tap = 왼쪽으로 옮겨짐 tab=오른쪽으로 옮겨짐
#가장 좋은 지점 하나만 저장됨

model.fit(x_train, y_train, epochs=10000, 
          callbacks=[es, mcp],
          validation_split=0.2)

model.save('./_save/MCP/keras27_3_save_model.h5')

#4. 평가, 예측
print("==================1. 기본 출력======================")
loss = model.evaluate(x_test,y_test, verbose=0)
print('loss : ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

print("=================2. load_model 출력==================")
model2 = load_model('./_save/MCP/keras27_3_save_model.h5')

loss = model2.evaluate(x_test,y_test, verbose=0 )
print('loss : ', loss)

y_predict = model2.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

print("=================3. MCP 출력==================")
model3 = load_model('./_save/MCP/keras27_3_MCP.hdf5')

loss = model3.evaluate(x_test,y_test, verbose=0)
print('loss : ', loss)

y_predict = model3.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)


#MCP와 restore_best_weights 같이 사용하여도 한가지만 사용하여도 상관없음
#밑의 결과와 같이 MCP는 제일 좋은 가중치를 저장 하는것이기 때문에 
#restore_best_weights 쓰지 않아도 제일 좋은 가중치로 모델이 돌아감
#restore_best_weights 쓰지 않을 경우 patience 만큼 밀린 가중치가 저장 되기 때문에
#값이 서로 다름

# ==================1. 기본 출력======================
# loss :  20.905786514282227
# r2스코어 :  0.7868474221296927
# =================2. load_model 출력==================
# loss :  20.905786514282227
# r2스코어 :  0.7868474221296927
# =================3. MCP 출력==================
# loss :  21.412107467651367
# r2스코어 :  0.7816850059355931
