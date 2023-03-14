from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,LeakyReLU,Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 

#1. 데이터
path = './_data/kaggle/' 
     
train_csv = pd.read_csv(path+'train.csv', index_col=0)  
test_csv = pd.read_csv(path+'test.csv', index_col=0)
train_csv = train_csv.dropna()

x = train_csv.drop(['count', 'casual', 'registered'], axis=1)
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=340, train_size=0.8
)

scaler = MinMaxScaler()
scaler.fit(x_train) #x를 바꿀 준비하라
x_train = scaler.transform(x_train) #x를 바꿔라
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)


#2. 모델 구성
model = Sequential()
model=Sequential()
model.add(Dense(128,input_dim=x.shape[1],activation=LeakyReLU(0.4)))
model.add(Dropout(0.1))
model.add(Dense(96,activation=LeakyReLU(0.4)))
model.add(Dense(72,activation=LeakyReLU(0.4)))
model.add(Dense(64,activation=LeakyReLU(0.4)))
model.add(Dense(32,activation=LeakyReLU(0.4)))
model.add(Dense(16,activation=LeakyReLU(0.4)))
model.add(Dense(1,activation="linear"))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

import datetime
date =datetime.datetime.now()
date = date.strftime('%m%d_%H%M')
filepath = './_save/MCP/kaggle/'
filename = '{val_loss:.4f}.hdf5'

from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=25, mode='min',
              verbose=1,
              restore_best_weights=True, #디폴트는 false
              ) 

mcp = ModelCheckpoint(monitor='val_loss', mode='auto',
                      verbose=1,
                      save_best_only=True,
                      filepath= "".join([filepath, 'kkagle_', date, '_', filename])
)


hist= model.fit(x_train, y_train, epochs=2000, batch_size=300,
                validation_split=0.2,
                verbose=1,
                callbacks=[es],#mcp
                )

#4. 평가, 예측
loss=model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2스코어: ', r2)

def RMSE(y_test, y_predict): 
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test,y_predict)
print("RMSE : ", rmse)

y_submit=model.predict(test_csv)

submission = pd.read_csv(path+'sampleSubmission.csv', index_col=0)
submission['count'] = y_submit

path2 = './_save/kaggle/' 
submission.to_csv(path2 + 'submit_0314_004.csv')

#seed:287
# model = Sequential()
# model=Sequential()
# model.add(Dense(32,input_dim=x.shape[1],activation=LeakyReLU(0.5)))
# model.add(Dropout(0.2))
# model.add(Dense(16,activation=LeakyReLU(0.5)))
# model.add(Dense(32,activation=LeakyReLU(0.5)))
# model.add(Dense(16,activation=LeakyReLU(0.5)))
# model.add(Dense(32,activation=LeakyReLU(0.5)))
# model.add(Dense(1,activation="linear"))
# epochs = 2000 , batch_size=300 patience=50
# loss :  21619.580078125
# r2스코어:  0.3064182409799393
# RMSE :  147.03597329488352
# 캐글:1.32306