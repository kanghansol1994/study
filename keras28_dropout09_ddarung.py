from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Dropout,LeakyReLU
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 


#1. 데이터
path = './_data/ddarung/'      
path_save = './_save/ddarung/'
train_csv = pd.read_csv(path+'train.csv', index_col=0)  


print(train_csv)
print(train_csv.shape)  #(1459,10)  


test_csv = pd.read_csv(path+'test.csv', index_col=0)

print(test_csv)
print(test_csv.shape)   #(715,9)

print(train_csv.columns)
print(train_csv.describe())

print(type(train_csv))    
print(train_csv.isnull().sum())

train_csv = train_csv.dropna()  
print(train_csv.isnull().sum())    

print(train_csv.info())
print(train_csv.shape)    

x = train_csv.drop(['count'], axis=1)
y = train_csv['count']

print(x)
print(y)



x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=362, test_size=0.2
)

scaler = MinMaxScaler()
scaler.fit(x_train) #x를 바꿀 준비하라
x_train = scaler.transform(x_train) #x를 바꿔라
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

#2. 모델 구성
model=Sequential()
model.add(Dense(100,input_dim=x.shape[1],activation=LeakyReLU(0.5)))
model.add(Dropout(0.2))
model.add(Dense(75,activation='relu'))
model.add(Dense(50,activation=LeakyReLU(0.5)))
model.add(Dense(25,activation=LeakyReLU(0.5)))
model.add(Dense(10,activation='relu'))
model.add(Dense(1,activation="linear"))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam',)

import datetime
date =datetime.datetime.now()
date = date.strftime('%m%d_%H%M')
filepath = './_save/MCP/ddarung/'
filename = '{val_loss:.4f}.hdf5'

from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=50, mode='min',
              verbose=1,
              restore_best_weights=True, #디폴트는 false
              ) 
mcp = ModelCheckpoint(monitor='val_loss', mode='auto',
                      verbose=1,
                      save_best_only=True,
                      filepath= "".join([filepath, 'ddarung_', date, '_', filename])
)


hist= model.fit(x_train, y_train, epochs=2000, batch_size=40,
                validation_split=0.2,
                verbose=1,
                callbacks=[es,]#mcp],
                )

#4. 평가, 예측
loss=model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2스코어: ', r2)

import numpy as np
def RMSE(y_test, y_predict): 
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test,y_predict)
print("RMSE : ", rmse)

y_submit=model.predict(test_csv)

submission = pd.read_csv(path+'submission.csv', index_col=0)
submission['count'] = y_submit
submission.to_csv(path_save+ 'submit_0316_0501.csv')

#문제점: 최적의 가중치가 저장 되는것이 아니라 마지막loss값이 가중치로 저장됨



#seed:286
# model=Sequential()
# model.add(Dense(32,input_dim=x.shape[1],activation=LeakyReLU(0.5)))
# model.add(Dropout(0.2))
# model.add(Dense(16,activation=LeakyReLU(0.5)))
# model.add(Dense(32,activation=LeakyReLU(0.5)))
# model.add(Dense(16,activation=LeakyReLU(0.5)))
# model.add(Dense(32,activation=LeakyReLU(0.5)))
# model.add(Dense(1,activation="linear"))
# patience=200
# x_train, y_train, epochs=2000, batch_size=40,
#                 validation_split=0.2,
#                 verbose=1,
#                 callbacks=[es,]#mcp],
# loss :  1842.41064453125
# r2스코어:  0.7439043255031311
# RMSE :  42.923309484243624
# 데이콘:61.50

# seed:287
# loss :  1599.135498046875
# r2스코어:  0.7946443292172138
# RMSE :  39.98919055489898
# 데이콘:63.4

# seed:290
# loss :  1554.4990234375
# r2스코어:  0.7778184708450062
# RMSE :  39.42713588836648
# 데이콘:64.38

# seed:291
# loss :  1819.462890625
# r2스코어:  0.7368562310805249
# RMSE :  42.655162756146254
# 데이콘:59.36

# seed:293
# loss :  1458.877685546875
# r2스코어:  0.7777958949541307
# RMSE :  38.19525896779339
# 데이콘:64.22

# seed: 336
# loss :  1402.7431640625
# r2스코어:  0.7975420103355931
# RMSE :  37.45321394604834
# 데이콘:61.92

# seed: 337
# loss :  1660.7752685546875
# r2스코어:  0.7717528093362228
# RMSE :  40.752609831632654
# 데이콘:65.232200007	

# seed:291
# model=Sequential()
# model.add(Dense(100,input_dim=x.shape[1],activation=LeakyReLU(0.5)))
# model.add(Dropout(0.2))
# model.add(Dense(75,activation='relu'))
# model.add(Dense(50,activation=LeakyReLU(0.5)))
# model.add(Dense(25,activation=LeakyReLU(0.5)))
# model.add(Dense(10,activation='relu'))
# model.add(Dense(1,activation="linear"))
# patience=50
# loss :  1637.9385986328125
# r2스코어:  0.7631095833998064
# RMSE :  40.471453678167265
# 데이콘 :59.0524

# model=Sequential()
# model.add(Dense(100,input_dim=x.shape[1],activation=LeakyReLU(0.5)))
# model.add(Dropout(0.1))
# model.add(Dense(75,activation='relu'))
# model.add(Dense(50,activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(25,activation=LeakyReLU(0.5)))
# model.add(Dense(10,activation='relu'))
# model.add(Dense(1,activation="linear"))
# loss :  1520.956787109375
# r2스코어:  0.7800283234934551
# RMSE :  38.9994462417146
# 데이콘: 62.29

#seed :360
# loss :  1316.0076904296875
# r2스코어:  0.8134450139098983
# RMSE :  36.27681782905521
# 데이콘 : 62.87






