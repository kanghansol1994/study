from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense,Input
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
    x, y, shuffle=True, random_state=44, test_size=0.1
)

scaler = MinMaxScaler()
scaler.fit(x_train) #x를 바꿀 준비하라
x_train = scaler.transform(x_train) #x를 바꿔라
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

#2. 모델 구성
# model = Sequential()
# model.add(Dense(40, input_dim=9))
# model.add(Dense(20))
# model.add(Dense(15))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(1))

input1 = Input(shape=(9,))
dense1 = Dense(40, activation='relu')(input1)
dense2 = Dense(20, activation='relu')(dense1)
dense3 = Dense(15, activation='relu')(dense2)
dense4 = Dense(10, activation='relu')(dense3)
dense5 = Dense(10, activation='relu')(dense4)
dense6 = Dense(10, activation='relu')(dense5)
dense7 = Dense(10, activation='relu')(dense6)
dense8 = Dense(10, activation='relu')(dense7)
output1 = Dense(1, activation='linear')(dense8)
model = Model(inputs = input1, outputs = output1)


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=100, mode='min',
              verbose=1,
              restore_best_weights=True, #디폴트는 false
              ) 

hist= model.fit(x_train, y_train, epochs=300, batch_size=40,
                validation_split=0.2,
                verbose=1,
                callbacks=[es],
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
submission.to_csv(path_save+ 'submit_0313_0504.csv')

#문제점: 최적의 가중치가 저장 되는것이 아니라 마지막loss값이 가중치로 저장됨

loss :  2825.89794921875
r2스코어 : 0.6170048667573036
RMSE :  53.159174801965484

loss :  0.00951689388602972
r2스코어:  0.7075440114195302
RMSE :  0.09755457045344555