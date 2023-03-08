from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
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
    x, y, shuffle=True, random_state=2536, test_size=0.2
)

#2. 모델 구성
model=Sequential()
model.add(Dense(20, activation='relu', input_dim=9))
model.add(Dense(40, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='linear'))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=500, mode='min',
              verbose=1,
              restore_best_weights=True, #디폴트는 false
              ) 

hist= model.fit(x_train, y_train, epochs=2000, batch_size=16,
                validation_split=0.2,
                verbose=1,
                callbacks=[es],
                )
'''
print("***********************************")
print(hist) 
#<tensorflow.python.keras.callbacks.History object at 0x000001C5335864C0>
print("***********************************")
print(hist.history)
print("***********************************")
print(hist.history['loss'])
print("*************발로스**********************")
print(hist.history['val_loss'])
print("*************발로스**********************")
'''
#4. 평가, 예측
loss=model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2스코어: ', r2)

#r2스코어:0.7188
y_submit=model.predict(test_csv)

submission = pd.read_csv(path+'submission.csv', index_col=0)
submission['count'] = y_submit
submission.to_csv(path_save+ 'submit_0308_0510.csv')
'''
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(9, 6))
plt.plot(hist.history['loss'], marker='.', c='red', label='로스')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='발_로스')
plt.title('따릉이')
plt.xlabel('epochs')
plt.ylabel('loss, val_loss')
plt.legend()
plt.grid()
plt.show()
'''
#문제점: 최적의 가중치가 저장 되는것이 아니라 마지막loss값이 가중치로 저장됨