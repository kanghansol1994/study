from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd

#1. 데이터
path = './_data/kaggle/' 
     
train_csv = pd.read_csv(path+'train.csv', index_col=0)  
test_csv = pd.read_csv(path+'test.csv', index_col=0)
train_csv = train_csv.dropna()

x = train_csv.drop(['count', 'casual', 'registered'], axis=1)
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=2536, train_size=0.7
)

#2. 모델 구성
model=Sequential()
model.add(Dense(12, activation='relu', input_dim=8))
model.add(Dense(24, activation='relu'))
model.add(Dense(48, activation='relu'))
model.add(Dense(96, activation='relu'))
model.add(Dense(48, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(1, activation='linear'))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=200, mode='min',
              verbose=1,
              restore_best_weights=True, #디폴트는 false
              ) 

hist= model.fit(x_train, y_train, epochs=2000, batch_size=200,
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
#r2스코어: 0.3232
y_submit=model.predict(test_csv)

submission = pd.read_csv(path+'sampleSubmission.csv', index_col=0)
submission['count'] = y_submit

path2 = './_save/kaggle/' 
submission.to_csv(path2 + 'submit_0308_010.csv')

'''
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(9, 6))
plt.plot(hist.history['loss'], marker='.', c='red', label='로스')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='발_로스')
plt.title('캐글')
plt.xlabel('epochs')
plt.ylabel('loss, val_loss')
plt.legend()
plt.grid()
plt.show()
'''
#문제점: 최적의 가중치가 저장 되는것이 아니라 마지막loss값이 가중치로 저장됨