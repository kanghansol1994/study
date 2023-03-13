from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense,Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
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
    x, y, shuffle=True, random_state=2536, train_size=0.7
)

scaler = MinMaxScaler()
scaler.fit(x_train) #x를 바꿀 준비하라
x_train = scaler.transform(x_train) #x를 바꿔라
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

#2. 모델 구성
# model = Sequential()
# model.add(Dense(12, input_dim=8))
# model.add(Dense(24, activation='relu'))
# model.add(Dense(48, activation='relu'))
# model.add(Dense(62, activation='relu'))
# model.add(Dense(86, activation='relu'))
# model.add(Dense(62, activation='relu'))
# model.add(Dense(48, activation='relu'))
# model.add(Dense(24, activation='relu'))
# model.add(Dense(12, activation='relu'))
# model.add(Dense(1, activation='linear'))

input1 = Input(shape=(8,))
dense1 = Dense(12, activation='relu')(input1)
dense2 = Dense(24, activation='relu')(dense1)
dense3 = Dense(48, activation='relu')(dense2)
dense4 = Dense(62, activation='relu')(dense3)
dense5 = Dense(86, activation='relu')(dense4)
dense6 = Dense(62, activation='relu')(dense5)
dense7 = Dense(48, activation='relu')(dense6)
dense8 = Dense(24, activation='relu')(dense7)
dense9 = Dense(12, activation='relu')(dense8)
output1 = Dense(1, activation='linear')(dense9)
model = Model(inputs = input1, outputs = output1)


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=200, mode='min',
              verbose=1,
              restore_best_weights=True, #디폴트는 false
              ) 

hist= model.fit(x_train, y_train, epochs=2000, batch_size=600,
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

y_submit=model.predict(test_csv)

submission = pd.read_csv(path+'sampleSubmission.csv', index_col=0)
submission['count'] = y_submit

path2 = './_save/kaggle/' 
submission.to_csv(path2 + 'submit_0313_002.csv')


#문제점: 최적의 가중치가 저장 되는것이 아니라 마지막loss값이 가중치로 저장됨