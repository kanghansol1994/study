from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test= train_test_split(x, y, train_size=0.8, 
                                                   random_state=1016)

# print(x.shape, y.shape) #(442, 10) (442,)

#[실습]
#R2 0.62 이상


#2. 모델구성
model=Sequential()
model.add(Dense(128, input_dim=10))
model.add(Dropout(0.2))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=64)

#4. 평가, 예측
loss=model.evaluate(x_test,y_test)
print('loss= ', loss)

y_predict= model.predict(x_test)
r2= r2_score(y_test, y_predict)
print('r2스코어: ', r2)