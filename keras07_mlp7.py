#x는 1개
#y는 3개

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1.데이터
x=np.array([range(10)])
print(x.shape) #(1,10)  #[실습] (1,10)을 (10,1)로 바꿔 보자
x = x.T #(10,1)

y=np.array([[1,2,3,4,5,6,7,8,9,10],
            [1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9],
            [9,8,7,6,5,4,3,2,1,0]]) #(3.10)

y=y.T #(10,3)

#2.모델구성
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(3))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=30, batch_size=3)

#4. 평가, 예측
loss=model.evaluate(x,y)
print('loss= ', loss)
result=model.predict([[9]])
print('[9]의 예측값', result)