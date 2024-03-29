import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,SimpleRNN,LSTM,GRU

#1. 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
             [5,6,7],[6,7,8],[7,8,9],[8,9,10],
             [9,10,11],[10,11,12],
             [20,30,40],[30,40,50],[40,50,60]])
             
y= np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
# print(x.shape, y.shape) (13,3) (13,)
x = x.reshape(13,3,1)
# print(x.shape) (13,3,1)

 #아워너 80
print(x.shape,y.shape) #(13, 3) (13,)
# x의 shape = (행,열, 몇개씩 훈련하는지)
x = x.reshape(13,3,1) #[[[1],[2],[3]],[[2],[3],[4],]]
print(x.shape) #(13, 3, 1)

#2. 모델 구성
model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(3,1), return_sequences=True))
model.add(LSTM(11, return_sequences=True))
model.add(GRU(12))
model.add(Dense(1, activation='linear'))

model.summary()
'''
# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=600)

#4. 평가 예측
loss = model.evaluate(x, y)
x_predict = np.array([8,9,10]).reshape(1, 3, 1) #[[[8], [9], [10]]]
print(x_predict.shape) #(1, 3, 1)

result = model.predict(x_predict)
print('loss : ', loss)
print('[8,9,10]의 결과', result)
'''