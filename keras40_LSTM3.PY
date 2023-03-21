import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,SimpleRNN,LSTM

#1. 데이터
datasets= np.array([1,2,3,4,5,6,7,8,9,10])
#y = ? 
x = np.array([[1,2,3,4,5],[2,3,4,5,6],[3,4,5,6,7],[4,5,6,7,8], [5,6,7,8,9]])
y = np.array([6,7,8,9,10])

print(x.shape,y.shape) #(5, 5) (5,)
# x의 shape = (행,열, 몇개씩 훈련하는지)
x = x.reshape(5,5,1) #[[[1],[2],[3]],[[2],[3],[4],]]
print(x.shape) #(5, 5, 1)

#2. 모델 구성
model = Sequential()

model.add(LSTM(512, activation='relu', input_shape=(5,1)))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='linear'))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
import time
start = time.time()
model.fit(x, y, epochs=600)
end = time.time()
#4. 평가 예측
loss = model.evaluate(x, y)
x_predict = np.array([6,7,8,9,10]).reshape(1, 5, 1) #[[[7],[8], [9], [10]]]
print(x_predict.shape) #(1, 3, 1)

result = model.predict(x_predict)
print('loss : ', loss)
print('[6,7,8,9,10]의 결과', result)
print('걸린시간 : ', round(end-start, 2))

# loss :  0.00013851493713445961
# [6,7,8,9,10]의 결과 [[11.00349]]
