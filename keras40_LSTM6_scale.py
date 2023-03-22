import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,SimpleRNN,LSTM

#1. 데이터
datasets= np.array([1,2,3,4,5,6,7,8,9,10,11,12,20,30,40,50,60,70])
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
             [5,6,7],[6,7,8],[7,8,9],[8,9,10],
             [9,10,11],[10,11,12],
             [20,30,40],[30,40,50],[40,50,60]])
             
y= np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
# print(x.shape, y.shape) (13,3) (13,)
x = x.reshape(13,3,1)
# print(x.shape) (13,3,1)

 #아워너 80


#만들기
#2. 모델 구성
model = Sequential()

model = Sequential()
model.add(LSTM(512, activation='relu', input_shape=(3,1)))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=300)
# model.save_weights('./_save/keras26_8_save_weights2.h5')
model.load_weights('./_save/keras26_8_save_weights2.h5')

#4. 평가, 예측
loss = model.evaluate(x, y)
x_predict = np.array([50,60,70]).reshape(1, 3, 1) #[[[50],[60],[70]]]
print(x_predict.shape) #(1, 3, 1)

result = model.predict(x_predict)
print('loss : ', loss)
print('[50,60,70]의 결과', result)

# [50,60,70]의 결과 [[80.03961]]