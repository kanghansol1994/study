import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x=np.array([1,2,3,4,5,6,7,8,9,10])
y=np.array([10,9,8,7,6,5,4,3,2,1,])

#[실습]넘파이 리스트의 슬라이싱 7:3으로 잘라라
#x_train = x[0:7]
x_train= x[:7] #파이썬에서 숫자는 0부터시작이라 0을빼고 적는게 깔끔하다
print(x_train) #[1,2,3,4,5,6,7]
# x_test= x[7:10] 
x_test = x[7:] #[8,9,10] #10을 쓰지않아도 끝이 10이기때문에 빼고 적는게 깔끔하다
y_train= y[:7] #[1,2,3,4,5,6,7]
y_test=y[7:] #[8,9,10]
#(7,)(3,)로 프린트 되는지 확인해보고 train과 test값도 각각 프린트하여 
#[1,2,3,4,5,6,7] 과[8,9,10]으로 프린트 되는지 확인해보기


print(x_train.shape, x_test.shape) #(7,)(3,)
print(y_train.shape, y_test.shape) #(7,)(3,)

#2. 모델구성
model= Sequential()
model.add(Dense(7, input_dim=1))
model.add(Dense(1))

#3. 컴파일,훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train , epochs=200, batch_size=4)

#4. 평가, 예측
loss= model.evaluate(x_test, y_test)
print('loss : ',loss) 
result=model.predict([11])
print('[11]의 예측값:', result)

#문제점:전체 데이터 범위 외부의 데이터를 예측 할때 오차가 큼
#전체 데이터를 섞은 후 랜덤하게 70%뽑는다(train) 나머지 30%(test)를 잡은 후
#전체 데이터 내부 값을 예측하게 되면 문제점 해결
