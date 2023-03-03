from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test= train_test_split(x, y, train_size=0.8, 
                                                   random_state=928)
# print(x.shape, y.shape) #(20640, 8) (20640,)
#[실습]
#1. R2 0.55~0.6 이상


#2. 모델구성
model= Sequential()
model.add(Dense(100, input_dim=8,))
model.add(Dense(1))
model.add(Dense(1))
model.add(Dense(1))
model.add(Dense(1))
model.add(Dense(1))
model.add(Dense(1))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=128,)


#4. 평가, 예측
loss=model.evaluate(x_test,y_test)
print('loss= ', loss)

y_predict= model.predict(x_test)
r2= r2_score(y_test, y_predict)
print('r2스코어: ', r2)