from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense,Input
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 
from sklearn.metrics import r2_score

#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets['target']

print(type(x)) # <class 'numpy.ndarray'>
print(x)

print(np.min(x), np.max(x)) # 0.0 711.0
scaler = MinMaxScaler()
scaler.fit(x) #x를 바꿀 준비하라
x = scaler.transform(x) #x를 바꿔라
print(np.min(x), np.max(x)) # 0.0 1.0

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,random_state=333,
)
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)


#2. 모델구성
# model = Sequential()
# model.add(Dense(5, activation='relu', input_dim=30))
# model.add(Dense(5, activation='relu'))
# model.add(Dense(5, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(1, activation='linear'))

input1 = Input(shape=(30,))
dense1 = Dense(5, activation='relu')(input1)
dense2 = Dense(5, activation='relu')(dense1)
dense3 = Dense(5, activation='relu')(dense2)
dense4 = Dense(10, activation='relu')(dense3)
output1 = Dense(1, activation='linear')(dense4)
model = Model(inputs = input1, outputs = output1)





#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=200, batch_size=12,
          validation_split=0.2,
          verbose=1,)


#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)