#보스턴 집값 맞추기용
from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target


x_train, x_test, y_train, y_test= train_test_split(x, y, train_size=0.9, 
                                                   random_state=1016)
# print(x)
# print(y)

# print(datasets)
# print(datasets.feature_names)
#['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
# 'B' 'LSTAT']

# print(datasets.DESCR)
# print(x.shape, y.shape) #(506,13) (506,)

#[실습]
#2. R2 0.8이상

#2. 모델구성
model=Sequential()
model.add(Dense(128, input_dim=13))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=32)

#4. 평가, 예측
loss=model.evaluate(x_test,y_test)
print('loss= ', loss)

y_predict= model.predict(x_test)
r2= r2_score(y_test, y_predict)
print('r2스코어: ', r2)

        