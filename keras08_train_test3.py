import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split 
#1. 데이터
x=np.array([1,2,3,4,5,6,7,8,9,10])
y=np.array([1,2,3,4,5,6,7,8,9,10])

x_train, x_test, y_train, y_test = train_test_split(
    x,y, 
    #train_size=0.7
    test_size=0.3,
    random_state=1234, #디폴트값이 랜덤
    shuffle=False,) #shuffle은 디폴트가 True
#1~10중 랜덤으로 값을 뽑아내지만 
#데이터가 계속 변경된다면 잘 만들어진건지 판단이 어려움
#이것을 잡아주는게 random_state 또는 랜덤시드라고 부른다
print(x_train)
print(x_test)

#[검색] train과 test를 섞어서 7:3으로 찾을 수 있는 방법! (4번,9번라인)
#힌트 사이킷런

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense(1))


#3. 컴파일,훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train , epochs=200, batch_size=4)

#4. 평가, 예측
loss= model.evaluate(x_test, y_test)
print('loss : ',loss) 
result=model.predict([11])
print('[11]의 예측값:', result)


