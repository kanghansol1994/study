import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,SimpleRNN

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
model = Sequential()                       #[batch, timesteps, feature]     #[배치 ,몇번 훈련했는지, 특성,열]
model.add(SimpleRNN(10, activation='relu', input_shape=(5,1)))
#units x(feature+ bias+units) = params
model.add(Dense(7, activation='relu'))
model.add(Dense(1, activation='linear'))

model.summary()

#rnn은 2개 연결이 안됨 rnn은 3차원을 받아서 2차원을 출력하기때문



