import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,SimpleRNN,LSTM,GRU
from tensorflow.keras.layers import Bidirectional

#1. 데이터
datasets= np.array([1,2,3,4,5,6,7,8,9,10])

x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6], [5,6,7],[6,7,8],[7,8,9]])
y = np.array([4,5,6,7,8,9,10])
x = x.reshape(7,3,1) #[[[1],[2],[3]],[[2],[3],[4],]]
print(x.shape) #(7, 3, 1)

#2. 모델 구성
model = Sequential()
# model.add(Bidirectional(SimpleRNN(10), input_shape=(3, 1)))
model.add(Bidirectional(LSTM(10, return_sequences=True), input_shape=(3, 1)))
model.add(LSTM(10, return_sequences=True))
model.add(Bidirectional(GRU(10)))

model.add(Dense(1))

model.summary()



