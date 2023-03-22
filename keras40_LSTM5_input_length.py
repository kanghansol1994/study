import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,SimpleRNN,LSTM


#2. 모델 구성
model = Sequential()                       #[batch, timesteps, feature]     #[배치 ,몇개씩 자를것인가, 특성,열]
model.add(LSTM(10, input_shape=(5,1)))     #[batch, input_length, input_dim]
# model.add(LSTM(10, input_length=5, input_dim=1))   #표현의 차이일뿐 성능에는 상관이 없음
# model.add(LSTM(10, input_dim=1, input_length=5, ))        #가독성이 약간 떨어짐
model.add(Dense(7, activation='relu'))
model.add(Dense(1, activation='linear'))

model.summary()
#4x((feature+1)xunits+units^2 = 480



