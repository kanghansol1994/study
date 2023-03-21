import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,SimpleRNN,LSTM


#2. 모델 구성
model = Sequential()                       #[batch, timesteps, feature]     #[배치 ,몇번 훈련했는지, 특성,열]
model.add(LSTM(10, activation='relu', input_shape=(5,1)))
model.add(Dense(7, activation='relu'))
model.add(Dense(1, activation='linear'))

model.summary()
#4x((feature+1)xunits+units^2 = 480
#rnn은 2개 연결이 안됨 rnn은 3차원을 받아서 2차원을 출력하기때문



