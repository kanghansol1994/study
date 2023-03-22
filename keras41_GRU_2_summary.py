import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,SimpleRNN,LSTM,GRU


#2. 모델 구성
model = Sequential()                       #[batch, timesteps, feature]     #[배치 ,몇개씩 자를것인가, 특성,열]
model.add(GRU(10, activation='relu', input_shape=(5,1)))
model.add(Dense(7, activation='relu'))
model.add(Dense(1, activation='linear'))

model.summary()



# forget, input gate는 update gate로 통합, output gate는 없어지고, reset gate로 대체

