from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D,Flatten

model = Sequential()                    #(N, 3)
model.add(Dense(10, input_shape=(3,)))  #(batch_size, input_dim) 
model.add(Dense(units=15))              #출력 (batch_size, units)
model.summary()

#Dense에서 10은(units)  3은 입력데이터의 크기가 (3,)인 벡터
#Conv2D에서 10은(filter) 갯수 3은 이미지의 크기가 3x3이고 채널수가 1 인것을 의미