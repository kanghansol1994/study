from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D,Flatten

model = Sequential()
model.add(Conv2D(7, (2,2), 
                 padding='same',
                 input_shape=(8,8,1)))  #출력 : (N, 7, 7, 7) / 
                                        #(batch_size, rows, columns, channels) 
model.add(Conv2D(filters=4,             #아웃풋이 4개
                 kernel_size=(3,3),     #자르는 사이즈
                 activation = 'relu'))  #출력 : (N, 5 , 5 , 4)
model.add(Conv2D(10, (2,2)))            #출력 : (N, 4, 4 ,10)
model.add(Flatten())                    #출력 : (N, 4*4*10) -> (N, 160)
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.summary()


# 30_1을 복붙
#padding=same 추가하니 (7,7,7) 패딩을 입어서 (8,8,7)로 바뀜
#padding=valid 추가하니 그대로임 디폴트값이 valid
#kernel_size가 늘어나면 패딩을 한겹씩 더 껴입음




