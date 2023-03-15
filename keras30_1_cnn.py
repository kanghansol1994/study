from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D,Flatten

model = Sequential()
model.add(Conv2D(7, (2,2), 
                 input_shape=(8,8,1)))  #출력 : (N, 7, 7, 7) / 
                                        #(batch_size, rows, columns, channels) 
model.add(Conv2D(filters=4,             #아웃풋이 4개
                 kernel_size=(3,3),     # 자르는 사이즈
                 activation = 'relu'))  #출력 : (N, 5 , 5 , 4)
model.add(Conv2D(10, (2,2)))            #출력 : (N, 4, 4 ,10)
model.add(Flatten())                    #출력 : (N, 4*4*10) -> (N, 160)
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.summary()

# Conv2D 레이어는 자신의 입력과 출력 채널 수, 커널 크기에 따라 필요한 파라미터 수가 결정
#첫번째 레이어의 경우 2x2의 크기를 가진 필터가 7개 입력데이터는 1개이므로 (2x2x1)x7 +바이어스 출력갯수7개 =35개
#두번째 레이어의 경우 3x3의 크기를 가진 필터가 4개 입력데이터는 7개 (3x3x7)x4 + 바이어스 4개 =256개
#세번째 레이어의 경우 2x2의 크기를 가진 필터가 10개 입력데이터는 4개 (2x2x4)x10+ 바이어스 10개 =170개
#가로위치값x세로위치값x인풋x아웃풋+바이어스출력갯수

#2로 자르면 1개 줄어듬 3로 자르면 2개 줄어듬 4로 자르면 3개 줄어듬

#다차원데이터를 일렬로 피는 작업을 먼저 한다.(Flatten)
#Dense 사용 전에 flatten 사용 해줘야 함
#ex) (N,4,4,10) 의 경우 (N,160)이 된다



