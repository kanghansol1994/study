#1. 데이터
import numpy as np
x1_dataset =  np.array([range(100), range(301, 401)])          #삼성, 아모레 주가
x2_dataset =  np.array([range(101, 201), range(411, 511), range(150, 250)])  #온도, 습도, 강수량
x3_dataset =  np.array([range(201, 301), range(511, 611), range(1300, 1400)])  
print(x1_dataset.shape)         #(2, 100)
print(x2_dataset.shape)         #(3, 100)
print(x3_dataset.shape)
x1 = np.transpose(x1_dataset) #행과 열 바꿔주는 2가지 방법
x2 = x2_dataset.T
x3 = x3_dataset.T
print(x1.shape)         #(100, 2)
print(x2.shape)         #(100, 3)
print(x3.shape)
y = np.array(range(2001, 2101))   #환율

from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y_train, y_test = train_test_split(x1, x2, x3, y, 
                                                                          train_size=0.7, random_state=333)
print(x1_train.shape, x1_test.shape) #(70,2) (30,2)
print(x2_train.shape, x2_test.shape) #(70,3) (30,3)
print(x3_train.shape, x3_test.shape) #(70,3) (30,3)
print(y_train.shape, y_test.shape) #(70,) (30,)



#2. 모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

#2-1 모델1
input1 = Input(shape=(2,))
dense1 = Dense(50, activation='relu', name='stock1')(input1)
dense2 = Dense(100, activation='relu', name='stock2')(dense1)
dense3 = Dense(150, activation='relu', name='stock3')(dense2)
output1 = Dense(10, activation='relu', name='output1')(dense3)

#2-2. 모델2
input2 = Input(shape=(3,))
dense11 = Dense(10, name='weather1')(input2)
dense12 = Dense(20, name='weather2')(dense11)
dense13 = Dense(40, name='weather3')(dense12)
dense14 = Dense(80, name='weather4')(dense13)
output2 = Dense(10, name='output2')(dense14)

#2-3. 모델3
input3 = Input(shape=(3,))
dense111 = Dense(10, name='ws1')(input3)
dense112 = Dense(20, name='ws2')(dense111)
dense113 = Dense(40, name='ws3')(dense112)
dense114 = Dense(80, name='ws4')(dense113)
output3 = Dense(10, name='output3')(dense114)

from tensorflow.keras.layers import concatenate, Concatenate
merge1 = concatenate([output1, output2,output3], name='mg1')
merge2 = Dense(10, activation='relu', name='mg2')(merge1)
merge3 = Dense(30, activation='relu', name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)

model = Model(inputs=[input1, input2, input3], outputs=last_output)

model.summary()

#앙상블 모델 사용시 합치기 전 모델 아웃풋을 꼭 1을 줄 필요는 없다
#1을 줄 경우 이미 그 모델에서 축소된 값이 나와 합쳐지므로 성능이 저하 될 수 있기 때문

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping

es = EarlyStopping(monitor='val_loss', patience=50, mode='min',
              verbose=1,
              restore_best_weights=True,) #디폴트는 false) 
model.fit([x1_train,x2_train,x3_train],y_train, epochs=300, batch_size=8, verbose=1)
#4. 평가, 예측
loss = model.evaluate([x1_test,x2_test,x3_test],y_test)
print('loss = ', loss)

from sklearn.metrics import r2_score,mean_squared_error

y_predict = model.predict([x1_test,x2_test,x3_test])
r2 = r2_score(y_test, y_predict)
print('r2스코어 = ', r2)
def RMSE(y_test, y_predict): 
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

################################################
# loss =  0.013805965892970562
# r2스코어 =  0.9999766072823623
# RMSE :  0.11749879182032076