from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten,MaxPooling2D
import numpy as np
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
#1. 데이터
(x_train,y_train), (x_test, y_test) = mnist.load_data()

scaler = MinMaxScaler()
x_train = x_train.reshape(-1,1)
x_train = scaler.fit_transform(x_train)
x_test = x_test.reshape(-1,1)
x_test = scaler.transform(x_test)
##########실습########
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1) #데이터의 구조만 바뀌는것 순서와,값이 바뀌는게 아님
print(np.unique(y_train, return_counts=True)) #[0,1,2,3,4,5,6,7,8,9]

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train)
print(y_train.shape)
#2. 모델구성
model = Sequential()
model.add(Conv2D(64, (2,2), padding='same',
                 input_shape=(28,28,1)))  
model.add(MaxPooling2D())                        #칸이 중첩되지 않게 설계                  
model.add(Conv2D(filters=64,             
                 kernel_size=(2,2), padding='valid',    
                 activation = 'relu'))  
model.add(Conv2D(32,2))             
model.add(Flatten())                    
model.add(Dense(10, activation='softmax')) 

model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_acc', patience=10, mode='max',
                    verbose=1,
                    restore_best_weights=True,)

model.fit(x_train, y_train, epochs=30, batch_size=128, verbose=1,
                 validation_split=0.1, callbacks=[es])

#4. 평가, 예측
result = model.evaluate(x_test,y_test)
print('loss : ', result[0])
print('acc : ', result[1])

# y_predict = model.predict(x_test)
# acc = accuracy_score(np.argmax(y_test,axis=1), np.argmax(y_predict,axis=1))
