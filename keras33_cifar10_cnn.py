from tensorflow.keras.datasets import cifar10
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten,MaxPooling2D
import numpy as np
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
#1. 데이터
(x_train,y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape)    #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)      #(10000, 32, 32, 3) (10000, 1)

scaler = MinMaxScaler() #이미지에서는 이렇게 사용가능
x_train = x_train/255.
x_test = x_test/255.
print(np.max(x_train), np.min(x_train)) #1.0 0.0

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train)
print(y_train.shape) #(50000,10)

#2. 모델
model = Sequential()
model.add(Conv2D(64,3, padding='same', activation='relu', input_shape=(32, 32, 3))),
model.add(MaxPooling2D())
model.add(Conv2D(128,3, padding='same', activation='relu')),
model.add(MaxPooling2D())
model.add(Conv2D(128,3 , padding='same', activation='relu')),
model.add(Flatten())
model.add(Dense(64, activation='relu')),
model.add(Dense(10, activation='softmax'))

model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_acc', patience=10, mode='max',
                    verbose=1,
                    restore_best_weights=True,)

model.fit(x_train, y_train, epochs=100, batch_size=128, verbose=1,
                 validation_split=0.1, callbacks=[es])
model.save('./_save/MCP/cifar10/_save_model.h1/')

#4. 평가, 예측
result = model.evaluate(x_test,y_test)
print('loss : ', result[0])
print('acc : ', result[1])

#######################################################################
# model = Sequential()
# model.add(Conv2D(32,4, activation='relu', input_shape=(32, 32, 3))),
# model.add(MaxPooling2D())
# model.add(Conv2D(64,4, activation='relu')),
# model.add(Conv2D(64,4 , activation='relu')),
# model.add(Flatten())
# model.add(Dense(64, activation='relu')),
# model.add(Dense(10, activation='softmax
# loss :  [0.9224437475204468, 0.6814000010490417]
# acc :  0.6814

# model = Sequential()
# model.add(Conv2D(64,4, activation='relu', input_shape=(32, 32, 3))),
# model.add(MaxPooling2D())
# model.add(Conv2D(128,4, activation='relu')),
# model.add(Conv2D(64,4 , activation='relu')),
# model.add(Flatten())
# model.add(Dense(64, activation='relu')),
# model.add(Dense(10, activation='softmax'))
#add : acc :  0.6949

# model = Sequential()
# model.add(Conv2D(64,5, padding='same', activation='relu', input_shape=(32, 32, 3))),
# model.add(MaxPooling2D())
# model.add(Conv2D(128,5, padding='same', activation='relu')),
# model.add(MaxPooling2D())
# model.add(Conv2D(128,5 , padding='same', activation='relu')),
# model.add(Flatten())
# model.add(Dense(64, activation='relu')),
# model.add(Dense(10, activation='softmax'))
# batch_size= 128
# acc : 0.7263

# model = Sequential()
# model.add(Conv2D(64,3, padding='same', activation='relu', input_shape=(32, 32, 3))),
# model.add(MaxPooling2D())
# model.add(Conv2D(128,3, padding='same', activation='relu')),
# model.add(MaxPooling2D())
# model.add(Conv2D(128,3 , padding='same', activation='relu')),
# model.add(Flatten())
# model.add(Dense(64, activation='relu')),
# model.add(Dense(10, activation='softmax'))
# acc : 0.7410