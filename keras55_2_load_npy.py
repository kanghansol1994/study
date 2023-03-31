import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import MaxPooling2D,LeakyReLU


#1. 데이터
path = 'd:/study_data/_save/_npy/'
# np.save(path + 'keras55_1_x_train.npy', arr = xy_train[0][0])
# np.save(path + 'keras55_1_x_test.npy', arr = xy_test[0][0])
# np.save(path + 'keras55_1_y_train.npy', arr = xy_train[0][1])
# np.save(path + 'keras55_1_y_test.npy', arr = xy_test[0][1])

x_train = np.load(path + 'keras55_1_x_train.npy')
x_test = np.load(path + 'keras55_1_x_test.npy')
y_train = np.load(path + 'keras55_1_y_train.npy')
y_test = np.load(path + 'keras55_1_y_test.npy')
# print(x_train)
print(x_train.shape, x_test.shape)       #(160, 100, 100, 1) (120, 100, 100, 1)
print(y_train.shape, y_test.shape)       #(160,) (120,)

# print('========================================')
# print(type(xy_train)) #<class 'keras.preprocessing.image.DirectoryIterator'>
# print(type(xy_train[0])) #<class 'tuple'>
# print(type(xy_train[0][0])) #<class 'numpy.ndarray'>
# print(type(xy_train[0][1])) #<class 'numpy.ndarray'>

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Flatten

model = Sequential()
model.add(Conv2D(32, (2,2), padding='same', input_shape=(100,100,1), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(10, activation=LeakyReLU()))
model.add(Dense(20, activation=LeakyReLU()))
model.add(Dense(40, activation=LeakyReLU()))
model.add(Dense(80, activation=LeakyReLU()))
model.add(Dense(120, activation=LeakyReLU()))
model.add(Dense(1, activation='sigmoid'))

# #3.컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

# # # model.fit(xy_train[:][0], xy_train[:][1], epochs=10) #에러
# # model.fit(xy_train[0][0], xy_train[0][1], epochs=10, batch_size=16,
# #           validation_data=(xy_test[0][0], xy_test[0][1])) #통배치 할 시 이것도 가능
# # # hist = model.fit_generator(xy_train, epochs=500,
# # #                     steps_per_epoch=32,         #전체데이터크기/batch = 160/5 = 32  #한계에 맞춰주는게 좋음
# # #                     validation_data=xy_test, 
# # #                     validation_steps=24,           #발리데이터/batch = 120/5 = 24
# #                     # )
hist = model.fit(x_train, y_train , epochs=30,
                            steps_per_epoch=1,         #전체데이터크기/batch = 160/5 = 32  #한계에 맞춰주는게 좋음
                            validation_data=[x_test,y_test], 
                            validation_steps=1,           #발리데이터/batch = 120/5 = 24
 )
loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']
# print(acc)
print('loss : ', loss[-1])
print('val_loss : ', val_loss[-1])

print('acc : ', acc[-1])
print('val_acc : ', val_acc[-1])

#1. 그림그리기 subplot()
#2. 튜닝 0.95 이상
import matplotlib.pyplot as plt 
#시각화
plt.subplot(1,2,1)
plt.plot(hist.history['loss'], label='loss', c='red')
plt.plot(hist.history['val_loss'], label='val_loss', c='blue')
plt.subplot(1,2,2)
plt.plot(hist.history['acc'], label='acc', c='red')
plt.plot(hist.history['val_acc'], label='val_acc', c='blue')
plt.legend()
plt.grid()
plt.show()


