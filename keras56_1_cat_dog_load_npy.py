#불러와서 모델 완성

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D,LeakyReLU,Conv2D,Flatten,Dense
import time
from sklearn.model_selection import train_test_split
#1. 데이터
path = 'd:/study_data/_data/cat_dog/Petimages/'
save_path = 'd:/study_data/_save/cat_dog/'

start_time = time.time()
x_train = np.load(save_path + 'keras55_1_x_train.npy')
y_train = np.load(save_path + 'keras55_1_y_train.npy')

end_time = time.time()
print('걸린시간 : ', round(end_time - start_time,2),'초') #1.99초

x = np.load(save_path +'keras55_1_x_train.npy') 
y = np.load(save_path +'keras55_1_y_train.npy')

x_train, x_test, y_train, y_test = train_test_split(x, y , train_size=0.7, shuffle=True)

#2.모델구성
model = Sequential()
model.add(Conv2D(32, (2,2), padding='same', input_shape=(300,300,3), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# #3.컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

hist = model.fit(x, y, epochs=100,
                       steps_per_epoch=17,         #전체데이터크기/batch = 160/5 = 32  #한계에 맞춰주는게 좋음
                       validation_split = 0.2, 
                       validation_steps=17,           #발리데이터/batch = 120/5 = 24
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