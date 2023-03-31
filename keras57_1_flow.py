from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

(x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    shear_range=0.7,
    fill_mode='nearest'
)
augment_size = 100      #증폭

print(x_train.shape)     #(60000,28,28)
print(x_train[0].shape)  #(28,28)
print(x_train[1].shape)  #(28,28)
print(x_train[0][0].shape) #(28,)

print(np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1,28,28,1).shape) #1개의 이미지를 100개로 증폭시켜라
#np.tile(데이터, 증폭시키는 갯수)
print(np.zeros(augment_size)) #(100,28,28,1)
print(np.zeros(augment_size).shape) #(100,)

x_data = train_datagen.flow(                             #원래 있는 데이터를 변환,증폭
    np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1,28,28,1),           #x데이터
    np.zeros(augment_size),            #y데이터: 그림만 그릴꺼라 필요없어서 걍 0넣어줌
    batch_size=augment_size,
    shuffle=True,
    )                          
print(x_data) #<keras.preprocessing.image.NumpyArrayIterator object at 0x00000281D91B6820>
print(x_data[0])    #x와 y가 모두 포함되어있음
print(x_data[0][0].shape) #(100,28,28,1)
print(x_data[0][1].shape) #(100,)

import matplotlib.pyplot as plt
plt.figure(figsize=(7,7))
for i in range(49):
    plt.subplot(7, 7, i+1)
    plt.axis('off')
    plt.imshow(x_data[0][0][i], cmap='gray')
plt.show()