import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import MaxPooling2D,LeakyReLU


#1. 데이터
train_datagen = ImageDataGenerator(
     rescale =1./255,                   
     horizontal_flip=True,                      
     vertical_flip=True, 
     width_shift_range=0.1, 
     height_shift_range=0.1, 
     rotation_range=5, 
     zoom_range=1.2, 
     shear_range=0.7,
     fill_mode='nearest') 


test_datagen = ImageDataGenerator(
     rescale =1./255) 
 #test는 평가데이터이므로 통상적으로 증폭을 시키는것은 데이터 조작

xy_train = train_datagen.flow_from_directory(
    'd:/study_data/_data/brain/train/',                        
     target_size=(100,100),                
     batch_size=5000,                  #전체데이터쓸려면 160(전체데이터갯수) 이상 넣으면 됨
     class_mode='binary',          #catrgorical: 원핫인코딩 사용과 같은 효과를 냄
     color_mode='grayscale',
    #  color_mode='rgb',
     shuffle=True,)     #폴더에서 가져올거야 ad,normal까지 가면 y로 지정을 할수가 없음
      #Found 160 images belonging to 2 classes.  
      #x_train = (160,200,200,1) y = (160, )
xy_test = test_datagen.flow_from_directory(
    'd:/study_data/_data/brain/test/',                        
     target_size=(100,100),
     batch_size=120,
     class_mode='binary',                       #y의 클래스 binary:y를 0과1로 수치화
     color_mode='grayscale',
     shuffle=True,
)     #Found 120 images belonging to 2 classes.
      #x_test = (120,200,200,1) y = (120,)
       
print(xy_train)
# <keras.preprocessing.image.DirectoryIterator object at 0x0000026E58295F70>
print(xy_train[0])
print(len(xy_train))                       #32
print(len(xy_train[0]))                    #2
print(xy_train[0][0])      #엑스 다섯개 들어가있다.
print(xy_train[0][1])      #[1,1,0,0,1]

print(xy_train[0][0].shape)  #(160, 100, 100, 1)  (배치사이즈, size , size, 1)
print(xy_test[0][0].shape)  #(120, 100, 100, 1)
print(xy_train[0][1].shape)  #(160,)
print(xy_test[0][1].shape)  #(120,)

path = 'd:/study_data/_save/_npy/'
np.save(path + 'keras55_1_x_train.npy', arr = xy_train[0][0])
np.save(path + 'keras55_1_x_test.npy', arr = xy_test[0][0])
np.save(path + 'keras55_1_y_train.npy', arr = xy_train[0][1])
np.save(path + 'keras55_1_y_test.npy', arr = xy_test[0][1])



# print('========================================')
# print(type(xy_train)) #<class 'keras.preprocessing.image.DirectoryIterator'>
# print(type(xy_train[0])) #<class 'tuple'>
# print(type(xy_train[0][0])) #<class 'numpy.ndarray'>
# print(type(xy_train[0][1])) #<class 'numpy.ndarray'>

#2. 모델구성
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense,Conv2D,Flatten

# model = Sequential()
# model.add(Conv2D(32, (2,2), padding='same', input_shape=(100,100,1), activation='relu'))
# model.add(MaxPooling2D())
# model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
# model.add(MaxPooling2D())
# model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
# model.add(MaxPooling2D())
# model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
# model.add(MaxPooling2D())
# model.add(Flatten())
# model.add(Dense(10, activation=LeakyReLU()))
# model.add(Dense(20, activation=LeakyReLU()))
# model.add(Dense(40, activation=LeakyReLU()))
# model.add(Dense(80, activation=LeakyReLU()))
# model.add(Dense(120, activation=LeakyReLU()))
# model.add(Dense(1, activation='sigmoid'))

# #3.컴파일, 훈련
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

# # # model.fit(xy_train[:][0], xy_train[:][1], epochs=10) #에러
# # model.fit(xy_train[0][0], xy_train[0][1], epochs=10, batch_size=16,
# #           validation_data=(xy_test[0][0], xy_test[0][1])) #통배치 할 시 이것도 가능
# # # hist = model.fit_generator(xy_train, epochs=500,
# # #                     steps_per_epoch=32,         #전체데이터크기/batch = 160/5 = 32  #한계에 맞춰주는게 좋음
# # #                     validation_data=xy_test, 
# # #                     validation_steps=24,           #발리데이터/batch = 120/5 = 24
# #                     # )
# hist = model.fit(xy_train, epochs=30,
#                             steps_per_epoch=32,         #전체데이터크기/batch = 160/5 = 32  #한계에 맞춰주는게 좋음
#                             validation_data=xy_test, 
#                             validation_steps=24,           #발리데이터/batch = 120/5 = 24
#  )
# loss = hist.history['loss']
# val_loss = hist.history['val_loss']
# acc = hist.history['acc']
# val_acc = hist.history['val_acc']
# # print(acc)
# print('loss : ', loss[-1])
# print('val_loss : ', val_loss[-1])

# print('acc : ', acc[-1])
# print('val_acc : ', val_acc[-1])

# #1. 그림그리기 subplot()
# #2. 튜닝 0.95 이상
# import matplotlib.pyplot as plt 
# #시각화
# plt.subplot(1,2,1)
# plt.plot(hist.history['loss'], label='loss', c='red')
# plt.plot(hist.history['val_loss'], label='val_loss', c='blue')
# plt.subplot(1,2,2)
# plt.plot(hist.history['acc'], label='acc', c='red')
# plt.plot(hist.history['val_acc'], label='val_acc', c='blue')
# plt.legend()
# plt.grid()
# plt.show()


