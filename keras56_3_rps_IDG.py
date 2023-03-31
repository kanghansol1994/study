import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Flatten,Dense, MaxPooling2D,LeakyReLU
import datetime
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

path = 'd:/study_data/_data/rps/'
save_path = 'd:/study_data/_save/rps/'
# np.save(save_path + '파일명', arr=???)
start_time=time.time()

train_datagen = ImageDataGenerator(
     rescale =1./255,)


# test_datagen = ImageDataGenerator(
#      rescale =1./255) 
 #test는 평가데이터이므로 통상적으로 증폭을 시키는것은 데이터 조작
xy_train = train_datagen.flow_from_directory(
    'd:/study_data/_data/rps/',                        
     target_size=(150,150),                
     batch_size=840,                  #전체데이터쓸려면 160(전체데이터갯수) 이상 넣으면 됨
     class_mode='categorical',          #catrgorical: 원핫인코딩 사용과 같은 효과를 냄
     color_mode='rgb',
    #  color_mode='rgb',
     shuffle=True,)     #폴더에서 가져올거야 ad,normal까지 가면 y로 지정을 할수가 없음
      #Found 160 images belonging to 2 classes.  
      #x_train = (160,200,200,1) y = (160, )
# xy_test = test_datagen.flow_from_directory(
#     'd:/study_data/_data/cat_dog/PetImages',                        
#      target_size=(300,300),
#      batch_size=500,
#      class_mode='binary',                       #y의 클래스 binary:y를 0과1로 수치화
#      color_mode='rgb',
#      shuffle=True,
     #Found 120 images belonging to 2 classes.
      #x_test = (120,200,200,1) y = (120,)
x= xy_train[0][0]
y= xy_train[0][1]
end_time = time.time()   
print('걸린시간 : ', round(end_time - start_time,2),'초')    #0.86초

x_train,x_test,y_train,y_test = train_test_split( x, y , train_size=0.7, shuffle=True)

# print(xy_train)
# # <keras.preprocessing.image.DirectoryIterator object at 0x0000026E58295F70>
# print(xy_train[0])
# print(len(xy_train))                       #2
# print(len(xy_train[0]))                    #2
# print(xy_train[0][0])      #엑스 다섯개 들어가있다.
# print(xy_train[0][1])      #[1,1,0,0,1]

# print(xy_train[0][0].shape)  #(12500, 200, 200, 3)  (배치사이즈, size , size, 1)
# print(xy_test[0][0].shape)  #(12500, 200, 200, 3)
# print(xy_train[0][1].shape)  #(12500,)
# print(xy_test[0][1].shape)  #(12500,)
start_time = time.time()

np.save(save_path + 'keras56_5_x_train.npy', arr = x_train)
np.save(save_path + 'keras56_5_x_test.npy', arr = x_test)
np.save(save_path + 'keras56_5_y_train.npy', arr = y_train)
np.save(save_path + 'keras56_5_y_test.npy', arr = y_test)


end_time = time.time()
print('걸린시간 : ', round(end_time - start_time,2),'초')  #10.57초

 
# print('========================================')
# print(type(xy_train)) #<class 'keras.preprocessing.image.DirectoryIterator'>
# print(type(xy_train[0])) #<class 'tuple'>
# print(type(xy_train[0][0])) #<class 'numpy.ndarray'>
# print(type(xy_train[0][1])) #<class 'numpy.ndarray'>

#2. 모델구성
# model = Sequential()
# model.add(Conv2D(32, (2,2), padding='same', input_shape=(300,300,3), activation='relu'))
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
# hist = model.fit(x,y, epochs=30,
#                             steps_per_epoch=32,         #전체데이터크기/batch = 160/5 = 32  #한계에 맞춰주는게 좋음
#                             validation_split=0.2, 
#                             validation_steps=24,           #발리데이터/batch = 120/5 = 24
#  )