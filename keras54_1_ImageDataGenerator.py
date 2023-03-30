import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
     rescale =1./255,)                     #부동소수점으로 연산을 해라, 정규화, 255로 나눠주기
#      horizontal_flip=True,                         #상하반전
#      vertical_flip=True, #좌우반전
#      width_shift_range=0.1, #10프로만큼 사진을 좌우로 이동 가능
#      height_shift_range=0.1, #10프로만큼 위아래 이동 가능
#      rotation_range=5, #회전횟수
#      zoom_range=1.2, # 배율
#      shear_range=0.7, #찌그러뜨리기
#      fill_mode='nearst', #근처값


test_datagen = ImageDataGenerator(
     rescale =1./255) 
 #test는 평가데이터이므로 통상적으로 증폭을 시키는것은 데이터 조작

xy_train = train_datagen.flow_from_directory(
    'd:/study_data/_data/brain/train/',                        
     target_size=(200,200),                
     batch_size=5,
     class_mode='binary',
     color_mode='grayscale',
     shuffle=True,
)     #폴더에서 가져올거야 ad,normal까지 가면 y로 지정을 할수가 없음
      #Found 160 images belonging to 2 classes.  
      #x_train = (160,200,200,1) y = (160, )
xy_test = test_datagen.flow_from_directory(
    'd:/study_data/_data/brain/test/',                        
     target_size=(200,200),
     batch_size=5,
     class_mode='binary',
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

print(xy_train[0][0].shape)  #(5,200,200,1)  (배치사이즈, size , size, 1)
print(xy_train[0][1].shape)  #(5,)

print('========================================')
print(type(xy_train)) #<class 'keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0])) #<class 'tuple'>
print(type(xy_train[0][0])) #<class 'numpy.ndarray'>
print(type(xy_train[0][1])) #<class 'numpy.ndarray'>