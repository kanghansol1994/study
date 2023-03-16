import numpy as np
from tensorflow.keras.datasets import mnist   #데이터셋만 끌고 오면 되서 줄 뜬 상태로 그냥 사용
                                             
(x_train, y_train), (x_test, y_test) = mnist.load_data() #데이터를 넣어줌
# print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,) #흑백은 1이라 굳이 나오지 않았음
# print(x_test.shape, y_test.shape) #(10000, 28, 28) (10000,)   #cnn 모델 사용할때는 reshape이용해서 (60000,28,28,1)로 만들어줘야함

# print(x_train)
# print(y_train)
print(x_train[0])
print(y_train[3333])

import matplotlib.pyplot as plt
plt.imshow(x_train[3333], 'gray') #그림 보여줌
plt.show()