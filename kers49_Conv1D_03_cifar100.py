from keras.datasets import cifar100
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling2D, Dropout, Input, LSTM
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
import datetime
#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()


print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)




x_train = x_train.reshape(50000, 32*32*3)/255.
x_test = x_test.reshape(10000, 32*32*3)/255.         #reshape와 scaling 동시에 하기.


x_train = x_train.reshape(50000, 32, 32*3)
x_test = x_test.reshape(10000, 32, 32*3)


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
filepath='./_save/cnn/cifar10/'
filename='{epoch:04d}-{val_acc:.4f}.hdf5'



#2. 모델구성


input1 = Input(shape=(32, 32*3))
conv1 = Conv1D(20, 2 ,activation='linear')(input1)
conv2 = Conv1D(20, 2)(conv1)
flatten = Flatten()(conv2)
dense1 = Dense(10, activation='relu')(flatten)
dense2 = Dense(10, activation='relu')(dense1)
dense3 = Dense(10, activation='relu')(dense2)
output1 = Dense(10, activation='softmax')(dense3)

model=Model(inputs=input1, outputs=output1) 



#3. 컴파일, 훈련
import time
start_time = time.time()


es = EarlyStopping(monitor='val_acc',
                   mode='auto',
                   restore_best_weights=True,
                   patience=100)

mcp = ModelCheckpoint(monitor='val_acc',
                      mode='auto',
                      save_best_only=True,
                      verbose=1,
                      filepath = ''.join([filepath+'_k33_2_'+date+'_'+filename]))


model.compile(loss='categorical_crossentropy', optimizer='adam',    
              metrics=['acc'])

hist = model.fit(x_train,y_train,
                 epochs = 50,
                 batch_size =3200,
                 verbose=1,
                 validation_split=0.2,
                 callbacks=[es])

end_time = time.time()

#4. 평가, 예측

result = model.evaluate(x_test, y_test)
print('result :', result)

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_argm = np.argmax(y_test, axis=1)
acc = accuracy_score(y_argm, y_pred)

print('acc :', acc)

print('걸린시간 : ', round(end_time - start_time,2),'초')   


