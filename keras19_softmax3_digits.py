import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
#1. 데이터
datasets = load_digits()
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets['target']
print(x.shape, y.shape) #(1797, 64) (1797, )
print(x)
print(y)
print('y의 라벨값 : ', np.unique(y)) #y의 라벨값 :  [0 1 2 3 4 5 6 7 8 9]
y = to_categorical(y)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y , shuffle=True, 
    random_state=2536, 
    train_size=0.8,
    stratify=y
)
print(y_train)
print(np.unique(y_train, return_counts=True))

#2.모델구성
model = Sequential()
model.add(Dense(50, activation='relu', input_dim=64))
model.add(Dense(40, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer='adam',
              metrics=['acc'])

from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='var_loss', patience=20, mode='min',
                   verbose=1,
                   restore_best_weights=True
                   )
model.fit(x_train, y_train, epochs=10, batch_size=50,
          validation_split=0.2,
          verbose=1,
          callbacks=[es]
          )

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('results : ', results)

y_predict = model.predict(x_test)
print(y_predict.shape)
y_predict = np.argmax(y_predict, axis=1)
print(y_predict.shape)

y_true = np.argmax(y_test, axis=1)

acc = accuracy_score(y_true, y_predict)
print('acc : ', acc)
#results :[0.1758653074502945, 0.9777777791023254]