from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from tensorflow.python.keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np


#1. 데이터
datasets = fetch_covtype()
# print(datasets.DESCR)
# print(datasets.feature_names)

x= datasets.data
y= datasets['target']
print(x.shape, y.shape) #(581012, 54) (581012, )
print('y의 라벨값 : ', np.unique(y)) #y의 라벨값 :  [1 2 3 4 5 6 7]
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y.reshape(-1, 1))
print(y)
print(y.shape) #(581012, 7)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True,
    random_state=150612,
    train_size=0.8,
    stratify=y
)
print(y_train)
print(np.unique(y_train, return_counts=True))

# #2. 모델구성
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=54))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(7, activation='softmax'))

#. 컴파일, 훈련
model.compile(loss= 'categorical_crossentropy', optimizer='adam',
              metrics=['acc'])
es = EarlyStopping(monitor = 'val_loss', patience=20, mode='min',
                   verbose=1,
                   restore_best_weights=True
                   )
model.fit(x_train, y_train, epochs=300, batch_size=100000,
          validation_split=0.2,
          verbose=1,
          callbacks=[es])

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print(results)
print('loss : ', results[0])
print('acc : ', results[1])

y_predict = model.predict(x_test)
# print(y_predict.shape)  (116203, 7)

y_predict = np.argmax(y_predict, axis=-1)
# print(y_predict.shape)  (116203, )

y_true = np.argmax(y_test, axis=-1)

acc = accuracy_score(y_true, y_predict)

#==========================================정리==========================================

#위의 모델 같은 경우 Y의 라벨값이 [1 2 3 4 5 6 7]로 1부터 시작하는데 keras를 이용하는 경우
#숫자가 0부터 시작하므로 print(y.shape)를 하면 (581012, 8)과 같이 라벨값에 0이 있다고 생각하며
#출력을 해줘 사용을 할 수가 없음
#이와 같이 사이킷런을 이용한 원핫인코더 같은 경우 0부터 시작하지 않을 경우 사용하기 좋음
#머신러닝에 사용하기 좋음