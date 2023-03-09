import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
#1. 데이터
datasets = load_iris()
print(datasets.DESCR) #판다스 describe()
print(datasets.feature_names) #판다스 columns
#['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

x = datasets.data
y = datasets['target']
print(x.shape, y.shape) #(150, 4) (150, )
print(x)
print(y)
print('y의 라벨값 : ', np.unique(y))  #y의 라벨값 :  [0 1 2]
# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)
# print(y)
#판다스에 겟더미, 사이킷런에 원핫인코더
y= pd.get_dummies(y)
# encoder = OneHotEncoder(sparse=False)
# y = encoder.fit_transform(y.reshape(-1, 1))
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y , shuffle=True, 
    random_state=456, 
    train_size=0.8,
    stratify=y
)
print(y_train)
print(np.unique(y_train, return_counts=True))

#2. 모델구성
model = Sequential()
model.add(Dense(50, activation='relu', input_dim=4))
model.add(Dense(40, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer='adam',
              metrics=['acc'])
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='var_loss', patience=20, mode='min',
                   verbose=1,
                   restore_best_weights=True
                   )

model.fit(x_train, y_train, epochs=100, batch_size=2,
          validation_split=0.2,
          verbose=1,
          callbacks=[es]
          )

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('results : ', results)

y_predict = np.round(model.predict(x_test))
from sklearn.metrics import r2_score, accuracy_score
acc = accuracy_score(y_test, y_predict)
print('acc스코어 : ', acc)

#accuracy_score를 사용해서 스코어를 빼세요
#results :  [0.11955048143863678, 0.9666666388511658]