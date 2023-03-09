import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,accuracy_score


#1. 데이터
path = './_data/dacon_diabetes/'
path_save ='./_save/dacon_diabetes/'

train_csv = pd.read_csv(path+'train.csv', index_col=0)
# print(train_csv.shape) #(652, 9)
test_csv = pd.read_csv(path+'test.csv', index_col=0)
# print(test_csv.shape) #(116, 8)
train_csv = train_csv.dropna()

x = train_csv.drop(['Outcome'], axis=1)
y = train_csv['Outcome']

x_train, x_test, y_train, y_test = train_test_split(x, y,
    test_size=0.3,
    shuffle=True,
    stratify=y,
    random_state=34356)

#2. 모델구성
model = Sequential()
model.add(Dense(10, activation='relu', input_dim=8))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam')
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='accuracy', patience=5, mode='min',
                    verbose=1,
                    restore_best_weights=True,
)
hist= model.fit(x_train, y_train, epochs=500, batch_size=10,
          validation_split=0.2,
          verbose=1,
          callbacks=[es],
          )


#4. 평가, 예측
results=model.evaluate(x_test, y_test)
print('results : ', results)
y_predict = np.round(model.predict(x_test))
acc = accuracy_score(y_test, y_predict)
print('acc : ', acc)
r2 = r2_score(y_test, y_predict)
print('r2스코어: ', r2)

y_submit=np.round(model.predict(test_csv))

submission = pd.read_csv(path+'sample_submission.csv', index_col=0)
submission['Outcome'] = y_submit
submission.to_csv(path_save+ 'submit_0309_0511.csv')