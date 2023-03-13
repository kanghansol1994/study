from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense,Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,accuracy_score
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 

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
    test_size=0.1,
    shuffle=True,
    stratify=y,
    random_state=228)

scaler = MinMaxScaler()
scaler.fit(x_train) #x를 바꿀 준비하라
x_train = scaler.transform(x_train) #x를 바꿔라
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

#2. 모델구성
# model = Sequential()
# model.add(Dense(8, activation='linear', input_dim=8))
# model.add(Dense(4, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

input1 = Input(shape=(8,))
dense1 = Dense(8, activation='linear')(input1)
dense2 = Dense(4, activation='relu')(dense1)
dense3 = Dense(8, activation='relu')(dense2)
output1 = Dense(1, activation='sigmoid')(dense3)
model = Model(inputs = input1, outputs = output1)

model.summary()

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=300, mode='min',
                    verbose=1,
                    restore_best_weights=True,
)
hist= model.fit(x_train, y_train, epochs=9999, batch_size=32,
          validation_split=0.1,
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
submission.to_csv(path_save+ 'submit_0312_0542.csv')