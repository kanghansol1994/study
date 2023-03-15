from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,LeakyReLU,Dropout
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
    test_size=0.2,
    shuffle=True,
    stratify=y,
    random_state=417)

scaler = MinMaxScaler()
scaler.fit(x_train) #x를 바꿀 준비하라
x_train = scaler.transform(x_train) #x를 바꿔라
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)


#2. 모델구성
model = Sequential()
model.add(Dense(48, activation=LeakyReLU(0.5), input_dim=8))
model.add(Dense(24, activation=LeakyReLU(0.5)))
model.add(Dropout(0.2))
model.add(Dense(48, activation=LeakyReLU(0.5)))
model.add(Dense(24, activation=LeakyReLU(0.5)))
model.add(Dense(12, activation=LeakyReLU(0.5)))
model.add(Dense(1, activation='sigmoid'))

# model.summary()

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=200, mode='min',
                    verbose=1,
                    restore_best_weights=True,
)
hist= model.fit(x_train, y_train, epochs=5000, batch_size=16,
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
submission.to_csv(path_save+ 'submit_0314_0568.csv')


# seed:345
# model = Sequential()
# model.add(Dense(48, activation=LeakyReLU(0.5), input_dim=8))
# model.add(Dense(24, activation=LeakyReLU(0.5)))
# model.add(Dropout(0.2))
# model.add(Dense(48, activation=LeakyReLU(0.5)))
# model.add(Dense(24, activation=LeakyReLU(0.5)))
# model.add(Dense(12, activation=LeakyReLU(0.5)))
# model.add(Dense(1, activation='sigmoid'))
# patience=200, epochs=5000 batch_size=16
# results :  [0.4569121301174164, 0.8091602921485901]
# 데이콘:0.78

#seed:349
# results :  [0.4659003019332886, 0.7938931584358215]
# 데이콘:0.79

#seed:351
# results :  [0.4421497583389282, 0.8015267252922058]
# 데이콘:0.79

#seed:354
# results :  [0.4589114785194397, 0.7938931584358215]
# 데이콘:0.81

#seed : 359
# results :  [0.4560081362724304, 0.8167939186096191]
# 데이콘:0.8362

#ssed: 412
# results :  [0.44695985317230225, 0.8167939186096191]
# 데이콘:0.7931