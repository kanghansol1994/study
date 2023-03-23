from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv1D, MaxPooling2D, Flatten, LSTM, GRU
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
import time
import pandas as pd
#1. 데이터

path = './_data/ddarung/'
path_save = './_save/ddarung/'
train_csv = pd.read_csv(path + 'train.csv', index_col = 0)
test_csv = pd.read_csv(path + 'test.csv', index_col = 0)
submission = pd.read_csv(path+'submission.csv', index_col = 0)

train_csv = train_csv.dropna()

x = train_csv.drop(['count'], axis=1)
y = train_csv['count']



# print(x.shape)  #(1328, 9)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    random_state=221,
                                                    )


scaler=StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# print(np.min(x_test), np.max(x_test))
print(x_train.shape, x_test.shape)  #(1062, 9) (266, 9)
x_train = x_train.reshape(-1, 3, 3)
x_test = x_test.reshape(-1, 3, 3)
test_csv=np.array(test_csv)
test_csv = scaler.transform(test_csv)
test_csv=test_csv.reshape(-1, 3, 3)



#2. 모델 구성

input1 = Input(shape=(3, 3))
conv1 = Conv1D(20, 2 ,activation='linear')(input1)
conv2 = Conv1D(20, 2)(conv1)
flatten = Flatten()(conv2)
dense1 = Dense(10, activation='relu')(flatten)
dense2 = Dense(10, activation='relu')(dense1)
dense3 = Dense(10, activation='relu')(dense2)
output1 = Dense(1)(dense3)

model=Model(inputs=input1, outputs=output1)


#3. 컴파일, 훈련

start_time=time.time()
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor = 'loss',
                   patience=20,
                   restore_best_weights=True,
                   verbose=1)
hist = model.fit(x_train, y_train,
                 epochs =300,
                 batch_size=128,
                 validation_split = 0.2,
                 callbacks=[es])

end_time=time.time()

#4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('result :', result)

y_pred=model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print('r2 :', r2)

#Conv1D
# result : 2659.190673828125
# r2 : 0.6188120280050757

#RNN
# result : 2540.751220703125
# r2 : 0.6357899519963421
