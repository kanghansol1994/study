# from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv1D, MaxPooling2D, Flatten, GRU, LSTM
import numpy as np
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
import time
import pandas as pd 
from tensorflow.keras.utils import to_categorical
#1. 데이터

path = './_data/dacon_diabetes/'
path_save = './_save/dacon_diabetes/'

datasets = pd.read_csv(path + 'train.csv',index_col=0)
test_set = pd.read_csv(path + 'test.csv',index_col=0)
submission = pd.read_csv(path +'sample_submission.csv', index_col=0)

# print(train_csv.shape) #(10886, 11)
x = datasets.drop(['Outcome'], axis=1)
y = datasets['Outcome']
print(x.shape, y.shape)


x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    random_state=221,
                                                    stratify=y
                                                    )

print(x_train.shape, x_test.shape)  



scaler=MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# print(np.min(x_test), np.max(x_test))
# print(x_train.shape, x_test.shape)  
x_train = x_train.reshape(-1, 4, 2)
x_test = x_test.reshape(-1, 4, 2)



#2. 모델 구성

input1 = Input(shape=(4, 2))
conv1 = Conv1D(20, 2 ,activation='linear')(input1)
conv2 = Conv1D(20, 2)(conv1)
flatten = Flatten()(conv2)
dense1 = Dense(10, activation='relu')(flatten)
dense2 = Dense(10, activation='relu')(dense1)
dense3 = Dense(10, activation='relu')(dense2)
output1 = Dense(1, activation='sigmoid')(dense3)

model=Model(inputs=input1, outputs=output1)

#3. 컴파일, 훈련
start_time=time.time()
model.compile(loss='binary_crossentropy', optimizer='adam')
es = EarlyStopping(monitor = 'val_loss',
                   patience=20,
                   restore_best_weights=True,
                   verbose=1)
hist = model.fit(x_train, y_train,
                 epochs =300,
                 batch_size=1024,
                 validation_split = 0.2,
                 callbacks=[es])

end_time=time.time()

#4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('result :', result)

y_pred=np.round(model.predict(x_test))
acc = accuracy_score(y_test, y_pred)
print('acc :', acc)


#RNN
# result : 0.549931526184082
# acc : 0.6946564885496184

#Conv1D
# result : 0.5436695218086243
# acc : 0.7404580152671756