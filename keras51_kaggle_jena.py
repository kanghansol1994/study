import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
#1.데이터
path = './_data/kaggle_jena/'
datasets = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col=0) 
print(datasets) #(420551, 14)
print(datasets.columns)
print(datasets.info())
print(datasets.describe())
print(datasets['T (degC)'])   
print(datasets['T (degC)'].values) #판다스를 넘파이로 바꿔주는 방법 #[-8.02 -8.41 -8.51 ... -3.16 -4.23 -4.82]
print(datasets['T (degC)'].to_numpy()) #바꿔주는 방법 2

# plt.plot(datasets['T (degC)'].values)
# plt.show()
df = pd.DataFrame(datasets)
x =df.drop(df.columns[1], axis=1).to_numpy()
# print(x.shape) #(420551,13)
y=df.iloc[:, 1].to_numpy()
# print(x.shape) #(420551, 13)
# print(y.shape) #(420551,)


timesteps = 6
def split_x(datasets, timesteps):
    x = []
    for i in range(len(datasets) - timesteps ):
        subset = datasets[i : (i + timesteps)]
        x.append(subset)
    return np.array(x)

x = split_x(x, timesteps)
print(x)
print(x.shape) #(420545,6,13)
y = y[timesteps:]
print(y)
print(y.shape) #(420545,)

x_train,x_test,y_train,y_test = train_test_split(x, y, train_size=0.7, shuffle=False)
print(x_train.shape, y_train.shape) #(294381,6,13) (294381,)
x_test,x_predict,y_test,y_predict = train_test_split(x_test,y_test, train_size=2/3, shuffle=False)
print(x_test.shape, y_test.shape) #(84109,6,13) (84109,)
print(x_predict.shape,y_predict.shape) #(42055,6,13) (42055)

x_train = x_train.reshape(294381, 78)
x_test = x_test.reshape(84109, 78)
scaler=StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(294381,6,13)
x_test = x_test.reshape(84109,6,13)

#2. 모델구성
model=Sequential()
model.add(LSTM(256,input_shape=(6,13)))
model.add(Dense(100,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(25,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1))

model.summary()





#3. 컴파일, 훈련
from tensorflow.python.keras.callbacks import EarlyStopping
model.compile(loss='mse',optimizer='adam')

es = EarlyStopping(monitor='val_loss', patience=50, mode='min',
              verbose=1,
              restore_best_weights=True, #디폴트는 false) 
              )
model.fit(x_train,y_train
          ,epochs=300,batch_size=1000
          ,validation_split=0.1,verbose=1, callbacks=[es])







#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
y_predict = model.predict(x_test)
def RMSE(y_test, y_predict): 
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict[:len(y_test)])
print("RMSE : ", rmse)
y_predict= model.predict(x_test)
r2= r2_score(y_test, y_predict)
print('r2스코어: ', r2)




# mse,rmse 쓰기 7 : 2 : 1 timesteps:10