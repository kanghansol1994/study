from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,LeakyReLU,Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,accuracy_score,mean_squared_error
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler, LabelEncoder 
from tensorflow.keras.utils import to_categorical

#1. 데이터
path = './_data/kaggle_house/'
path_save ='./_save/kaggle_house/'

train_csv = pd.read_csv(path+'train.csv', index_col=0)
print(train_csv.shape) #(1460, 80)
test_csv = pd.read_csv(path+'test.csv', index_col=0)
print(test_csv.shape) #(1459, 79)
print(train_csv.info())
print(test_csv.info())
non_numeric_cols = train_csv.select_dtypes(exclude=np.number).columns.tolist()
train_csv = train_csv.drop(non_numeric_cols, axis=1)
test_csv = test_csv.drop(non_numeric_cols, axis=1)
train_csv = train_csv.dropna()
train_csv = train_csv.fillna(0)
print(train_csv.isnull().sum())
print(train_csv.shape)

le=LabelEncoder()
for i in train_csv.columns:
    if train_csv[i].dtype=='object':
        train_csv[i] = le.fit_transform(train_csv[i])
        test_csv[i] = le.fit_transform(test_csv[i])
print(len(train_csv.columns))
print(train_csv.info())
train_csv=train_csv.dropna()
print(train_csv.shape)

x = train_csv.drop(['SalePrice'], axis=1)
x = x.dropna()
y = train_csv['SalePrice']
print(x.shape,y.shape) #(1121, 36) (1121,)


x_train, x_test, y_train, y_test = train_test_split(x, y,
    test_size=0.2,
    shuffle=True,
    random_state=328)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)




#2. 모델구성
model = Sequential()
model.add(Dense(64, activation=LeakyReLU(0.5), input_dim=36))
model.add(Dense(64, activation=LeakyReLU(0.5)))
model.add(Dense(32, activation=LeakyReLU(0.5)))
model.add(Dense(64, activation=LeakyReLU(0.5)))
model.add(Dense(48, activation=LeakyReLU(0.5)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

# model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam',)
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=50, mode='min',
                    verbose=1,
                    restore_best_weights=True,
)

hist= model.fit(x_train, y_train, epochs=1000, batch_size=256,
          validation_split=0.2,
          verbose=1,
          callbacks=[es],
          )


#4. 평가, 예측
results=model.evaluate(x_test, y_test)
print('results : ', results)
y_predict = model.predict(x_test)


y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2스코어: ', r2)

def RMSE(y_test, y_predict): 
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test,y_predict)
print("RMSE : ", rmse)
#4-1 내보내기
y_submit = model.predict(test_csv)
submission = pd.read_csv(path+'sample_submission.csv', index_col=0)
y_submit = pd.DataFrame(y_submit)
y_submit = y_submit.fillna(0)
y_submit = np.array(y_submit)
submission['SalePrice'] = y_submit
submission.to_csv(path_save+ 'submit_0318_0510.csv')


