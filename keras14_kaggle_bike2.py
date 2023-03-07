

################  < 정리된 실행 부분 >  ##################


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd


path = './_data/kaggle_bike/' 
     
train_csv = pd.read_csv(path+'train.csv', index_col=0)  

# print(train_csv)
# print(train_csv.shape)    


test_csv = pd.read_csv(path+'test.csv', index_col=0)

# print(test_csv)
# print(test_csv.shape)    

# print(train_csv.columns)
# print(train_csv.describe())

# print(type(train_csv))    
# print(train_csv.isnull().sum())

train_csv = train_csv.dropna()  
# print(train_csv.isnull().sum())    

# print(train_csv.info())
# print(train_csv.shape)    

x = train_csv.drop(['count', 'casual', 'registered'], axis=1)
y = train_csv['count']

# print(x)
# print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, 
    train_size=0.7,  
    shuffle=True,
    random_state=1000)

# print(x_train.shape, x_test.shape)   
# print(y_train.shape, y_test.shape)   


model = Sequential()
model.add(Dense(12, input_dim=8))
model.add(Dense(24, activation='relu'))
model.add(Dense(48, activation='relu'))
model.add(Dense(62, activation='relu'))
model.add(Dense(86, activation='relu'))
model.add(Dense(62, activation='relu'))
model.add(Dense(48, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=600, batch_size=2000, verbose=0)


loss= model.evaluate(x_test, y_test)
# print('loss : ', loss) 


y_predict=model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 =', r2)


def RMSE(y_test, y_predict):      
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)  
print("RMSE : ", rmse)


y_submit=model.predict(test_csv)


submission = pd.read_csv(path+'sampleSubmission.csv', index_col=0)
submission['count'] = y_submit

path2 = './_save/kaggle_bike/' 
submission.to_csv(path2 + 'submit_0307_003.csv')



################  < 작업 결과 >  ##################


# ( 작업 중, 가장 좋은 결과들 모음 )

# r2 = 0.32323733195602444
# RMSE :  146.79071851619048

# r2 = 0.3059892250809192
# RMSE :  146.12992994533235

# r2 = 0.31776197757603697
# RMSE :  144.8852015938704

# r2 = 0.31859675448754843
# RMSE :  144.79653470856368

#------------------

# r2 = 0.32013177053489894
# RMSE :  144.63334916265177

# model = Sequential()
# model.add(Dense(12, input_dim=8))
# model.add(Dense(24, activation='relu'))
# model.add(Dense(48, activation='relu'))
# model.add(Dense(62, activation='relu'))
# model.add(Dense(86, activation='relu'))
# model.add(Dense(62, activation='relu'))
# model.add(Dense(48, activation='relu'))
# model.add(Dense(24, activation='relu'))
# model.add(Dense(12, activation='relu'))
# model.add(Dense(1))

#    random_state=1000)
# model.fit(x_train, y_train, epochs=700, batch_size=2000, verbose=0)
# submission.to_csv(path2 + 'submit_0307_001.csv')

#------------------

# r2 = 0.32770557450924587
# RMSE :  143.82547751491177

# 위와 동일한 모델
#    random_state=1000)
# model.fit(x_train, y_train, epochs=600, batch_size=2000, verbose=0)
# submission.to_csv(path2 + 'submit_0307_002.csv')

#--------------------
