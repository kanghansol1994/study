

################  < 정리된 실행 부분 >  ##################


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd


path = './_data/kaggle_bike/' 
     
train_csv = pd.read_csv(path+'train.csv', index_col=0)  

print(train_csv)
print(train_csv.shape)    


test_csv = pd.read_csv(path+'test.csv', index_col=0)

print(test_csv)
print(test_csv.shape)    

print(train_csv.columns)
print(train_csv.describe())

print(type(train_csv))    
print(train_csv.isnull().sum())

train_csv = train_csv.dropna()  
print(train_csv.isnull().sum())    

print(train_csv.info())
print(train_csv.shape)    

x = train_csv.drop(['count', 'casual', 'registered'], axis=1)
y = train_csv['count']

print(x)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, 
    train_size=0.7,  
    shuffle=True,
    random_state=1500)

print(x_train.shape, x_test.shape)   
print(y_train.shape, y_test.shape)   


model = Sequential()
model.add(Dense(12, input_dim=8))
model.add(Dense(24))
model.add(Dense(48))
model.add(Dense(96))
model.add(Dense(48))
model.add(Dense(24))
model.add(Dense(12))
model.add(Dense(6))
model.add(Dense(1))