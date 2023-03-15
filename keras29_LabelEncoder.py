import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd


path = './_data/dacon_wine/'      
train_csv = pd.read_csv(path+'train.csv', index_col=0)  


print(train_csv)
print(train_csv.shape)  #(5497,13)  


test_csv = pd.read_csv(path+'test.csv', index_col=0)

print(test_csv)
print(test_csv.shape)   #(1000,12)

from sklearn.preprocessing import LabelEncoder,RobustScaler
le = LabelEncoder()
le.fit(train_csv['type'])
aaa = le.transform(train_csv['type'])
print(aaa)
print(type(aaa))   #<class 'numpy.ndarray'>
print(aaa.shape)   #(5497,)

# print(np.unique(aaa, return_counts=True)) #(array([0, 1]), array([1338, 4159] 0이 1338개 1이 4159개

train_csv['type'] = aaa
print(train_csv)
test_csv['type'] = le.transform(test_csv['type'])

print(le.transform(['red', 'white']))   #[0 1] 
print(le.transform(['white', 'red']))   #[1 0] 

#LabelEncoder : 사용 코드가 스케일러랑 유사함, 라벨 인코더는 분류형을 수치로 변환해줌

# scaler = RobustScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
'''
print(train_csv.columns)
print(train_csv.describe())

print(type(train_csv))    
print(train_csv.isnull().sum())

train_csv = train_csv.dropna()  
print(train_csv.isnull().sum())    

print(train_csv.info())
print(train_csv.shape)    

x = train_csv.drop(['count'], axis=1)
y = train_csv['count']

print(x)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, 
    train_size=0.7,  
    shuffle=True,
    random_state=1500)

print(x_train.shape, x_test.shape)   #(929, 9) (399, 9)
print(y_train.shape, y_test.shape)   #(929, ) (399, )


model = Sequential()
model.add(Dense(12, input_dim=9))
model.add(Dense(24))
model.add(Dense(48))
model.add(Dense(96))
model.add(Dense(48))
model.add(Dense(24))
model.add(Dense(12))
model.add(Dense(6))
model.add(Dense(1))
'''
