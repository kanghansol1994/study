import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd


path = './_data/ddarung/'      
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

x = train_csv.drop(['count'], axis=1)
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
model.add(Dense(12, input_dim=9))
model.add(Dense(24))
model.add(Dense(48))
model.add(Dense(96))
model.add(Dense(48))
model.add(Dense(24))
model.add(Dense(12))
model.add(Dense(6))
model.add(Dense(1))


################  < 작업 결과 >  ##################


#  수업 내용 가운데 대부분의 출력 결과들이 표기되어 있음


################  < 수업 내용 >  ##################


# import numpy as np
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score, mean_squared_error
# import pandas as pd


#1 데이터

# path = './_data/ddarung/'       # . ---> study folder를 의미함

# train_csv = pd.read_csv(path+'train.csv', index_col=0)   # 첫번째 칼럼 id를 제외함

#   --->   train_csv = pd.read_csv('./_date/ddarung/train.csv') 를 간략히 한 것

# print(train_csv)

# print(train_csv.shape)    # (1459, 10)  ---> id 제외됨, 헤더 역시 연산하지 않음


# test_csv = pd.read_csv(path+'test.csv', index_col=0)

# print(test_csv)
# print(test_csv.shape)    # (715, 9)  ---> id, count 제외됨, 헤더 역시 연산하지 않음


#==================================================================================

# print(train_csv.columns)

#  Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#         'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#         'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#        dtype='object')

# print(train_csv.info())

#  0   hour                    1459 non-null   int64
#  1   hour_bef_temperature    1457 non-null   float64
#  2   hour_bef_precipitation  1457 non-null   float64
#  3   hour_bef_windspeed      1450 non-null   float64
#  4   hour_bef_humidity       1457 non-null   float64
#  5   hour_bef_visibility     1457 non-null   float64
#  6   hour_bef_ozone          1383 non-null   float64
#  7   hour_bef_pm10           1369 non-null   float64
#  8   hour_bef_pm2.5          1342 non-null   float64
#  9   count                   1459 non-null   float64


# print(train_csv.describe())

# print(type(train_csv))    # <class 'pandas.core.frame.Data


################ 결측치 처리 ###############

# print(train_csv.isnull().sum())

# hour                        0
# hour_bef_temperature        2
# hour_bef_precipitation      2
# hour_bef_windspeed          9
# hour_bef_humidity           2
# hour_bef_visibility         2
# hour_bef_ozone             76
# hour_bef_pm10              90
# hour_bef_pm2.5            117
# count                       0


# train_csv = train_csv.dropna()     # 결측치 제거

# print(train_csv.isnull().sum())    # 결측치 숫자 확인

# hour                      0
# hour_bef_temperature      0
# hour_bef_precipitation    0
# hour_bef_windspeed        0
# hour_bef_humidity         0
# hour_bef_visibility       0
# hour_bef_ozone            0
# hour_bef_pm10             0
# hour_bef_pm2.5            0
# count                     0


# print(train_csv.info())

# Int64Index: 1328 entries, 3 to 2179


# print(train_csv.shape)     # (1328, 10)

#  0   hour                    1328 non-null   int64
#  1   hour_bef_temperature    1328 non-null   float64
#  2   hour_bef_precipitation  1328 non-null   float64
#  3   hour_bef_windspeed      1328 non-null   float64
#  4   hour_bef_humidity       1328 non-null   float64
#  5   hour_bef_visibility     1328 non-null   float64
#  6   hour_bef_ozone          1328 non-null   float64
#  7   hour_bef_pm10           1328 non-null   float64
#  8   hour_bef_pm2.5          1328 non-null   float64
#  9   count                   1328 non-null   float64


################ train_csv 데이터에서 x와 y를 분리 ##################

# x = train_csv.drop(['count'], axis=1)
# y = train_csv['count']

# print(x)
# print(y)


# x_train, x_test, y_train, y_test = train_test_split(x, y, 
#     train_size=0.7,  
#     shuffle=True,
#     random_state=1500)

# print(x_train.shape, x_test.shape)    # (929, 9) (399, 9)
# print(y_train.shape, y_test.shape)    # (929,) (399,)



# 모델구성

# model = Sequential()
# model.add(Dense(12, input_dim=9))
# model.add(Dense(24))
# model.add(Dense(48))
# model.add(Dense(96))
# model.add(Dense(48))
# model.add(Dense(24))
# model.add(Dense(12))
# model.add(Dense(6))
# model.add(Dense(1))