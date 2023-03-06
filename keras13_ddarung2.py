################  < 정리된 실행 부분 >  ##################


import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


path = './_data/ddarung/'     

train_csv = pd.read_csv(path+'train.csv', index_col=0)   
test_csv = pd.read_csv(path+'test.csv', index_col=0)
train_csv = train_csv.dropna()    

x = train_csv.drop(['count'], axis=1)
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(x, y, 
    train_size=0.7,  
    shuffle=True,
    random_state=300)


model = Sequential()
model.add(Dense(12, input_dim=9))
model.add(Dense(24))
model.add(Dense(48))
model.add(Dense(24))
model.add(Dense(12))
model.add(Dense(6))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=150, batch_size=60, verbose=0)


loss= model.evaluate(x_test, y_test)
print('loss : ', loss) 


y_predict=model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 =', r2)


def RMSE(y_test, y_predict):      
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)  
print("RMSE : ", rmse)


y_submit=model.predict(test_csv)

submission = pd.read_csv(path+'submission.csv', index_col=0)
submission['count'] = y_submit
submission.to_csv(path + 'submit_0306_0508.csv')


################  < 작업 결과 >  ##################


#********* ( No.2 )  ***********

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

# submission.to_csv(path + 'submit_0306_0501.csv')
#     random_state=1500)
# model.fit(x_train, y_train, epochs=350, batch_size=64, verbose=0)

# r2 = 0.6098745073855022
# RMSE :  48.00909290263378



#********* ( No.3 )  ***********

# 모델은 동일함
# submission.to_csv(path + 'submit_0306_0502.csv')
#     random_state=1500)
# model.fit(x_train, y_train, epochs=320, batch_size=30, verbose=0)

# r2 = 0.6100956554267617
# RMSE :  47.995483663939254


#********* ( No.4 )  ***********

# 모델은 동일함
# submission.to_csv(path + 'submit_0306_0503.csv')
#     random_state=1500)
# model.fit(x_train, y_train, epochs=500, batch_size=30, verbose=0)

# r2 = 0.6103661810705288
# RMSE :  47.978830525498765


#********* ( No.5 )  ***********

# model = Sequential()
# model.add(Dense(24, input_dim=9))
# model.add(Dense(48))
# model.add(Dense(126))
# model.add(Dense(48))
# model.add(Dense(24))
# model.add(Dense(12))
# model.add(Dense(1))

# submission.to_csv(path + 'submit_0306_0504.csv')
#     random_state=1500)
# model.fit(x_train, y_train, epochs=300, batch_size=10, verbose=0)

# r2 = 0.6110284604846858
# RMSE :  47.93803721616891


#********* ( No.6 )  ***********

# model = Sequential()
# model.add(Dense(10, input_dim=9))
# model.add(Dense(20))
# model.add(Dense(30))
# model.add(Dense(40))
# model.add(Dense(30))
# model.add(Dense(12))
# model.add(Dense(1))

# submission.to_csv(path + 'submit_0306_0505.csv')
#     random_state=300)
# model.fit(x_train, y_train, epochs=700, batch_size=80, verbose=0)

# r2 = 0.6341166401424012
# RMSE :  49.32214942862533


#********* ( No.7 )  ***********

# 모델은 동일함
# submission.to_csv(path + 'submit_0306_0506.csv')
#     random_state=300)
# model.fit(x_train, y_train, epochs=300, batch_size=70, verbose=0)

# r2 = 0.6312793867518595
# RMSE :  49.51301516165311


#********* ( No.8 )  ***********

# model = Sequential()
# model.add(Dense(12, input_dim=9))
# model.add(Dense(24))
# model.add(Dense(48))
# model.add(Dense(24))
# model.add(Dense(12))
# model.add(Dense(6))
# model.add(Dense(1))

# submission.to_csv(path + 'submit_0306_0507.csv')
#     random_state=300)
# model.fit(x_train, y_train, epochs=150, batch_size=60, verbose=0)

# r2 = 0.6284077139513462
# RMSE :  49.705450036067944


# --> DACON 리더보드 점수 = ( 74.3988109093 ) ----> 지금까지 작업해서 올린 8개 결과물 가운데 최고치 임


#********* ( No.9 )  ***********







################  < 수업 내용 >  ##################


# import numpy as np
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score, mean_squared_error
# import pandas as pd


# #1 데이터

# path = './_data/ddarung/'       # . ---> study folder를 의미함

# train_csv = pd.read_csv(path+'train.csv', index_col=0)   # 첫번째 칼럼 id를 제외함

#   --->   train_csv = pd.read_csv('./_date/ddarung/train.csv') 를 간략히 한 것

# print(train_csv)

# print(train_csv.shape)    # (1459, 10)  ---> id 제외됨, 헤더 역시 연산하지 않음


# test_csv = pd.read_csv(path+'test.csv', index_col=0)

# print(test_csv)
# print(test_csv.shape)    # (715, 9)  ---> id, count 제외됨, 헤더 역시 연산하지 않음


# #==================================================================================

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

# ################ 결측치 처리 ###############

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


# ################ train_csv 데이터에서 x와 y를 분리 ##################

# x = train_csv.drop(['count'], axis=1)
# y = train_csv['count']

# print(x)
# print(y)


# x_train, x_test, y_train, y_test = train_test_split(x, y, 
#     train_size=0.7,  
#     shuffle=True,
#     random_state=1000)

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


# 컴파일, 훈련

# model.compile(loss='mse', optimizer='adam')
# model.fit(x_train, y_train, epochs=250, batch_size=64, verbose=0)


# 평가, 예측

# loss= model.evaluate(x_test, y_test)
# print('loss : ', loss) 


# y_predict=model.predict(x_test)

# r2 = r2_score(y_test, y_predict)

# print('r2 =', r2)


# def RMSE(y_test, y_predict):       # RMSE 함수 정의
#     return np.sqrt(mean_squared_error(y_test, y_predict))

# rmse = RMSE(y_test, y_predict)   # RMSE 함수 사용

# print("RMSE : ", rmse)


# ######## submission.csv를 만들어 봅시다 !!! ##############


# print(test_csv.isnull().sum())    # 결측치 숫자 확인 ---> 여기도 결측치 있음


# y_submit=model.predict(test_csv)

# print(y_submit)

# submission = pd.read_csv(path+'submission.csv', index_col=0)

# submission['count'] = y_submit

# print(submission)

# submission.to_csv(path + 'submit_0306_0501.csv')
