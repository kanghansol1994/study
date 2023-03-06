import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, 
    train_size=0.7,  
    shuffle=True,
    random_state=800)

model = Sequential()
model.add(Dense(8, input_dim=13))
model.add(Dense(24))
model.add(Dense(36))
model.add(Dense(48))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1))

model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs=10, batch_size=2, verbose=0)

loss= model.evaluate(x_test, y_test)
print('loss : ', loss) 

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2 =', r2)


################ < 작업 결과 > #####################


# To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
# 5/5 [==============================] - 0s 4ms/step - loss: 5.8228
# loss :  5.822839736938477
# 5/5 [==============================] - 0s 728us/step
# r2 = 0.1716769246589145


################ < 수업 내용 > #####################

# import numpy as np

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from sklearn.datasets import load_boston
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score

# datasets = load_boston()
# x = datasets.data
# y = datasets.target

# x_train, x_test, y_train, y_test = train_test_split(x, y, 
#     train_size=0.7,  
#     shuffle=True,
#     random_state=800)

# model = Sequential()
# model.add(Dense(8, input_dim=13))
# model.add(Dense(24))
# model.add(Dense(36))
# model.add(Dense(48))
# model.add(Dense(64))
# model.add(Dense(32))
# model.add(Dense(16))
# model.add(Dense(8))
# model.add(Dense(1))

# model.compile(loss='mae', optimizer='adam')
# model.fit(x_train, y_train, epochs=10, batch_size=2, verbose='0')

# # verbose 
# # 0 아무것도 안나온다.
# # 1 다 보여준다 / 디폴트 값이며 'auto'로 표시 가능함
# # 2 프로그래스바만 없어진다
# # 3,4,5.... 에포만 나온다. ( 0,1,2 를 제외한 모든 값(음수 포함)에 대하여 )

# loss= model.evaluate(x_test, y_test)
# print('loss : ', loss) 

# y_predict = model.predict(x_test)

# r2 = r2_score(y_test, y_predict)
# print('r2 =', r2)