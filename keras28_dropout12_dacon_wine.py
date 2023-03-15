from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,LeakyReLU,Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,accuracy_score
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler, LabelEncoder 
from keras.utils import to_categorical

#1. 데이터
path = './_data/dacon_wine/'
path_save ='./_save/dacon_wine/'

train_csv = pd.read_csv(path+'train.csv', index_col=0)
print(train_csv.shape) #(5497, 13)
test_csv = pd.read_csv(path+'test.csv', index_col=0)
print(test_csv.shape) #(1000, 12)
train_csv = train_csv.dropna()


x = train_csv.drop(['quality'], axis=1)
y = train_csv['quality']
print(x.shape, y.shape) #(5497, 11) (5497, )
print('y의 라벨값 : ', np.unique(y)) #[3,4,5,6,7,8,9]

y= pd.get_dummies(y)
y=np.array(y)
print(y.shape) #(5497,7)

# y = to_categorical(y)
# print(y.shape)
x_train, x_test, y_train, y_test = train_test_split(x, y,
    test_size=0.2,
    shuffle=True,
    stratify=y,
    random_state=335)

le = LabelEncoder()
le.fit(train_csv['type'])
aaa = le.transform(train_csv['type'])

train_csv['type'] = aaa
print(train_csv)
test_csv['type'] = le.transform(test_csv['type'])

print(le.transform(['red', 'white']))   #[0 1] 
print(le.transform(['white', 'red']))   #[1 0] 

le = LabelEncoder()
x_train['type'] = le.fit_transform(x_train['type'])
x_test['type'] = le.transform(x_test['type'])

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델구성
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=12))
model.add(Dropout(0.1))
model.add(Dense(96, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(7, activation='softmax'))

# model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=50, mode='min',
                    verbose=1,
                    restore_best_weights=True,
)

hist= model.fit(x_train, y_train, epochs=2000, batch_size=100,
          validation_split=0.2,
          verbose=1,
          callbacks=[es],
          )


#4. 평가, 예측
results=model.evaluate(x_test, y_test)
print('results : ', results)
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=-1)
y_test = np.argmax(y_test, axis=-1)

acc = accuracy_score(y_test, y_predict)
print('acc : ', acc)

#4-1 내보내기
submission = pd.read_csv(path+'sample_submission.csv', index_col=0)
y_submit=model.predict(test_csv)

y_submit=np.argmax(y_submit, axis=1)
y_submit += 3
submission['quality'] = y_submit
submission.to_csv(path_save+ 'submit_0316_0523.csv')



#seed:123
# model = Sequential()
# model.add(Dense(24, activation='relu', input_dim=11))
# model.add(Dense(12, activation='relu'))
# model.add(Dense(24, activation='relu'))
# model.add(Dense(12, activation='relu'))
# model.add(Dense(24, activation='relu'))
# model.add(Dense(7, activation='softmax'))
# epochs=2000, batch_size=100, patience=200
# [1.0394725799560547, 0.5454545617103577]
# 데이콘:0.561

#seed:141
# [1.0135375261306763, 0.5727272629737854]
# 데이콘:0.563

#seed:169
# results :  [1.047060251235962, 0.550000011920929]
# 데이콘:0.55

#seed:172
# results :  [1.0527664422988892, 0.5581818222999573]
# 데이콘:0.568

#seed:204
# results :  [1.0137754678726196, 0.5754545331001282]
# 데이콘:0.552

#seed:291
# results :  [1.0357983112335205, 0.5563636422157288]
# 데이콘:0.566

# model = Sequential()
# model.add(Dense(24, activation='relu', input_dim=11))
# model.add(Dropout(0.1))
# model.add(Dense(12, activation='relu'))
# model.add(Dense(24, activation='relu'))
# model.add(Dropout(0.4))
# model.add(Dense(12, activation='relu'))
# model.add(Dense(24, activation='relu'))
# model.add(Dense(7, activation='softmax'))
# results :  [1.031816840171814, 0.5618181824684143]
# 데이콘:0.584

# model = Sequential()
# model.add(Dense(24, activation=LeakyReLU(0.4), input_dim=11))
# model.add(Dropout(0.1))
# model.add(Dense(12, activation=LeakyReLU(0.4)))
# model.add(Dense(24, activation=LeakyReLU(0.4)))
# model.add(Dropout(0.4))
# model.add(Dense(12, activation=LeakyReLU(0.4)))
# model.add(Dense(24, activation=LeakyReLU(0.4)))
# model.add(Dropout(0.2))
# model.add(Dense(7, activation='softmax'))
# results :  [1.0164812803268433, 0.5609090924263]
# 데이콘:0.572

# model = Sequential()
# model.add(Dense(48, activation='relu', input_dim=11))
# model.add(Dropout(0.1))
# model.add(Dense(24, activation='relu'))
# model.add(Dense(48, activation='relu'))
# model.add(Dropout(0.4))
# model.add(Dense(24, activation='relu'))
# model.add(Dense(48, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(7, activation='softmax'))
#patience=100
# results :  [1.0401531457901, 0.5554545521736145]
# 데이콘:0.564

# model = Sequential()
# model.add(Dense(128, activation='relu', input_dim=11))
# model.add(Dropout(0.1))
# model.add(Dense(96, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.4))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(7, activation='softmax'))
# patience=50
# results :  [1.026586890220642, 0.5590909123420715]
# 데이콘:0.579

#seed:293
# results :  [0.9833370447158813, 0.5872727036476135]
# 데이콘:0.576

#seed:294
# results :  [1.0071685314178467, 0.5718181729316711]
# 데이콘:0.576

#seed:295
# results :  [0.9984596967697144, 0.581818163394928]
#데이콘:0.572

#seed:304
# results :  [1.0008772611618042, 0.5709090828895569]
#데이콘:0.559

#seed:307
# results :  [1.0019313097000122, 0.5899999737739563]
#데이콘:0.575

#seed:315
# results :  [1.0197746753692627, 0.5772727131843567]
# 데이콘:0.573

#seed:325
# results :  [0.9789302349090576, 0.581818163394928]
# 데이콘:0.586

#seed:329
# results :  [1.0095678567886353, 0.5681818127632141]
# 데이콘:0.581

#seed:330
# results :  [1.0138932466506958, 0.5827272534370422]
# 데이콘:0.572

#seed:335
# results :  [0.988271951675415, 0.5809090733528137]
# 데이콘:0.589

#LabelEncoder 추가
#results :  [0.9894194006919861, 0.5899999737739563]
#데이콘:

# input_dim 제대로 넣음
# results :  [1.0054124593734741, 0.5863636136054993]
# 데이콘: