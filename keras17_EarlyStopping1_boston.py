from sklearn.datasets import load_boston
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets['target']  #데이터셋에 있는 타겟이라는 콜롬 가져오겠다
print(x.shape, y.shape) #(506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123, test_size=0.2
)

#2. 모델 구성
model=Sequential()
model.add(Dense(20, activation='relu', input_dim=13))
model.add(Dense(40, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='linear'))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=20, mode='min',
              verbose=1,
              restore_best_weights=True, #디폴트는 false
              ) 

hist= model.fit(x_train, y_train, epochs=2000, batch_size=12,
                validation_split=0.2,
                verbose=1,
                callbacks=[es],
                )
'''
print("***********************************")
print(hist) 
#<tensorflow.python.keras.callbacks.History object at 0x000001C5335864C0>
print("***********************************")
print(hist.history)
print("***********************************")
print(hist.history['loss'])
print("*************발로스**********************")
print(hist.history['val_loss'])
print("*************발로스**********************")
'''
#4. 평가, 예측
loss=model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2스코어: ', r2)
#r2스코어:0.7485
'''
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(9, 6))
plt.plot(hist.history['loss'], marker='.', c='red', label='로스')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='발_로스')
plt.title('보스턴')
plt.xlabel('epochs')
plt.ylabel('loss, val_loss')
plt.legend()
plt.grid()
plt.show()
'''
#문제점: 최적의 가중치가 저장 되는것이 아니라 마지막loss값이 가중치로 저장됨