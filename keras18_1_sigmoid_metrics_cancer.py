import numpy as np
from sklearn.datasets import load_breast_cancer #유방암 데이터
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = load_breast_cancer()
# print(datasets)
# print(datasets.DESCR)      #판다스: .describe()
# print(datasets.feature_names)   #판다스:.columns()

x = datasets['data']
y = datasets.target

print(x.shape, y.shape)  #(569, 30) (569,)
# print(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=1013, test_size=0.2
)

#2. 모델구성
model = Sequential()
model.add(Dense(12, activation='relu', input_dim=30))
model.add(Dense(10, activation='linear'))
model.add(Dense(8, activation='linear'))
model.add(Dense(6, activation='linear'))
model.add(Dense(1, activation='sigmoid')) #0과 1사이로 한정시킨다

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics= ['accuracy', 'mse',] #'mean_squared_error', 'acc']
              )
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='var_loss', patience=500, mode='min',
                   verbose=1,
                   restore_best_weights=True
                   )

model.fit(x_train, y_train, epochs=1000, batch_size=10,
          validation_split=0.2,
          verbose=1,
          callbacks=[es]
          )

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('results : ', results)

y_predict = np.round(model.predict(x_test))
# print("=========================")
# print(y_test [:5])
# print(y_predict[:5])
# print(np.round(y_predict[:5]))
# print("=========================")
from sklearn.metrics import r2_score, accuracy_score
acc = accuracy_score(y_test, y_predict)
print('acc : ', acc) #acc : 0.9035087719298246 
# y_test= [1 0 1 1 0]의 형태 y_predict= [[ 0.45126796]
#[ 0.33138537]] 와 같이 형태가 다르므로 분류가 다르다는 벨류에러 발생 
