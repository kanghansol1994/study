import tensorflow as tf
import numpy as np
import pandas as pd
import random
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error

def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))

# 0. seed initialization
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. data prepare
path='./_data/DDarung/'
path_save='./_save/DDarung/'
df=pd.read_csv(path+'train.csv',index_col=0)
df=df.dropna()
print(df.shape)
print(df.isnull().sum())

x=df.drop([df.columns[-1]],axis=1)
y=df[df.columns[-1]]
print(x.shape,y.shape)

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,random_state=seed)

#2. model
model=Sequential()
layer123=[2**6,2**5,2**6]
model.add(Dense(layer123[0],input_dim=9,activation='sigmoid'))
for i in layer123[1:]:
    model.add(Dense(i,activation='relu'))
model.add(Dense(1))

epo=3000
#3. compile ,training
model.compile(loss='mse',optimizer='adam')
model.fit(x,y,batch_size=len(x),epochs=epo,verbose=True,validation_split=0.2)

#4. evaluation,prediction
loss=model.evaluate(x_test,y_test)
print(f'loss : {loss}')
y_predict=model.predict(x_test)
rmse=RMSE(y_test,y_predict)
print(f'rmse : {rmse}')

##### submission.csv를 만들어봅시다!!! #####
test_csv=pd.read_csv(path+'test.csv',index_col=0)
y_submit=model.predict(test_csv)
print(test_csv.isnull().sum())

submission = pd.read_csv(path+'submission.csv',index_col=0)
submission['count'] =y_submit
print(submission)
substr='submission_'
for i in layer123:
    substr+=str(i)
substr+=str(epo)+'.csv'
submission.to_csv(path_save+substr)
#mrmes:45.17439031380902 mlay:6,7,2 mepo:3200