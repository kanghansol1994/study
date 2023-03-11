import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import accuracy_score
import pandas as pd
#1. 데이터
datasets = load_iris()
print(datasets.DESCR) #판다스 describe()
print(datasets.feature_names) #판다스 columns
#['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

x = datasets.data
y = datasets['target']
print(x.shape, y.shape) #(150, 4) (150, )
print(x)
print(y)
print('y의 라벨값 : ', np.unique(y))  #y의 라벨값 :  [0 1 2]
y= pd.get_dummies(y)
print(y.shape)
#=====================================정리==================
#위와 같이 코드가 매우 간결하여 사용하기가 쉬움
#카테고리가 매우 적을때 효과적 카테고리가 많아질수록 더 많은 메모리와 연산시간 필요
#데이터프레임 형태로 출력되므로 데이터프레임 형태로 데이터 유지 가능