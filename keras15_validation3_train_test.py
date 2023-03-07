

import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split 

x_train = np.array(range(1, 17))
y_train = np.array(range(1, 17))

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=13/16, random_state=1234, shuffle=False)     
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=10/13, random_state=1234, shuffle=False)     

print(x_train, x_test, x_val)
print(x_train, x_test, x_val)


################  < 작업 결과 >  ##################

# ras\keras15_validation3_train_test.py' 
# [ 1  2  3  4  5  6  7  8  9 10] [11 12 13] [14 15 16]
# [ 1  2  3  4  5  6  7  8  9 10] [11 12 13] [14 15 16]
#
#