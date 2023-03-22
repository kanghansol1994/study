import numpy as np

dataset = np.array(range(1, 11))
timesteps = 5

def split_x(dataset, timesteps):
    aaa = []
    for i in range(len(dataset) - timesteps + 1):
        subset = dataset[i : (i + timesteps)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(dataset, timesteps)
print(bbb)
print(bbb.shape) #(6, 5)

# [[ 1  2  3  4  5]
#  [ 2  3  4  5  6]
#  [ 3  4  5  6  7]
#  [ 4  5  6  7  8]
#  [ 5  6  7  8  9]
#  [ 6  7  8  9 10]]
# (6, 5)

# x = bbb[:, :4]
x = bbb[:, :-1]
y = bbb[:, -1]
print(x)
print(y)

# [[1 2 3 4]
#  [2 3 4 5]
#  [3 4 5 6]
#  [4 5 6 7]
#  [5 6 7 8]
#  [6 7 8 9]]
# [ 5  6  7  8  9 10]

#######################################
# split_x라는 함수를 만들어줌
# aaa라는 리스트 만들어줌
# len dataset 데이터셋의 길이(10) - timesteps =5  +1 for i in range 6번 반복하겠음
# subset이라는 곳에 dataset[0 : 0+5] 0번째 부터 5번째 까지 데이터셋이 들어감
# aaa라는 리스트에 subset 데이터셋인 [1,2,3,4,5] 들어감 1번 반복시 
# 반환되어 i가 1 subset이라는 곳에 dataset[1 : 1+5] 1번째 부터 5번째 까지 데이터셋이 들어감
# aaa라는 리스트에 subset 데이터셋인 [2,3,4,5,6] 들어감 2번 반복시 
# 6번 반복시
# bbb = [1,2,3,4,5]
#          [2,3,4,5,6]
#          [3,4,5,6,7]
#          [4,5,6,7,8]
#          [5,6,7,8,9]
#          [6,7,8,9,10]
# 이 됨
# 그래서  bbb.shape는 6행 5열인 2차원 데이터가 만들어짐
#i in range 가 행의 수 timesteps가 열이 됨