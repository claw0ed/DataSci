# DL02
# iris 데이터 집합으로 폼종 구분을 딥러닝으로 구현
# 꽃잎의 모양과 길이로 여러가지 품종으로 나뉨
# 앞서 푼 폐암환자 생존율(0/1) 계산과 달리
# 품종에 대한 범주값이 3가지임 - 다중분류

import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sbs
import matplotlib.pyplot as plt

# 아이리스 데이터 읽기
df = pd.read_csv('data/iris.csv', names=['sepal_length',
                                         'sepal_width','petal_length','petal_width',
                                         'species'])
print(df.head())

# 아이리스 데이터 산점도 확인
sbs.pairplot(df, hue='species')
plt.show()

