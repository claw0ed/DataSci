# 결측치
import numpy as np
import pandas as pd

# 난수를 이용해서 5x3 데이터프레임 생성
df = pd.DataFrame(np.random.rand(5,3),
                  columns = ['col1','col2','col3'])
# print(df)

# 결측치 삽입
df.ix[0,0] = None
df.ix[1,['col1', 'col3']] = np.nan
df.ix[2, 'col2'] = np.nan
df.ix[3, 'col2'] = np.nan
df.ix[4, 'col2'] = np.nan

# print(df)

# 결측치 해결1 - fillna를 이용 : 0으로 채움
df0 = df.fillna(0)
# print(df0)

# 결측치 해결2 - fillna를 이용 : 문자열로 채움
dfs = df.fillna('결측')
# dfs = df.fillna('miss')
# print(dfs)

# 결측치 해결3 - fillna를 이용 : 평균값으로 채움
df_mean = df.mean()
dfm = df.fillna(df_mean)
# print(dfm)

# NCS 05 PDF p71
hdr = ['V1','V2','V3','V4','V5','V6','V7','V8','V9']
df = pd.read_csv('c:/Java/phone-02.csv', header=None, names=hdr)
# print(df)

# 결측치 삽입
df.ix[1, 'V7'] = np.nan
df.ix[3, 'V7'] = np.nan
df.ix[16, 'V7'] = np.nan
# print(df)

# 결측치 해결1 - fillna를 이용 : 0으로 채움
# df0 = df.fillna(0)
# print(df0)

# 결측치 해결2 - fillna를 이용 : 문자열로 채움
# dfs = df.fillna('결측')
# dfs = df.fillna('miss')
# print(dfs)

# 결측치 해결3 - fillna를 이용 : 평균값으로 채움
df_mean = df.mean()
dfm = df.fillna(df_mean)
print(dfm)

