#头文件导入
import numpy as np
import pandas as pd
#导入数据集
df=pd.read_csv('ecommerce_data.csv')
df['constant']=1
num=df.iloc[0,1]
print(num)
