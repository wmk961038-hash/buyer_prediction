#头文件导入
import numpy as np
import pandas as pd
#导入数据集
df=pd.read_csv('ecommerce_data.csv')
df['constant']=1 #增加最后一项，用于处理偏置项

print(df.head())
