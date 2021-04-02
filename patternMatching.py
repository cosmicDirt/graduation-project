import numpy as np
import pandas as pd
import flexmatcher

f1=open('../dataset/推荐系统数据样例/智能EPG补充元数据.csv')
f2=open('../dataset/推荐系统数据样例/优酷爱奇艺腾讯元数据.csv')
df1=pd.read_csv(f1,dtype=object)
df2=pd.read_csv(f2,dtype=object)
df3=pd.read_csv('../dataset/douban.csv',dtype=object)

df3_mapping= {'ID':'ID',
              'NAME':'NAME',
              'SCORE':'SCORE',
              'SCORE_NUM':'SCORE_NUM',
              'DIRECTORS':'DIRECTORS',
              'ACTORS':'ACTORS',
              'YEAR':'YEAR',
              'REGION':'REGION',
              'TAGS':'TAGS',
              'DESCRIPTION':'DESCRIPTION'
              }
print(df3.dtypes)
schema_list = [df3]
mapping_list = [df3_mapping]
fm = flexmatcher.FlexMatcher(schema_list, mapping_list, sample_size=500)
fm.train()
predicted_mapping = fm.make_prediction(df1)
print(predicted_mapping)