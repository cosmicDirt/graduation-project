import numpy as np
import pandas as pd
import flexmatcher

f1=open('../dataset/智能EPG补充元数据.csv')
f2=open('../dataset/爱奇艺优酷腾讯元数据.csv')
f3=open('../dataset/同洲媒资库元数据.csv')

df1=pd.read_csv(f1,dtype=object)
df2=pd.read_csv(f2,dtype=object)
df3=pd.read_csv(f3,dtype=object)
df4=pd.read_csv('../dataset/douban.csv',dtype=object)

df1_mapping={
             'ACTORS':'ACTORS',
             'ALIAS':'ALIAS',
             'ASPECT':'DESCRIPTION_BRIEF',
             'CHAPTER':'CHAPTER',
             'DIRECTORS':'DIRECTORS',
             'DOUBAN_SCORE':'DOUBAN_SCORE',
             'DURATION':'DURATION',
             'IMG':'IMG',
             'LANGUAGES':'LANGUAGES',
             'MD5':'MD5',
             'NAME':'NAME',
             'REGIONS':'REGIONS',
             'RELEASE_DATE':'RELEASE_DATE',
             'SHOW_TYPE':'SHOW_TYPE',
             'STATUS':'STATUS',
             'STORYLINE':'DESCRIPTION_FULL',
             'TAGS':'TAGS',
             'WRITER':'WRITER',
             'YEAR':'YEAR'}
df4_mapping= {
              'NAME':'NAME',
              'SCORE':'DOUBAN_SCORE',
              'SCORE_NUM':'DOUBAN_VOTES',
              'DIRECTORS':'DIRECTORS',
              'ACTORS':'ACTORS',
              'YEAR':'YEAR',
              'REGION':'REGIONS',
              'TAGS':'TAGS',
              'DESCRIPTION':'DESCRIPTION_BRIEF'
              }

schema_list = [df1,df4]
mapping_list = [df1_mapping,df4_mapping]
fm = flexmatcher.FlexMatcher(schema_list, mapping_list, sample_size=200)
fm.train()
predicted_mapping = fm.make_prediction(df2)
print(predicted_mapping)