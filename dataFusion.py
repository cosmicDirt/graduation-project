import py_entitymatching as em
import pandas as pd

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)

pathA = '../dataset/metadata/爱奇艺优酷腾讯元数据.csv'
pathB = '../dataset/metadata/智能EPG补充元数据.csv'
pathC = '../dataset/metadata/同洲媒资库元数据.csv'
pathAB = '../dataset/metadata/predictionAB.csv'

A = em.read_csv_metadata(pathA, key='ID', dtype=str)
B = em.read_csv_metadata(pathB, key='ID', dtype=str)
C = em.read_csv_metadata(pathC, key='ID', dtype=str)
# 读取_id、ltable_ID、rtable_ID、predicted（匹配为1，否则为0）四列
predictionAB = em.read_csv_metadata(pathAB, key='_id', dtype=str, usecols=[1, 2, 3, 58])
AB = pd.concat([A, B])
# 分组融合
AB['group'] = 0
AB.fillna('', inplace=True)

i = 1
for row in predictionAB.itertuples():
    if (row.predicted == '1'):
        # 只有ltable已有匹配分组，将rtable加入该分组
        if (AB.loc[AB['ID'] == row.ltable_ID, 'group'].values[0] != 0 and
                AB.loc[AB['ID'] == row.rtable_ID, 'group'].values[0] == 0):
            AB.loc[AB['ID'] == row.rtable_ID, 'group'] = AB.loc[AB['ID'] == row.ltable_ID, 'group'].values[0]
        # 只有rtable已有匹配分组，将ltable加入该分组
        if (AB.loc[AB['ID'] == row.ltable_ID, 'group'].values[0] == 0 and
                AB.loc[AB['ID'] == row.rtable_ID, 'group'].values[0] != 0):
            AB.loc[AB['ID'] == row.ltable_ID, 'group'] = AB.loc[AB['ID'] == row.rtable_ID, 'group'].values[0]
        # 都没有分组
        if (AB.loc[AB['ID'] == row.ltable_ID, 'group'].values[0] == 0 &
                AB.loc[AB['ID'] == row.rtable_ID, 'group'].values[0] == 0):
            AB.loc[AB['ID'] == row.ltable_ID, 'group'] = i
            AB.loc[AB['ID'] == row.rtable_ID, 'group'] = i

        # 都有分组，认为该次匹配多余，什么都不做
        else:
            continue
        i = i + 1
# AB.to_csv('AB.csv')
groupedAB = AB.groupby('group')
