import py_entitymatching as em
import pandas as pd

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)

pathA = '../dataset/metadata/爱奇艺优酷腾讯元数据.csv'
pathB = '../dataset/metadata/智能EPG补充元数据.csv'
pathC = '../dataset/metadata/同洲媒资库元数据.csv'
pathpAB = '../dataset/metadata/predictionAB.csv'
pathAB = '../dataset/metadata/AB.csv'

A = em.read_csv_metadata(pathA, key='ID', dtype=str)
B = em.read_csv_metadata(pathB, key='ID', dtype=str)
C = em.read_csv_metadata(pathC, key='ID', dtype=str)

# # 读取_id、ltable_ID、rtable_ID、predicted（匹配为1，否则为0）四列
# predictionAB = em.read_csv_metadata(pathpAB, key='_id', dtype=str, usecols=[1, 2, 3, 58])
# AB = pd.concat([A, B])
# # 分组融合
# AB['group'] = 0
# AB.fillna('', inplace=True)
#
# i = 1
# for row in predictionAB.itertuples():
#     if (row.predicted == '1'):
#         # 只有ltable已有匹配分组，将rtable加入该分组
#         if (AB.loc[AB['ID'] == row.ltable_ID, 'group'].values[0] != 0 and
#                 AB.loc[AB['ID'] == row.rtable_ID, 'group'].values[0] == 0):
#             AB.loc[AB['ID'] == row.rtable_ID, 'group'] = AB.loc[AB['ID'] == row.ltable_ID, 'group'].values[0]
#         # 只有rtable已有匹配分组，将ltable加入该分组
#         if (AB.loc[AB['ID'] == row.ltable_ID, 'group'].values[0] == 0 and
#                 AB.loc[AB['ID'] == row.rtable_ID, 'group'].values[0] != 0):
#             AB.loc[AB['ID'] == row.ltable_ID, 'group'] = AB.loc[AB['ID'] == row.rtable_ID, 'group'].values[0]
#         # 都没有分组
#         if (AB.loc[AB['ID'] == row.ltable_ID, 'group'].values[0] == 0 &
#                 AB.loc[AB['ID'] == row.rtable_ID, 'group'].values[0] == 0):
#             AB.loc[AB['ID'] == row.ltable_ID, 'group'] = i
#             AB.loc[AB['ID'] == row.rtable_ID, 'group'] = i
#
#         # 都有分组，认为该次匹配多余，什么都不做
#         else:
#             continue
#         i = i + 1
# AB.to_csv('AB.csv')

AB = em.read_csv_metadata(pathAB, key='ID', dtype=str)
AB.fillna('', inplace=True)
AB0 = AB[AB['group'] == '0']
AB = AB[AB['group'] != '0']


# 定义分组后各属性融合规则
def ABFusion(grouped):
    cols = grouped.columns.tolist()
    # col1中属性融合时 保留分组内最长的属性值
    col1 = ['ACTORS', 'DIRECTORS', 'NAME', 'STORYLINE', 'DURATION','DOUBAN_SCORE','CHAPTER','STATUS']
    # col2中属性融合时 保留分组内出现次数最多的属性值
    col2 = ['COSHIP_FLAG','YEAR','RELEASE_YEAR', 'RELEASE_DATE']
    result = pd.Series({
        'test': ','.join(grouped['ID'])
    })
    for col in cols:
        if (col in col1):
            # 同组字符串仅保留最长的
            result[col] = grouped[col].max()
        elif(col in col2):
            # 同组字符串仅保留最多的
            result[col] = grouped[col].value_counts().index[0] if grouped[col].value_counts().index[0]!='' else grouped[col].max()
        else:
            # 同组字符串用逗号拼接，并去除重复与空串
            result[col] = ','.join(x for x in list(set(grouped[col])) if x != '')
    result.pop('test')
    result.pop('group')
    return result


# grouped = grouped['TAGS'].apply(lambda x: ",".join(list(set(x.str.cat(sep=',').split(','))))).reset_index()

# 分组聚合+拼接
ABfusion = AB.groupby(['group']).apply(ABFusion).reset_index()
ABfusion.to_csv('ABfusion.csv')
