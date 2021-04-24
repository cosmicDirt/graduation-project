import py_entitymatching as em
import os
import pandas as pd
import pandas_profiling as pp

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)

pathA = '../dataset/metadata/爱奇艺优酷腾讯元数据.csv'
pathB = '../dataset/metadata/智能EPG补充元数据.csv'
pathC = '../dataset/metadata/同洲媒资库元数据.csv'

A = em.read_csv_metadata(pathA, key='ID', dtype=str)
B = em.read_csv_metadata(pathB, key='ID', dtype=str)
C = em.read_csv_metadata(pathC, key='ID', dtype=str)

# 从A、B数据集中采样，size为从B表中采样元组数，y_param为sample_B中每个元组所匹配的A表元组数
sample_A, sample_B = em.down_sample(A, B, size=2000, y_param=30, show_progress=False)

# blocking
ob = em.OverlapBlocker()
# 对NAME用N-gram进行分词，q_val为N值，overlap_size为两表在每个token上的最少重叠数
C1 = ob.block_tables(A, B, 'NAME', 'NAME', word_level=False, q_val=3, overlap_size=3,
                     l_output_attrs=['NAME', 'ACTORS', 'DIRECTORS', 'TAGS', 'REGION', 'LANGUAGE', 'RELEASE_YEAR'],
                     r_output_attrs=['NAME', 'ACTORS', 'DIRECTORS', 'TAGS', 'REGIONS', 'LANGUAGES', 'YEAR'],
                     allow_missing=False, show_progress=False)
block_f = em.get_features_for_blocking(sample_A, sample_B, validate_inferred_attr_types=False)
# 对ACTORS及DIRECTORS属性进行相似度blocking
rb = em.RuleBasedBlocker()
rb.add_rule(['ACTORS_ACTORS_jac_qgm_3_qgm_3(ltuple, rtuple) < 0.8'], block_f)
rb.add_rule(['DIRECTORS_DIRECTORS_jac_qgm_3_qgm_3(ltuple, rtuple) < 0.8'], block_f)
C2 = rb.block_candset(C1, show_progress=False)

print(len(sample_A), len(sample_B), len(C1), len(C2))
# print(C2.head(20))
# 从blocking后的数据集采样进行标注
# S = em.sample_table(C1, 300)
C2.to_csv('ABmatching.csv')

A.fillna(' ', inplace=True)
B.fillna(' ', inplace=True)
C.fillna(' ', inplace=True)

# generate features
atypes1 = em.get_attr_types(A)
atypes2 = em.get_attr_types(B)
block_c = em.get_attr_corres(A, B)
block_c['corres'].clear()
block_c['corres'] = [('ACTORS', 'ACTORS'), ('DIRECTORS', 'DIRECTORS'), ('NAME', 'NAME'),
                     ('TAGS', 'TAGS'), ('REGION', 'REGIONS'), ('LANGUAGE', 'LANGUAGES'), ('RELEASE_YEAR', 'YEAR')]
tok = em.get_tokenizers_for_matching()
sim = em.get_sim_funs_for_matching()
feature_table = em.get_features(A, B, atypes1, atypes2, block_c, tok, sim)

# matching
path_labeled_data = '../dataset/metadata/labeldata5.csv'
matching_data = em.read_csv_metadata(path_labeled_data,
                                     key='_id', dtype=str,
                                     ltable=A, rtable=B,
                                     fk_ltable='ltable_ID', fk_rtable='rtable_ID')
matching_data.fillna(' ', inplace=True)
matching_data['label'] = matching_data['label'].astype(int)

train_test = em.split_train_test(matching_data, train_proportion=0.5, random_state=0)
train = train_test['train']
test = train_test['test']

dt = em.DTMatcher(name='DecisionTree', random_state=0)
svm = em.SVMMatcher(name='SVM', random_state=0)
nb = em.NBMatcher(name='NaiveBayes')
rf = em.RFMatcher(name='RF', random_state=0)
lg = em.LogRegMatcher(name='LogReg', random_state=0)
ln = em.LinRegMatcher(name='LinReg')

# Convert the train into a set of feature vectors using feature_table
F = em.get_features_for_matching(A, B, validate_inferred_attr_types=False)
feature_vectors_train = em.extract_feature_vecs(train,
                                                feature_table=feature_table,
                                                attrs_after='label',
                                                show_progress=False)
feature_vectors_test = em.extract_feature_vecs(test,
                                               feature_table=feature_table,
                                               attrs_after='label',
                                               show_progress=False)
# feature_vectors = em.impute_table(feature_vectors,
#                 exclude_attrs=['_id', 'ltable_ID', 'rtable_ID', 'label'],
#                 strategy='mean')


# # 利用交叉验证选择最优分类器
# result = em.select_matcher([dt, rf, svm, nb, ln, lg], table=feature_vectors_train,
#         exclude_attrs=['_id', 'ltable_ID', 'rtable_ID', 'label'],
#         k=5,
#         target_attr='label', metric_to_select_matcher='f1', random_state=0)
# print(result['cv_stats'])

# 训练集
rf.fit(table=feature_vectors_train, exclude_attrs=['_id', 'ltable_ID', 'rtable_ID', 'label'], target_attr='label')
# 测试集
predictions = rf.predict(table=feature_vectors_test, exclude_attrs=['_id', 'ltable_ID', 'rtable_ID', 'label'],
                         append=True, target_attr='predicted', inplace=True, return_probs=True,
                         probs_attr='proba')

# #随机测试集
# labeldata1 = em.read_csv_metadata('../dataset/metadata/labeldata1.csv',
#                          key='_id',dtype=str,
#                          ltable=A, rtable=B,
#                          fk_ltable='ltable_ID', fk_rtable='rtable_ID')
# labeldata1.fillna(' ',inplace=True)
# labeldata1['label']=labeldata1['label'].astype(int)
# feature_vectors_labeldata1 = em.extract_feature_vecs(labeldata1,
#                             feature_table=feature_table,
#                             attrs_after='label',
#                             show_progress=False)
# predictions1 = rf.predict(table=feature_vectors_labeldata1, exclude_attrs=['_id', 'ltable_ID', 'rtable_ID', 'label'],
#               append=True, target_attr='predicted', inplace=False, return_probs=True,
#                         probs_attr='proba')

# # Evaluate the predictions
# eval_result = em.eval_matches(predictions, 'label', 'predicted')
# em.print_eval_summary(eval_result)

# Data fusion
feature_vectors_AB = em.extract_feature_vecs(C2,
                                             feature_table=feature_table,
                                             show_progress=False)
predictionAB = rf.predict(table=feature_vectors_AB, exclude_attrs=['_id', 'ltable_ID', 'rtable_ID'],
                          append=True, target_attr='predicted', inplace=True, return_probs=True,
                          probs_attr='proba')
# feature_vectors_AB.to_csv('predictionAB.csv')
