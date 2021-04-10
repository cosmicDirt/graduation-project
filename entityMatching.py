import py_entitymatching as em
import os
import pandas as pd
import pandas_profiling as pp

#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)

pathA='../dataset/metadata/智能EPG补充元数据.csv'
pathB='../dataset/metadata/爱奇艺优酷腾讯元数据.csv'
pathC='../dataset/metadata/同洲媒资库元数据.csv'


A=em.read_csv_metadata(pathA,key='ID',dtype=str)
B=em.read_csv_metadata(pathB,key='ID',dtype=str)

#从A、B数据集中采样，size为从B表中采样元组数，y_param为sample_B中每个元组所匹配的A表元组数
sample_A, sample_B = em.down_sample(A, B, size=1000, y_param=10, show_progress=False)

#blocking
ob = em.OverlapBlocker()
#对NAME用N-gram进行分词，q_val为N值，overlap_size为两表在每个token上的最少重叠数
C1 = ob.block_tables(sample_A, sample_B, 'NAME', 'NAME', word_level=False, q_val=3, overlap_size=4,
                    l_output_attrs=['NAME','ACTORS','DIRECTORS' ],
                    r_output_attrs=['NAME', 'ACTORS','DIRECTORS'],
                    allow_missing=False,show_progress=False)
block_f = em.get_features_for_blocking(sample_A, sample_B, validate_inferred_attr_types=False)
#对ACTORS及DIRECTORS属性进行相似度blocking
rb = em.RuleBasedBlocker()
rb.add_rule(['ACTORS_ACTORS_lev_sim(ltuple, rtuple) < 0.8'], block_f)
rb.add_rule(['DIRECTORS_DIRECTORS_jac_qgm_3_qgm_3(ltuple, rtuple) < 0.8'], block_f)
C2 = rb.block_candset(C1, show_progress=False)

#generate features
atypes1 = em.get_attr_types(sample_A)
atypes2 = em.get_attr_types(sample_B)
block_c = em.get_attr_corres(sample_A,sample_B)
block_c['corres'].clear()
block_c['corres']=[('ACTORS','ACTORS'),('DIRECTORS','DIRECTORS'),('NAME','NAME'),
                   ('TAGS','TAGS'),('REGIONS','REGION'),('LANGUAGES','LANGUAGE')]
tok = em.get_tokenizers_for_blocking()
sim = em.get_sim_funs_for_blocking()
feature_table = em.get_features(sample_A, sample_B, atypes1, atypes2, block_c, tok, sim)

print(len(sample_A),len(sample_B),len(C1),len(C2))
print(C1.head())
#从blocking后的数据集采样进行标注
# S = em.sample_table(C2, 150)
# S.to_csv('labeldata.csv')