import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datawig
from sklearn.experimental import enable_iterative_imputer
from sklearn.datasets import fetch_california_housing
from sklearn.impute import SimpleImputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from fancyimpute import KNN, NuclearNormMinimization, SoftImpute, BiScaler, SimpleFill


pathAB = '../dataset/metadata/ABfusion.csv'
ABfusion=pd.read_csv(pathAB)

# 目标列
index=2
# 缺失值间隔
interval=6

cols_numerical=['DURATION','CHAPTER','DOUBAN_SCORE','STATUS','YEAR']
ABfusion.loc[ABfusion['DOUBAN_SCORE']==0,'DOUBAN_SCORE']=np.nan
df_numerical = pd.DataFrame(ABfusion[ABfusion['DOUBAN_SCORE'].notnull()],columns = cols_numerical)
# df_numerical=(df_numerical-df_numerical.min(axis=0))/(df_numerical.max(axis=0)-df_numerical.min(axis=0))

X_complete=df_numerical.copy().values
X_incomplete=X_complete.copy()
X_incomplete[::interval,index]=np.nan


# 均值填充
meanFill = SimpleFill("mean")
X_filled_mean = meanFill.fit_transform(X_incomplete)
# 中位数填充
medianFill = SimpleFill("median")
X_filled_median = medianFill.fit_transform(X_incomplete)
# KNN填充
X_filled_knn = KNN(k=15).fit_transform(X_incomplete)

# Instead of solving the nuclear norm objective directly, instead
# induce sparsity using singular value thresholding
softImpute = SoftImpute()
# simultaneously normalizes the rows and columns of your observed data,
# sometimes useful for low-rank imputation methods
biscaler = BiScaler()
X_incomplete_normalized = biscaler.fit_transform(X_incomplete)
X_filled_softimpute_normalized = softImpute.fit_transform(X_incomplete_normalized)
X_filled_softimpute = biscaler.inverse_transform(X_filled_softimpute_normalized)
X_filled_softimpute_no_biscale = softImpute.fit_transform(X_incomplete)

def evalNumerical(X_filled):
    mae = (abs(X_filled[::interval, index] - X_complete[::interval, index])).mean()
    mse = ((X_filled[::interval,index] - X_complete[::interval,index]) ** 2).mean()
    mape=(abs(X_filled[::interval,index] - X_complete[::interval,index])/X_complete[::interval,index]).mean()
    #r2=1-((X_filled[::interval,index] - X_complete[::interval,index]) ** 2)\
    #   /(X_complete[::interval,index].mean()-X_complete[::interval,index]) ** 2
    r2=r2_score(X_complete[::interval,index],X_filled[::interval,index])
    print("MAE: %f" % mae)
    print("MSE: %f" % mse)
    print("MAPE: %f" % mape)
    print("R2: %f" % r2)
    return mae,mse,mape,r2

print('MEAN:')
evalNumerical(X_filled_mean)
print('MEDIAN:')
evalNumerical(X_filled_median)
print('SOFT:')
evalNumerical(X_filled_softimpute_no_biscale)
print('KNN:')
evalNumerical(X_filled_knn)
# medianfill_mse = ((X_filled_median[::interval,index] - X_complete[::interval,index]) ** 2).mean()
# print("medianFill MSE: %f" % medianfill_mse)
#
# softImpute_mse = ((X_filled_softimpute[::interval,index] - X_complete[::interval,index]) ** 2).mean()
# print("SoftImpute MSE: %f" % softImpute_mse)
# softImpute_no_biscale_mse = (
#     (X_filled_softimpute_no_biscale[::interval,index] - X_complete[::interval,index]) ** 2).mean()
# print("SoftImpute without BiScale MSE: %f" % softImpute_no_biscale_mse)
#
# knn_mse = ((X_filled_knn[::interval,index] - X_complete[::interval,index]) ** 2).mean()
# print("knnImpute MSE: %f" % knn_mse)


# # 深度神经网络填充字符属性(n-gram分词后哈希向量化)
# df_train1, df_test1 = datawig.utils.random_split(ABfusion[ABfusion['LANGUAGES'].notnull()])
#
# #初始化一个简单的imputer模型
# imputer1 = datawig.SimpleImputer(
#     input_columns=['ACTORS','DIRECTORS','NAME','TAGS'], # 我们要输入的列
#     output_column= 'LANGUAGES', # 我们要为其注入值的列
#     output_path = 'imputer_model' #存储模型数据和度量
#     )
#
# #拟合训练数据的模型
# imputer1.fit(train_df=df_train1, num_epochs=10)
#
# #输入丢失的值并返回原始的数据模型和预测
# imputed1 = imputer1.predict(df_test1)




# 深度神经网络填充数值属性
df_train2, df_test2 = datawig.utils.random_split(ABfusion[ABfusion['DOUBAN_SCORE'].notnull()])

#Initialize a SimpleImputer model
imputer2 = datawig.SimpleImputer(
    input_columns=['ACTORS','DIRECTORS','NAME','TAGS','LANGUAGES','REGIONS'], # column(s) containing information about the column we want to impute
    output_column='DOUBAN_SCORE', # the column we'd like to impute values for
    output_path = 'imputer_model' # stores model data and metrics
    )

#Fit an imputer model on the train data
imputer2.fit(train_df=df_train2, num_epochs=10)

#Impute missing values and return original dataframe with predictions
imputed2 = imputer2.predict(df_test2)
mse=mean_squared_error(imputed2['DOUBAN_SCORE'].values,imputed2['DOUBAN_SCORE_imputed'].values)
print(mse)