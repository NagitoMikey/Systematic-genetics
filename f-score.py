import pandas as pd
import numpy as np
import operator


def f_score(data):
    df = pd.read_csv(data,header = None)
    df.rename(columns={df.columns[-1]:'label'},inplace=True)
    df0 = df[df.label == 0]
    df1 = df[df.label == 1] #对数据分类
    n = df.shape[0]
    n1 = n/2
    n0 = n/2
    lst =[]
    feature_len = df.shape[1] - 1

    for i in range(feature_len) :
        m0_feature_mean = df0[i].mean()  # 计算负样本在第m维特征上的均值
        m0_SW = sum((df0[i] - m0_feature_mean) ** 2)

        m1_feature_mean = df1[i].mean()  # 计算正样本在第m维特征上的均值
        m1_SW = sum((df1[i] - m1_feature_mean) ** 2)

        m_all_feature_mean = df[i].mean()  # 所有样本在第m维上的均值

        m0_SB = n0 / n * (m0_feature_mean - m_all_feature_mean) ** 2
        m1_SB = n1 / n * (m1_feature_mean - m_all_feature_mean) ** 2

        m_SB = m1_SB + m0_SB
        m_SW = (m0_SW + m1_SW) / n

        if m_SW == 0:

            m_f_score = 0
        else:
            # 计算F-score
            m_f_score = m_SB / m_SW
        lst.append(m_f_score)

    dic = dict(zip(range(len(lst)),lst))
    return dic

data = 'E388.csv' #读取样本文件
print(f_score(data))


sorted_tup = sorted((f_score(data)).items(), key=lambda x:x[1], reverse=True) #对f-score降序排列
print(sorted_tup)
f_lst = []
for i in range(100):
    #取排名前100的特征
    f_lst.append(sorted_tup[i][0])
f_lst.append(864) #加入标签列
f_lst = sorted(f_lst) #生序排列
print(f_lst)

df = pd.read_csv('E388.csv',header = None)
df_final = df[f_lst] #从原来数据框中提取筛选出的特征列
print(df_final)
pd.DataFrame(df_final).to_csv('E388_2.csv',header=None,index=False)



