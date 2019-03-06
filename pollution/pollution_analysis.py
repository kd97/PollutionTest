import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis
from pandas import DataFrame
from sklearn.cluster import KMeans
import seaborn as sns
from numpy import eye, asarray, dot, sum, diag
from numpy.linalg import svd

factor = ["factor1", "factor2", "factor3"]
standard = ["CODMn", "NH3-N", "KMnO4", "TP", "PH", "DO"]
#算法实现

#因子降维
def factor_dim(df):
    #主成份分析
    pmodel = PCA(n_components=3)
    lower_mat = pmodel.fit_transform(df)
    df_array   = df.values[:]
    lower_df = DataFrame(lower_mat,columns=["factor1","factor2","factor3"])
    #因子分析
    fmodel =FactorAnalysis (n_components=3,random_state=0)
    lower_fac = fmodel.fit_transform(df)
    #lower_df = DataFrame(lower_fac,columns=["factor1","factor1","factor1"])
    print(lower_df)
    return lower_df

#相关系数求解
def factor_cor(df,lower_df):
    result = []
    for i in range(len(standard)):
        op = []
        for j in range(len(factor)):
            op.append(lower_df[factor[j]].corr(df[standard[i]]))
        result.append(op)
    result = np.array(result)
    print(result)
    return result

#因子旋转
def varimax(Phi, gamma = 1.0, q =5, tol = 1e-6): #定义方差最大旋转函数
    p,k = Phi.shape #给出矩阵Phi的总行数，总列数
    R = eye(k) #给定一个k*k的单位矩阵
    d=0
    for i in range(q):
        d_old = d
        Lambda = dot(Phi, R)#矩阵乘法
        u,s,vh = svd(dot(Phi.T,asarray(Lambda)**3 - (gamma/p) * dot(Lambda, diag(diag(dot(Lambda.T,Lambda)))))) #奇异值分解svd

        R = dot(u,vh)#构造正交矩阵R

        d = sum(s)#奇异值求和

        if d_old!=0 and d/d_old:
            return dot(Phi, R)#返回旋转矩阵Phi*R

#筛选污染因子
def factor_select(result):
    rotation_mat=varimax(result)
    print(rotation_mat)
    answer = []
    for i in range(len(rotation_mat[:,0])):
        if(rotation_mat[i][0]>0.5):
            answer.append(standard[i])
    print(answer)
    print(df[answer])
    return answer

#聚类分析
def cluster_analyse(ep):
    kmeans =  KMeans(n_clusters=3,random_state=0).fit(ep)
    print(kmeans.labels_)
    col1=ep_raw.loc[:,"Name"]
    col2 = kmeans.labels_
    kmeans_result=pd.DataFrame([col1,col2]).T
    kmeans_result.columns = ["Name","Cluster"]
    print(kmeans_result)
    return np.array(kmeans_result["Cluster"])


#成对比较矩阵
def comparision(W0):  # W为每个信息值的权重
    n=len(W0)
    F=np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            if i==j:
                F[i,j]=1
            else:
                F[i,j]=W0[i]/W0[j]
    return F

#一致性检验
def isConsist(F):
    n=np.shape(F)[0]
    a,b=np.linalg.eig(F)
    maxlam=a[0].real
    CI=(maxlam-n)/(n-1)
    if CI<0.1:
        return bool(1)
    else:
        return bool(0)

#单层排序,相对重要度
def ReImpo(F):
    n=np.shape(F)[0]
    W=np.zeros([1,n])
    for i in range(n):
        t=1
        for j in range(n):
            t=F[i,j]*t
        W[0,i]=t**(1/n)
    W=W/sum(W[0,:])  # 归一化 W=[0.874,2.467,0.464]
    return W.T


#层次分析矩阵构建
def layer_matrix(kmeans_result):
    # 定义目标层 距离 排水量 事故系数权重 F12
    F12 = np.array([[1, 8, 8], [1 / 8, 1, 2], [1 / 8, 1 / 2, 1]])
    cluster_cmp = 2 ** kmeans_result
    # 定义准则层
    # 聚类污染因子准则
    F231 = comparision(cluster_cmp)
    # 事故系数准则
    F232 = comparision(ep_raw.loc[:, "Event"])
    # 排水量准则
    F233 = comparision(ep_raw.loc[:, "Qual"])
    if isConsist(F12) and isConsist(F231) and isConsist(F232) and isConsist(F233):
        W12 = ReImpo(F12)
        print(W12)
        W231 = ReImpo(F231)
        W232 = ReImpo(F232)
        W233 = ReImpo(F233)
        W23 = np.hstack([W231, W232, W233])
        print(W23)
    else:
        print("成对比较矩阵不一致，请调整权重后重试！")
        return 0
    return [W12,W23]

#层次分析计算
def layer_computer(W12,W23):
    n = len(W23)
    C = np.zeros([1, n])
    for i in range(n):
        t = W23[i, :]
        C[0, i] = sum((W12.T * t)[0])
        print(C[0, i])
    print('最佳方案为第', np.argmax(C) + 1, '个方案.', '综合推荐指数为', max(C[0, :]))
    return C

if __name__ == '__main__':
    df = pd.read_csv('./section.csv')
    # print(df)
    # TODO 数据预处理
    #因子分析
    lower_df = factor_dim(df)  #因子分析降维
    result = factor_cor(df,lower_df)    #相关性载荷函数
    answer = factor_select(result)   #筛选因子
    #聚类分析
    ep_raw = pd.read_csv('./enterprise.csv')
    ep = ep_raw.loc[:, answer]
    kmeans_result = cluster_analyse(ep)   #聚类分析
    #层次分析
    com = layer_matrix(kmeans_result)  #构建矩阵
    layer_result = layer_computer(com[0],com[1])     #加权计算
    print(layer_result)

