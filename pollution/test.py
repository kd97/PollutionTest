import numpy as np #导入numpy库
import pandas as pd #导入pandas库
from math import sqrt #导入求根函数sqrt
from numpy import eye, asarray, dot, sum, diag #导入eye,asarray,dot,sum,diag 函数

from numpy.linalg import svd #导入奇异值分解函数
import numpy.linalg as nlg
X=pd.read_csv('./section.csv')
X1=(X-X.mean())/X.std() # 0均值规范化
C=X1.corr() #相关系数矩阵
print(C)

#导入nlg函数，linalg=linear+algebra
eig_value,eig_vector=nlg.eig(C) #计算特征值和特征向量
eig=pd.DataFrame() #利用变量名和特征值建立一个数据框
eig['names']=X.columns#列名
eig['eig_value']=eig_value#特征值
print(eig)

col0=list(sqrt(eig_value[0])*eig_vector[:,0]) #因子载荷矩阵第1列

col1=list(sqrt(eig_value[1])*eig_vector[:,1]) #因子载荷矩阵第2列

col2=list(sqrt(eig_value[2])*eig_vector[:,2]) #因子载荷矩阵第3列

col3=list(sqrt(eig_value[3])*eig_vector[:,3]) #因子载荷矩阵第4列
A=pd.DataFrame([col0,col1,col2,col3]).T
print(A)

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
rotation_mat=varimax(A)
print(rotation_mat)