""" # 生成随机矩阵
import numpy as np

#　设置随机种子，保证每次生成的随机数一样，可以不设置（去除下面一行代码，将所有的 rd 替换成 np.random 即可）
rd = np.random.RandomState(888) 

# 随机整数
S = rd.randint(-2, 3, (100, 100)) # 随机生成[-2,3)的整数，10x10的矩阵
# matrix = rd.randint(-2, 3, [10, 10]) # 效果同上
# print(matrix)

# 随机浮点数
X = rd.random((100, 20)) # 随机生成一个 [0,1) 的浮点数 ，5x5的矩阵
# print(matrix1)

K = np.dot(S, X)
print(K.shape) """
import numpy as np
#from learn_SOC_TRUE import learnSOCmodel
from learnSOC_control import learnSOCmodel_withControl
import torch
import numpy as np
from scipy.linalg import pinv, norm, polar
from scipy.linalg import solve
from scipy.sparse.linalg import eigs
from scipy.linalg import sqrtm
from scipy.linalg import solve_discrete_lyapunov as dlyap
import math
import time
import sys
import torch
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
# A = np.array([[1, 2], [3, 4]])
# B = np.array([[2, 3], [4, 5]])

# # result = np.linalg.lstsq(A, B, rcond=None)[0]
# result = A/B
# print(result)
# import numpy as np

# XU = np.array([[1, 2], [3, 4]])
# a = np.array([[2, 3], [4, 5]])

# result = np.linalg.lstsq(a.T, XU.T, rcond=None)[0].T
# print(result)


# options = {}
# options["graphic"] = 0
# options["posdef"] = 10e-12
# options["maxiter"] = 1000
# X = torch.DoubleTensor(10.2*np.eye(5))
# Y = torch.DoubleTensor(4.5*np.eye(5))
# U = torch.DoubleTensor(7*np.eye(5))
# U0 = 0
# A_SOC, B_SOC, err, _ = learnSOCmodel_withControl(X, Y, U, U0, options)
# A_SOC, _ = learnSOCmodel(X, Y, options)
# print(A_SOC)
# print(B_SOC)

A = np.array([[1.2],[4.5],[2.9]])
B = np.array([[1,2,3]])
A = torch.DoubleTensor(A)
B = torch.DoubleTensor(B)
lB = nn.Linear(3,1,bias=False)
Bk = lB.state_dict()
print(Bk["weight"])
Bk["weight"] = B
lB.load_state_dict(Bk)
Bk = lB.state_dict()
print(Bk["weight"])
# print(A*B)
# print(torch.matmul(A,B))
# print(A+B)
# P, S, Q = np.linalg.svd(A, full_matrices=False)
# print(P)
# print(S)
# print(Q)
pass