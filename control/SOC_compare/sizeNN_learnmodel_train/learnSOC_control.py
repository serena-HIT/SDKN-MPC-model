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
## 小函数1
# 将矩阵 Q 投影到半正定锥上 
# 这需要进行特征分解，然后将负特征值设置为零， 
# 或者将所有特征值设置在区间 [epsilon, delta]（如果指定）。
def projectInvertible(S, epsilon):
    s, v, d = np.linalg.svd(S)
    if np.min(v) < epsilon:
        Sp = np.dot(np.dot(s, np.diag(np.maximum(np.diag(v), epsilon))), d.T)
    else:
        Sp = S
    return Sp

## 小函数2
def projectPSD(Q, epsilon=0, delta=float('inf')):
    # if not Q:
    #     return Q
    if epsilon == 0:
        epsilon = 0
    if delta == float('inf'):
        delta = +float('inf')
    Q = (Q + Q.T) / 2
    # if np.any(np.isnan(Q)) or np.any(np.isinf(Q)):
    #     raise ValueError('Input matrix has infinite or NaN entries')
    e, V = np.linalg.eig(Q)
    e = np.diag(e)
    Qp = V.dot(np.diag(np.minimum(delta, np.maximum(np.diag(e), epsilon)))).dot(V.T) 
    return Qp

## 小函数3
def poldec(A):
    # POLDEC   Polar decomposition.
    #         [U, H] = POLDEC(A) computes a matrix U of the same dimension
    #         (m-by-n) as A, and a Hermitian positive semi-definite matrix H,
    #         such that A = U*H.
    #         U has orthonormal columns if m >= n, and orthonormal rows if m <= n.
    #         U and H are computed via an SVD of A.
    #         U is a nearest unitary matrix to A in both the 2-norm and the
    #         Frobenius norm.
    #
    #         Reference:
    #         N. J. Higham, Computing the polar decomposition---with applications,
    #         SIAM J. Sci. Stat. Comput., 7(4):1160--1174, 1986.
    #
    #         (The name `polar' is reserved for a graphics routine.)
    
    P, S, Q = np.linalg.svd(A, full_matrices=False)  # Economy size.
    # if m < n                # Ditto for the m<n case.
    #    S = S[:, 1:m];
    #    Q = Q[:, 1:m];
    # end
    Q = Q.T
    S = np.diag(S)
    U = P.dot(Q.T)
    if len(np.shape(U)) == 2 and np.shape(U)[0] == np.shape(U)[1]:
        H = Q.dot(np.dot(S, Q.T))
    else:
        H = None
    pass
    return U, H

## 小函数4
def checkdstable(A, epsilon):
    n = len(A)
    P = dlyap(A.T, np.eye(n))

    if len(np.shape(P)) == 2 and np.shape(P)[0] == np.shape(P)[1]:
        S = sqrtm(P)
        OC = np.dot(np.dot(S, A), np.linalg.inv(S))
        O, C = poldec(OC)
        C = projectPSD(C, 0, 1 - epsilon)
        return P, S, O, C
    else:
        return P

## 主函数
def learnSOCmodel_withControl(X, Y, U, options):
    XU = torch.cat([X,U],axis=0)
    #Y = torch.cat([Y,U_next],axis=-1)
    X = X.cpu().detach().numpy()
    Y = Y.cpu().detach().numpy()
    U = U.cpu().detach().numpy()
    XU = XU.cpu().detach().numpy() 
    # print(X.shape)
    # print(XU.shape)
    n = X.shape[0]
    # nA2 = np.linalg.norm(Y, 'fro')**2

    XYtr = np.dot(X, Y.T)
    XXt = np.dot(X, X.T)
    XUtr = np.dot(X, U.T)
    YUtr = np.dot(Y, U.T)
    UUtr = np.dot(U, U.T)
    #options
    if 'options' not in locals() or options is None:
        options = {}

    if 'maxiter' not in options:
        options['maxiter'] = float('inf')

    if 'timemax' not in options:
        options['timemax'] = 10

    if 'posdef' not in options:
        options['posdef'] = 1e-12

    if 'astab' not in options:
        options['astab'] = 0
    else:
        if options['astab'] < 0 or options['astab'] > 1:
            raise ValueError('options.astab must be between 0 and 1.')

    if 'display' not in options:
        options['display'] = 1

    if 'graphic' not in options:
        options['graphic'] = 0

    if 'init' not in options:
        options['init'] = 1
    
    #初始化
    e100 = np.full((100, 1), np.nan)  # Preallocate for speed
    # Given initialization + projection
    # Standard initialization
    S = np.eye(n)
    AB_ls = np.dot(Y, pinv(XU))
    nA2 = norm(Y - np.dot(AB_ls, XU), 'fro')**2 / 2
    O, C = polar(AB_ls[0:n, 0:n])  # Y = K*X, so K = Y * pinv(X)
    try:
        # sometimes, due to numerics O*C can be stable, but S\O*C*S may not
        # -- theoretically, that is not possible.
        eigvals, _ = eigs(solve(S, np.dot(O, np.dot(C, np.dot(S, np.eye(n))))), 1, which='LM', maxiter=1000, ncv=n)
        eA = np.abs(eigvals[0])
    except:
        # print('Max eigenvalue not converged. Project C matrix \n')
        eA = 2  # anything larger than 1
    B = AB_ls[0:n, n:]
    if eA > 1 - options['astab']:
        C = projectPSD(C, 0, 1 - options['astab'])
        # print(Y.shape)
        # print(S.shape)
        # print(O.shape)
        # print(C)
        # print(B.shape)
        # print(U.shape)
        e_old = norm(Y - np.dot(solve(S, np.dot(O, np.dot(C, S))), X) - np.dot(B, U), 'fro')**2 / 2
    else:
        e_old = norm(Y - np.dot(AB_ls, XU), 'fro')**2 / 2

    maxeA = max(1, eA)
    Astab = AB_ls[0:n, 0:n] / maxeA

    _, Stemp, Otemp, Ctemp = checkdstable(0.9999 * Astab, options['astab'])
    etemp = norm(Y - np.dot(solve(Stemp, np.dot(Otemp, np.dot(Ctemp, Stemp))), X) - np.dot(B, U), 'fro')**2 / 2

    if etemp < e_old:
        S = Stemp
        O = Otemp
        C = Ctemp
        e_old = etemp
    options['alpha0'] = 0.5  # Parameter of FGM, can be tuned.
    options['lsparam'] = 1.5
    options['lsitermax'] = 60
    options['gradient'] = 0

    i = 1
    alpha = options['alpha0']
    Ys = S
    Yo = O
    Yc = C
    Yb = B
    restarti = 1
    begintime0 = time.time()

    #Main loop
    while i < options['maxiter']:
        if time.time() - begintime0 > 1800:
            break
        alpha_prev = alpha
        # Compute gradient计算梯度
        Atemp = solve(S, np.dot(O, np.dot(C, S)))
        #result = np.linalg.lstsq(a.T, XU.T, rcond=None)[0].T##XU/a
        # test1 = np.linalg.lstsq(S.T, (XYtr - np.dot(XXt, Atemp.T) - np.dot(XUtr, B.T)), rcond=None)[0].T
        # print(test1)
        Z = - np.linalg.lstsq(S.T, (XYtr - np.dot(XXt, Atemp.T) - np.dot(XUtr, B.T)).T, rcond=None)[0].T
        gS = (np.dot(Z, np.dot(O, C)) - np.dot(Atemp, Z)).T
        gO = (np.dot(C, np.dot(S, Z))).T
        gC = (np.dot(S, np.dot(Z, O))).T
        gB = (np.dot(Atemp, XUtr) + np.dot(B, UUtr) - YUtr)
        inneriter = 1
        step = 1
        # For i == 1, we always have a descent direction
        # 对于 i == 1，我们总是有一个下降方向
        Sn = Ys - gS * step
        On = Yo - gO * step
        Cn = Yc - gC * step
        Bn = Yb - gB * step
        # Project onto feasible set
        Sn = projectInvertible(Sn, options['posdef'])
        try:
            test2 = eigs(solve(Sn, np.dot(On, np.dot(Cn, Sn))), 1, which='LM', maxiter=1000, ncv=n)
            #print(len(test2))
            index = np.argmax(np.abs(test2[0]))
            maxE = np.abs(test2[0])[index]
        except:
            # print('Max eigenvalue not converged. Project On and Cn matrices \n')
            maxE = 2  # anything larger than 1
        if maxE > 1 - options['astab']:
            On = polar(On)[0]
            Cn = projectPSD(Cn, 0, 1 - options['astab'])
        # print(On.shape)
        # print(Cn.shape)
        # print(Sn.shape)
        # print(solve(Sn, np.dot(On, np.dot(Cn, Sn))).shape)
        # print(X.shape)
        e_new = norm(Y - np.dot(solve(Sn, np.dot(On, np.dot(Cn, Sn))), X) - np.dot(Bn, U), 'fro')**2 / 2
        
        # Barzilai and Borwein
        while e_new > e_old and inneriter <= options['lsitermax']:
            # 对于 i == 1，我们总是有一个下降方向
            Sn = Ys - gS * step
            On = Yo - gO * step
            Cn = Yc - gC * step
            Bn = Yb - gB * step
            # 投影到可行集上
            Sn = projectInvertible(Sn, options['posdef'])
            try:
                test3 = eigs(solve(Sn, np.dot(On, np.dot(Cn, Sn))), 1, which='LM', maxiter=1000, ncv=n)
                #print(test3.shape)
                #print(test3)
                index = np.argmax(np.abs(test3[0]))
                maxE = np.abs(test3[0])[index]
            except:
                # print('Max eigenvalue not converged. Project On and Cn matrices \n')
                maxE = 2  # 任何大于1的值
            if maxE > 1 - options['astab']:
                On = polar(On)[0]
                Cn = projectPSD(Cn, 0, 1 - options['astab'])
            e_new = norm(Y - np.dot(solve(Sn, np.dot(On, np.dot(Cn, Sn))), X)- np.dot(Bn, U), 'fro')**2 / 2
            if e_new < e_old and e_new > prev_error:
                break
            if inneriter == 1:
                prev_error = e_new
            else:
                if e_new < e_old and e_new > prev_error:
                    break
                else:
                    prev_error = e_new
            step = step / options['lsparam']
            inneriter = inneriter + 1
        pass
        # 如果获得了减小，则使用 FGM 权重共轭，否则重新启动 FGM
        alpha = (math.sqrt(alpha_prev**4 + 4*alpha_prev**2) - alpha_prev**2) / 2
        beta = alpha_prev * (1 - alpha_prev) / (alpha_prev**2 + alpha)
        if e_new > e_old:  # 线搜索失败
        #if inneriter > options['lsitermax']:  # 线搜索失败
            if restarti == 1:
                # 如果不是下降方向，则重新启动 FGM
                restarti = 0
                alpha = options['alpha0']
                Ys = S
                Yo = O
                Yc = C
                Yb = B
                e_new = e_old
                # 重新初始化步长
            elif restarti == 0:  # 没有先前的重新启动且没有下降方向 => 收敛
                e_new = e_old
                break
        else:
            restarti = 1
            # 共轭
            if options['gradient'] == 1:
                beta = 0
            Ys = Sn + beta * (Sn - S)
            Yo = On + beta * (On - O)
            Yc = Cn + beta * (Cn - C)
            Yb = Bn + beta * (Bn - B)
            # 保留新的迭代结果
            S = Sn
            O = On
            C = Cn
            B = Bn
        i = i + 1
        current_i = (i % 100) + ((i % 100) == 0) * 100  # i 落在 [1, 100] 范围内
        e100[current_i - 1] = e_old
        current_i_min_100 = ((current_i + 1) % 100) + (((current_i + 1) % 100) == 0) * 100
        # 检查误差是否很小（1e-6 相对误差）
        if e_old < 1e-6 * nA2 or (i > 100 and e100[current_i_min_100 - 1] - e100[current_i - 1] < 1e-8 * e100[current_i_min_100 - 1]):
            break
        e_old = e_new
    pass
    # 优化解
    A = solve(S, np.dot(O, np.dot(C, S)))
    # 沿着不稳定的 A 的方向移动，直到达到稳定边界（以减小误差）
    e0 = 0.001
    e_step = 0.0001
    e = e0
    n = len(A)
    #AB_ls = np.dot(Y, np.linalg.pinv(np.vstack((X, U))))
    grad = AB_ls - np.hstack((A, B))
    # A_ls = AB_ls[:n, :n]
    # B_ls = AB_ls[:n, n:]
    AB_new = np.hstack((A, B)) + e * grad
    Anew = AB_new[:n, :n]
    try:
        test4 = eigs(Anew, 1, which='LM', maxiter=1000, ncv=n)
        #print(test4.shape)
        #print(test4)
        index = np.argmax(np.abs(test4[0]))
        maxE = np.abs(test4[0])[index]
    except:
        # print('Max eigenvalue not converged. Consider Anew unstable \n')
        maxE = 2  # 任何大于1的值
    # while (maxE <= 1) and np.linalg.norm(Anew - A_ls, 'fro') > 0.00001:  # 不稳定的操作符
    while maxE < 1 and np.linalg.norm(AB_ls - AB_new, 'fro')**2 / 2 > 0.01:  # 不稳定的操作符
        e = e + e_step
        AB_new = np.hstack((A, B)) + e * grad
        Anew = AB_new[:n, :n]
        try:
            test5 = eigs(Anew, 1, which='LM', maxiter=1000, ncv=n)
            index = np.argmax(np.abs(test5[0]))
            maxE = np.abs(test5[0])[index]
        except:
            # print('Max eigenvalue not converged. Consider Anew unstable \n')
            maxE = 2  # 任何大于1的值
    pass
    if e != e0:
        ABtemp = np.hstack((A, B)) + (e - e_step) * grad
        A = ABtemp[:n, :n]
        B = ABtemp[:n, n:]
    stored_vars = globals().copy()
    memory_used = 0
    for var_name, var_value in stored_vars.items():
        memory_used += sys.getsizeof(var_value)
    pass
    error = np.linalg.norm(Y - np.dot(A, X) - np.dot(B, U), 'fro')**2 / 2
    # A = torch.from_numpy(A)
    # B = torch.from_numpy(B)

    return A, B, error, memory_used
