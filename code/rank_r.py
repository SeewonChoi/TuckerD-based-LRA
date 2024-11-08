import numpy as np
from sklearn.utils.extmath import randomized_svd
import time
import torch
import tensorly as tl

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def svds(A, m):
    U, Sigma, VT = randomized_svd(A, n_components = m, random_state = None)
    return U, Sigma, VT

def ten2mat(tensor, mode):
    return tl.unfold(tensor, mode)

def HOOI(X, k):
    if X.ndim != 3:
        raise ValueError("Please ensure X is a 3-d tensor!")
    U = []
    # U = self.st_HOSVD(k)
    U.append(torch.randn(X.shape[1], k).to(device))
    U.append(torch.randn(X.shape[2], k).to(device))
    iters = 1
    i = 0
    oshape = X.shape
    while i < iters:
        for j in [0, 1]:
            B = tl.unfold(X, j + 1)
            B = torch.matmul(U[j].T, B)
            nshape = list(oshape)
            nshape[j+1] = U[j].shape[1]
            B = tl.fold(B, j + 1, tuple(nshape))
            B = tl.unfold(B, 2-j)
            # u, __, __ = self.svds(B, k)
            u, __, __ = torch.svd(B, some = True)
            u = u[:, :k]
            U[1 - j] = u
        i += 1
    return U

def cal_sketch(X, args):
    k, side = args.k, args.side
    if side == 0:
        A = ten2mat(X, 1)
        # U, __, __ = svds(A, k)
        U, __, __ = torch.svd(A, some = True)
        S = U[:, :k].T
        return S
    elif side == 1:
        A = ten2mat(X, 2)
        # U, __, __ = svds(A, k)
        U, __, __ = torch.svd(A, some = True)
        W = U[:, :k].T
        return W
    elif side == 2:
        U = HOOI(X, k)
        return [U[0].T, U[1].T]
    else:
        print("Value of side must be  0 or 1 or 2!")
        raise ValueError 

def cal_lowrank(U, A, args):
    k, r, side = args.k, args.r, args.side
    m, n = A.shape[0], A.shape[1]
    if side == 0:
        if isinstance(U, tuple):
            SA = torch.Tensor(k, n).fill_(0).to(device)
            sketch_vector = U[0]
            sketch_value = U[1]
            for i in range(m): 
                mapR = sketch_vector[i]  
                SA[mapR] += A[i] * sketch_value[i]  
        else:
            SA = U@A
        Q, __ = torch.linalg.qr(SA.T, "reduced")
        # U1, Sigma, VT = svds(A@Q, r)
        U1, Sigma, V = torch.svd(A@Q, some = True)
        VT = V.T
        U1, Sigma, VT = U1[:, :r], Sigma[:r], VT[:r, :]
        return U1@torch.diag(Sigma)@VT@(Q.T)
    elif side == 1:
        if isinstance(U, tuple): 
            WAt = torch.Tensor(k, m).fill_(0).to(device)
            sketch_vector = U[0]
            sketch_value = U[1]
            for i in range(n): 
                mapR = sketch_vector[i]  
                WAt[mapR] += A[:, i] * sketch_value[i]  
        else:
            WAt = U@A.T
        Q, __ = torch.linalg.qr(WAt.T, "reduced")
        # U1, Sigma, VT = svds((Q.T)@A, r)
        U1, Sigma, V = torch.svd((Q.T)@A, some = True)
        V = V.T
        U1, Sigma, VT = U1[:, :r], Sigma[:r], VT[:r, :]
        return Q@U1@torch.diag(Sigma)@VT
    elif side == 2:
        if isinstance(U, tuple):
            SA = torch.Tensor(k, n).fill_(0).to(device)
            WAt = torch.Tensor(k, m).fill_(0).to(device)
            sketch_vector1, sketch_value1 = U[0], U[1]
            sketch_vector2, sketch_value2 = U[2], U[3]
            for i in range(m): 
                mapR = sketch_vector1[i]  
                SA[mapR] += A[i] * sketch_value1[i]  
            for i in range(n): 
                mapR = sketch_vector2[i]  
                WAt[mapR] += A[:, i] * sketch_value2[i]  
        else:
            SA = U[0]@A
            WAt = U[1]@A.T
        Q, __ = torch.linalg.qr(SA.T, "reduced")
        P, __ = torch.linalg.qr(WAt.T, "reduced")
        # U1, Sigma, VT = svds(P.T@A@Q, r)
        U1, Sigma, V = torch.svd(P.T@A@Q, some = True)
        return P@U1[:, :r]@torch.diag(Sigma[:r])@V[:, :r].T@(Q.T)       
    else:
        raise ValueError("Incorrect value of side!")

def error(Abest, Te, args, U):
    err = 0.0
    timeR = 0.0 
    for j in range(Te.shape[0]):
        A = Te[j]
        best = tl.tensor(Abest[j]).to(device)
        tic = time.time()
        A_approx = cal_lowrank(U, A, args)
        toc = time.time()
        timeR += toc - tic 
        err += (torch.linalg.norm(A - A_approx) - torch.linalg.norm(A - best))/torch.linalg.norm(A - best)
    return [err.item()/Te.shape[0], timeR]
