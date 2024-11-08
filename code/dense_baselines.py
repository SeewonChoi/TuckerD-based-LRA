import torch
import time
import rank_r
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_hostname():
    with open("/etc/hostname") as f:
        hostname=f.read()
    hostname=hostname.split('\n')[0]
    return hostname

def fewshotsgd(A_train, args):
    r = args.r
    k = args.k
    lr = 1.0
    m, n=A_train.shape[1], A_train.shape[2]
    N_train, N_test = args.N_train, args.N_test
    print("Working on data: ", args.dataType)
    print_freq = 1
    sketch_vector = torch.randint(k, [m]).int().to(device)  # k * m
    sketch_value = ((torch.randint(2, [m]).float() - 0.5) * 2).to(device)
    S = torch.Tensor(k, m).fill_(0).to(device)
    for i in range(m):
        S[sketch_vector[i], i] = sketch_value[i]
    S = S/torch.norm(S)
    S.requires_grad = True
    for bigstep in range(args.iter):
        I0 = torch.Tensor(r, n).fill_(0)
        for i in range(r):
            I0[i, i] = 1.0
        A = A_train[int(torch.randint(N_train, [1]).item())].to(device)
        U, __, __ = torch.svd(A)
        Uk = U[:, :r]
        loss = torch.norm(Uk.T@S.T@S@U - I0)**2
        loss.backward()
        if bigstep%print_freq == 0:
            print("epoch:", bigstep, " loss:", loss.cpu().item())
        S.data -= lr*S.grad.data
        S.grad.data.fill_(0)
        del I0, A, U, Uk, loss
    print("Generating sketch done!\n")
    return S.detach()

def one_shot_one_vec(A_train, args):
    k = args.k
    m = A_train.shape[1]
    A = A_train[int(torch.randint(A_train.shape[0], [1]).item())]
    sketch_vector = torch.randint(k, [m]).int().to(device)  # k * m
    sketch_value = torch.randint(2, [m]).float().to(device)
    I = list()
    for i in range(k):
        I.append(np.where(sketch_vector.cpu() == i)[0])
    for i in range(k):
        u, __, __ = torch.svd(A[I[i]], some = True)
        for j in range(I[i].shape[0]):
            sketch_value[I[i][j]] = u[j, 0]
    return  (sketch_vector, sketch_value)

def dense_ivy(A_train, args):
    r=args.r
    k=args.k
    lr=10
    lr = lr*args.lr_S
    m,n=A_train.shape[1], A_train.shape[2]
    N_train=args.N_train
    print("Working on data ", args.dataType)
    print_freq=50
    if args.sketchType == "IVY_dense":
        sketch_vector = torch.randint(k, [m]).int().to(device)  # k * m
        sketch_value = ((torch.randint(2, [m]).float() - 0.5) * 2).to(device)
    if args.sketchType == "1Shot1Vec_IVY_dense":
        sketch_vector, sketch_value = one_shot_one_vec(A_train, args)
    sketch=torch.zeros((k, m)).to(device)
    for i in range(m):
        sketch[sketch_vector[i], i] = sketch_value[i]
    sketch.requires_grad=True
    for bigstep in range(args.iter):
        if ((bigstep+1)%1000==0) and lr>1:
            lr=lr*0.3 
        A = A_train[int(torch.randint(N_train, [1]).item())].to(device)/args.scale
        SH = torch.mm(sketch, A)
        __, __, V2 = torch.linalg.svd(SH, full_matrices = False)
        V2 = V2.T
        AU = A.mm(V2)
        U3, Sigma3, V3 = torch.linalg.svd(AU, full_matrices = False)
        V3 = V3.T
        ans = U3[:, :r].mm(torch.diag(Sigma3[:r])).mm(V3.t()[:r]).mm(V2.t())
        loss = torch.norm(ans - A)**2
        loss.backward()
        if bigstep%print_freq==0:
            print("epoch", bigstep, " loss:", loss.cpu().item())
        sketch.data -= lr*sketch.grad.data
        sketch.grad.data.fill_(0)
        del A, SH, V2, AU, U3, Sigma3, V3, ans, loss
    print("Generating sketch done!\n")
    return sketch.detach()


