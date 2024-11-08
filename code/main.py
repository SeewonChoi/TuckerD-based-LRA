import time
import argparse
import tensorly as tl
import rank_r
import sys
import pickle
import torch
from misc_utils import *

tl.set_backend("pytorch")
rawdir = 'data/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Process necessary parameters.')
    def add(*args, **kwargs):
        parser.add_argument(*args, **kwargs)
    add("--problemType", type = str, default = "rankr", help = "rankr")
    add("--dataType", type = str, default = "brain", help = "hsi|logo|brain")
    add("--sketchType", type = str, default = "1Shot1Vec_IVY_dense", help = "Tensorb|FewShotSGD-2_dense|FewShotSGD-3_dense|IVY_dense|1Shot1Vec_IVY_dense")
    add("--side", type = int, default = 0, choices = [0, 1, 2], help = "0->SA|1->SWt|2->SAWt")
    add("--N", type = int, default = 150, help = "range of train")
    add("--N_train", type = int, default = 30, help = "#train")
    add("--N_test", type = int, default = 120, help = "#test")
    add("--r", type = int, default = 10, help = "truncation for rank-r approximation")
    add("--k", type  = int, default = 20, help = "sketch size")
    add("--raw", type = int, default = 0, help = "1:True|0:False")
    add("--bestdone", type = int, default = 1, help = "1:True|0:False")
    add("--scale", type=int, default=1, help="scale")
    add("--lr_S", type=float, default=1, help="learning rate scale")
    add("--normalize", type=int, default=1, help="1:True|0:False")
    add("--iter", type=int, default=1000, help="total iterations")

    args = parser.parse_args()
    print(args)
    tic = time.time()
    A_train, A_test = load_data(rawdir, args)
    toc = time.time()
    print("Time for generating training/testing set: ", toc - tic)
    best_path = rawdir + 'trainset/' + args.dataType + '_best' + '.dat'
    if not args.bestdone:
        print("Start calculating best approximation...")
        A_best = []
        timeE = 0.0
        for i in range(A_test.shape[0]):
            tic = time.time()
            u, s, v = torch.svd(A_test[i], some = True)
            vt = v.T
            timeE += time.time() - tic
            A_best.append(u[:, :args.r]@torch.diag(s[:args.r])@vt[:args.r, :])
        print("Best approximation calculating finished.")
        with open(best_path, 'wb') as f:
            pickle.dump([A_best, timeE], f)
            print("Best approximation storing finished.")
    else:
        with open(best_path, 'rb') as f:
            A_best, timeE = pickle.load(f)
            
    print("Train data size:", A_train.shape)
    print("Test data size:", A_test.shape)
    #  to device
    A_train = tl.tensor(A_train).to(device)
    A_test = tl.tensor(A_test).to(device)
    if args.problemType == "rankr":
        tic = time.time()
        U = load_sketch(A_train, args)
        toc = time.time()
        timeS = toc - tic
        print("Time for exact SVD:", timeE)
        print("Time for computing sketching matrix:", timeS)
        ans = rank_r.error(A_best, A_test, args, U)
        print("Time for testing:", ans[1])
        print("Test error:", ans[0])
    
    else:
        print("Wrong problemType!")
        sys.exit()



        




















