from data import gen_hsi, gen_brain, gen_video
import rank_r, dense_baselines

def load_data(rawdir, args):
    if args.dataType == "hsi":
        A_train, A_test = gen_hsi.data(rawdir, args)
    elif args.dataType == "brain":
        A_train, A_test= gen_brain.data(rawdir, args)
    elif args.dataType == "logo":
        A_train, A_test = gen_video.data(rawdir, args)
    else:
        raise AttributeError("Please input correct datatype!")
    return A_train, A_test

def load_sketch(A_train, args):
    if args.sketchType == "Tensorb":
        U = rank_r.cal_sketch(A_train, args)
    elif args.sketchType == "FewShotSGD-2_dense":
        args.iter = 2
        U = dense_baselines.fewshotsgd(A_train, args)
    elif args.sketchType == "FewShotSGD-3_dense":
        args.iter = 3
        U = dense_baselines.fewshotsgd(A_train, args)
    elif args.sketchType == "IVY_dense":
        U = dense_baselines.dense_ivy(A_train, args)
    elif args.sketchType == "1Shot1Vec_IVY_dense":
        U = dense_baselines.dense_ivy(A_train, args)
    else:
        raise AttributeError("Please input available sketch type!")
    return U
    
