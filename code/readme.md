# Tensor-based sketching method for the low-rank  approximation of data streams.

This repository is the official implementation of  our paper, **Tensor-based sketching method for the low-rank approximation of data streams**.  The experiments are done for low-rank approximation of  third-order tensors which are composed of matrices in data streams. 

## Environment

Our experiments are implemented with **Python 3.8.12** . 

## Requirements

All requirements :

```setup
pip install -r requirements.txt
```

> "requirements.txt" contains packages that are required in the experiment:
> 
> scipy==1.8.0
> 
> opencv-python==4.5.5.62
> 
> h5py==3.6.0
> 
> torch==1.11.0
> 
> argparse==1.1
> 
> tensorly==0.7.0

## Data resources

To test the performance of our method, we use four datasets, including *hsi, logo, brain and cavity*. Due to  memory limit, we provide the following link of these datasets for users' downloading.  

https://figshare.com/articles/dataset/data_zip/19679505

## Running

To obtain experiment results in the paper, run this command:

```run
python main.py --dataType brain --normalize 1 --side 0 --r 10 --k 20 --raw 1 --bestdone 0 --N_train 100
```

> --dataType: 
> 
> choose datasets , including *hsi, logo, brain, cavity.* 
> 
> --normalize
> 
> scale the data by its top singular value.
> 
> --side:
> 
> choose methods in one-side (*side = 0*)  or two-side (*side = 2*) case. In one-side case, only one sketch matrix $S$ is used while in two-side case, two sketch matrices $S$ and $W$ are used for left and right multiplication, seperately.
> 
> --r:
> 
> target rank.
> 
> --k:
> 
> choose the sketch size $k$ , which is the number of rows of the sketch matrix $S$ . The sketch size $l$ of  $W$ is set to be equal to $k$.  Default $k$ equals $20$.
> 
> --raw:
> 
> choose whether to store the training and test sets. If training and test sets have  been stored,  set *raw = 0*, else, set *raw = 1*.
> 
> --bestdone:
> 
> choose whether the best approximations are done or not.
> 
> --N_train:
> 
> choose the size of training set, i.e., the number of matrices for training.

## Evaluation

We focus on the **accuracy**  and **efficiency** of  algorithms in the experiment. 

For accuracy, we use the following error metric:

$Error = \frac{\|A - \hat{A}\|_F - \|A - A_{opt}\|_F}{\|A - A_{opt}\|_F}$,

where $A$ is a matrix in the data stream, $A_{opt}$ is the best rank-$r$ approximation of $A$ and $\hat{A}$ is the low-rank approximation using our algorithm and the baselines.

For efficiency, we compare the time cost for generating sketch matrices with different sketching methods.

## Results

Our model compared to the baselines achieves the following performance on the dataset *hsi* when $k=l=20$.

| Method                          | Error  | Training time (s) |
| ------------------------------- | ------ | ----------------- |
| Tensor-based (ours)             | 0.019  | 0.53              |
| Tensor-Based (Two-Sided) (ours) | 0.069  | 0.39              |
| IVY                             | 0.089  | 205.9             |
| FewShot-2                       | 0.123  | 0.80              |
| FewShot-3                       | 0.092  | 0.99              |
| 1Shot1Vec+IVY                   | 0.0361 | 204.6             |
| 1Shot2Vec                       | 0.145  | 0.63              |
| Butterfly                       | 0.056  | 139.9             |
