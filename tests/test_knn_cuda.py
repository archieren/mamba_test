import torch
import numpy as np
from sklearn.neighbors import KDTree

import time
s = time.time()
from knn_cuda import KNN
t = time.time()
print("耗时: {:.2f}秒".format(t - s))



def t2n(t):
    return t.detach().cpu().numpy()


def run_kdtree(ref, query, k):
    bs = ref.shape[0]
    D, I = [], []
    for j in range(bs):
        tree = KDTree(ref[j], leaf_size=100)
        d, i = tree.query(query[j], k=k)
        D.append(d)
        I.append(i)
    D = np.stack(D)
    I = np.stack(I)
    return D, I


def run_knnCuda(ref, query, k):
    ref = torch.from_numpy(ref).float().cuda()
    query = torch.from_numpy(query).float().cuda()
    knn = KNN(k, transpose_mode=True)
    d, i = knn(ref, query)
    return t2n(d), t2n(i)


def compare(k, dim, n1, n2=-1):
    if n2 < 0:
        n2 = n1
    for _ in range(5):
        ref = np.random.random((2, n1, dim))
        query = np.random.random((2, n2, dim))

        kd_dist, kd_idices = run_kdtree(ref, query, k)
        kn_dist, kn_idices = run_knnCuda(ref, query, k)

        diff = (kd_idices - kn_idices) != 0
        print(kd_dist[diff])
        print(kn_dist[diff])

        np.testing.assert_almost_equal(kd_dist, kn_dist, decimal=3)
        #np.testing.assert_array_equal(kd_idices, kn_idices)


def test_knn_cuda_400_5_1000():
    compare(400, 5, 1000)

def test_knn_cuda_400_5_100():
    compare(10, 5, 100)

def test_knn_cuda_400_5_10():
    compare(2, 5, 10)

def test_knn_cuda_400_5_1001():
    compare(400, 5, 1001)

def test_knn_cuda_400_5_101():
    compare(10, 5, 101)

def test_knn_cuda_400_5_11():
    compare(2, 5, 11)

def test_knn_cuda_400_5_300000_50():
    compare(400, 5, 30000, 50)

def test_knn_cuda_400_5_300001_50():
    compare(400, 5, 30001, 50)

def test_knn_cuda_400_5_10000():
    compare(400, 5, 10000)

def test_knn_cuda_400_5_10001():
    compare(400, 5, 10001)

import time
s = time.time()
test_knn_cuda_400_5_300001_50()
t = time.time()
print("耗时: {:.2f}秒".format(t - s))
