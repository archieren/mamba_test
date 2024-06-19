"""
关于点集的操作
"""
import torch
device='cuda'
torch.device(device)

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def test_pointops():
    from pointops import farthest_point_sampling as fps
    x = torch.randn(10000,3).to(device)
    offset = torch.tensor([4999,5001], device=device).cumsum(0)
    new_offset = torch.tensor([100,101],device=device).cumsum(0)
    idx = fps(x, offset, new_offset)
    print(new_offset)
    print(idx.shape)
    
def test_ball_query():
    print("test_ball_query")
    from pointops import ball_query as bq
    from pointops import farthest_point_sampling as fps

    x = torch.randn(10000,3).to(device)
    offset = torch.tensor([4999,5001], device=device).cumsum(0)
    new_offset = torch.tensor([100,101],device=device).cumsum(0)
    idx = fps(x, offset, new_offset)
    centers= x[idx]
    idx_b, dist = bq(7, 0.04, 0.000001, x, offset, centers, new_offset)  # 感觉返回的dist似乎有问题! 最好别用!
    print(idx_b[0,:])
    print(dist[0,:])
    print(centers[0])
    print(x[idx_b[0,0]])
    print(square_distance(centers[0].view(1,1,3),x[idx_b[0,0]].view(1,1,3)))
    print(square_distance(centers[0].view(1,1,3),x[idx_b[0,0:7]].view(1,7,3)))

def test_knn_query():
    print("test_knn_query")
    from pointops import knn_query as kq
    from pointops import farthest_point_sampling as fps

    x = torch.randn(10000,3).to(device)
    offset = torch.tensor([4999,5001], device=device).cumsum(0)
    new_offset = torch.tensor([100,101],device=device).cumsum(0)
    idx = fps(x, offset, new_offset)
    centers= x[idx]
    idx_b, dist = kq(7, x, offset, centers, new_offset) 
    # print(idx_b[0,:])
    print(dist[0,:])
    # print(centers[0])
    # print(x[idx_b[0,0]])
    # print(square_distance(centers[0].view(1,1,3),x[idx_b[0,0]].view(1,1,3)))
    # print(square_distance(centers[0].view(1,1,3),x[idx_b[0,0:7]].view(1,7,3)))

def test_knn_query_2():
    print("test_knn_query_2")
    from pointops2.pointops2 import knnquery as kq
    from pointops2.pointops2 import furthestsampling as fps

    x = torch.randn(10000,3).to(device)
    offset = torch.tensor([4999,5001], device=device).cumsum(0).int()  #和pointops里的比,还要注意类型转换!
    new_offset = torch.tensor([100,101],device=device).cumsum(0).int()
    print(offset[0])
    idx = fps(x, offset, new_offset)
    centers= x[idx]
    idx_b, dist = kq(7, x,  centers, offset, new_offset)   # 和pointops里的比,参数次序发生了变化,还要注意类型转换!
    # print(idx_b[0,:])
    print(dist[0,:])
    # print(centers[0])
    # print(x[idx_b[0,0]])
    # print(square_distance(centers[0].view(1,1,3),x[idx_b[0,0]].view(1,1,3)))
    # print(square_distance(centers[0].view(1,1,3),x[idx_b[0,0:7]].view(1,7,3)))


# test_pointops()
# test_ball_query()
test_knn_query()
test_knn_query_2()