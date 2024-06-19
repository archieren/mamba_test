import torch
#关于点云数据的处理!
@torch.inference_mode()
def offset2bincount(offset):
    """
    args:
        "offset": 应当是[N1, N1+N2, N1+N2+N3, ..., N1+N2+.....+Nb]
    return:
        应当是[N1,N2,N3,...,Nb]
    """
    return torch.diff(
        offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long)
    )


@torch.inference_mode()
def offset2batch(offset):
    """
    args:
        "offset": 应当是[N1, N1+N2, N1+N2+N3, ..., N1+N2+.....+Nb]
    return: 应当是[N1s 0, N2s 1,..., Nbs (b-1)]
    """
    bincount = offset2bincount(offset)
    return torch.arange(
        len(bincount), device=offset.device, dtype=torch.long
    ).repeat_interleave(bincount)


@torch.inference_mode()
def batch2offset(batch):
    """
    args: 
        "batch": 应当形如[N1s 0, N2s 1,..., Nbs (b-1)]
    return:
        应当是[N1, N1+N2, N1+N2+N3, ..., N1+N2+.....+Nb]
    """
    return torch.cumsum(batch.bincount(), dim=0).long()