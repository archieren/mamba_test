# 测试那个官方实现!!!
import torch
import time

device='cuda'
torch.device(device)

def time_it(start_time):
    stop_time = time.time()
    print("耗时: {:.2f}秒".format(stop_time - start_time))
    return

def test_mamba_ssm():
    from mamba_ssm import Mamba2 as Mamba
    from torchinfo import summary
    import torch.autograd.profiler as profiler 
    
    batch, length, dim = 2, 1024*128, 256
    x = torch.randn(batch, length, dim).to(device)
    x.requieres_grad = True
    #  对 Mamba2 要 d_model * expand / headdim = multiple of 8
    model = Mamba(
        # This module uses roughly 3 * expand * d_model^2 parameters
        d_model=dim, # Model dimension d_model
        d_state=32,  # SSM state expansion factor
        d_conv=4,    # Local convolution width
        expand=2,    # Block expansion factor
    ).to(device)
    print(model.A_log.shape)
    print(sum([p.numel() for p in model.parameters()]))
    input_size = (batch,length,dim)
    summary(model,input_size)
    s = time.time()
    y = model(x)
    t = time.time()
    print("耗时: {:.2f}秒".format(t - s))
    assert y.shape == x.shape
    print(y.shape)

    with profiler.profile(record_shapes=True, use_cuda=True, profile_memory=True) as prof:
        with profiler.record_function("model_forward"):
            output = model(x)
            loss = output.sum()
        with profiler.record_function("model_backward"):
            loss.backward()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    print(f"Peak CUDA Memory Usage: {prof.total_average().cuda_memory_usage / (1024 ** 2)} MB")

def test_causal_conv1d():
    from causal_conv1d import causal_conv1d_fn

    batch, dim, seq, width = 10, 5, 17, 4
    x = torch.zeros((batch, dim, seq)).to(device)
    weight = torch.zeros((dim, width)).to(device)
    bias = torch.zeros((dim, )).to(device)

    y = causal_conv1d_fn(x, weight, bias, None)
    print(y.shape)

def test_mamba_ssm_var_input():
    from mamba_ssm import Mamba2 as Mamba
    from torchinfo import summary
    import torch.autograd.profiler as profiler 
    
    batch, length, dim = 2, 1024*128, 256
    # x = torch.randn(batch, length, dim).to(device)
    # x.requieres_grad = True
    #  对 Mamba2 要 d_model * expand / headdim = multiple of 8
    model = Mamba(
        # This module uses roughly 3 * expand * d_model^2 parameters
        d_model=dim, # Model dimension d_model
        d_state=32,  # SSM state expansion factor
        d_conv=4,    # Local convolution width
        expand=2,    # Block expansion factor
    ).to(device)

    # 场景: 2个批次，每个批次包含不同长度的序列
    max_seqlen = length

    # 输入数据 (标准格式)
    x = torch.randn(batch, max_seqlen, dim).to(device)

    # cu_seqlens:
    cu_seqlens = torch.tensor([0, 1024*32, 1024*128]).int().to(device)

    # seq_idx: 标识子序列 (每个批次内分成2个子序列)
    seq_idx = torch.tensor([
        [0]*(1024*32) + [1]*(1024*32) + [2]*(1024*64),  # 第一个批次: 12+12+8=32
        [0]*(1024*64) + [1]*(1024*64)                   # 第二个批次: 16+16=32
    ]).int().to(device)

    # 同时使用两个参数
    _ = model(x, seq_idx=seq_idx)
    
    start_time = time.time()
    output = model(x, seq_idx=seq_idx)
    time_it(start_time)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"seq_idx 形状: {seq_idx.shape}")
#test_causal_conv1d()

test_mamba_ssm_var_input()