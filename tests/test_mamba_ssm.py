# 测试那个官方实现!!!
import torch
import time

device='cuda'
torch.device(device)

def test_mamba_ssm():
    from mamba_ssm import Mamba
    from torchinfo import summary
    import torch.autograd.profiler as profiler 
    
    batch, length, dim = 2, 1024*16, 32
    x = torch.randn(batch, length, dim).to(device)
    x.requieres_grad = True
    model = Mamba(
        # This module uses roughly 3 * expand * d_model^2 parameters
        d_model=dim, # Model dimension d_model
        d_state=32,  # SSM state expansion factor
        d_conv=4,    # Local convolution width
        expand=2,    # Block expansion factor
    ).to(device)
    print(model.A_log.shape)
    print(sum([p.numel() for p in model.parameters()]))
    # input_size = (batch,length,dim)
    # summary(model,input_size)
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

test_causal_conv1d()

test_mamba_ssm()