import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd.profiler as profiler

class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1)
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)
        energy = torch.bmm(x_q, x_k)
        attention = F.softmax(energy / np.sqrt(x_q.size(-1)), dim=-2)
        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x_r)))
        x = x + x_r
        return x
    
class Point_Transformer_Last(nn.Module):
    def __init__(self, channels=128):
        super(Point_Transformer_Last, self).__init__()
        self.sa1 = SA_Layer(channels)
        self.sa2 = SA_Layer(channels)
        self.sa3 = SA_Layer(channels)
        self.sa4 = SA_Layer(channels)

    def forward(self, x):
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x = torch.cat((x1, x2, x3, x4), dim=1)

        return x
    
device='cuda'

B, C, L= 1, 128, 1024*16
X = torch.randn(B, C, L).to(device, non_blocking=True)
model= Point_Transformer_Last().to(device)

model(X)

print(X.size())
print(X.shape)
input()
with profiler.profile(record_shapes=True, use_cuda=True, profile_memory=True) as prof:
    with profiler.record_function("model_forward"):
        output = model(X)
        loss = output.sum()
    with profiler.record_function("model_backward"):
        loss.backward()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
print(f"Peak CUDA Memory Usage: {prof.total_average().cuda_memory_usage / (1024 ** 2)} MB")