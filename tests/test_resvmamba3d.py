import os,sys,time
sys.path.append(os.getcwd()) # 先这样!!!

import torch
import torch.nn as nn

from resvmamba.resvmamba3d import ResVMamba3dClassifierModel, make_default_config

def time_it(start_time):
    stop_time = time.time()
    print("耗时: {:.4f}秒".format(stop_time - start_time))
    return

config = make_default_config()

model = ResVMamba3dClassifierModel(config).cuda().eval()

data = torch.randn((4, 1, 128, 128, 128)).cuda()

for i in range(10):
    print(f"Run{i}")
    start_time = time.time()
    result= model(data)
    print(result.shape)
    torch.cuda.empty_cache()
    time_it(start_time)
  

input()
