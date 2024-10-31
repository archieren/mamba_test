安装causal-conv1d,mamba-ssm,及point ops的注意事项
我采用的是从源码安装.
0) 前期工作,大概
$conda install -c "nvidia/label/cuda-xx.xx.xx" cuda-toolkit
$conda install gcc_linux-64==11.2.0 gxx_linux-64==11.2.0  #注意版本,也不用自己搞符号链接!!!

1) 去掉系统设置的CUDA_PATH,CUDA_HOME项,以便系统找到虚拟python环境下的相应cuda环境!
$unset CUDA_PATH 

2) 检查gencode选项,是否指定生成针对本机GPU Capability的Image.
例如,我这台机器 GTX 1080 ti的计算能力是6.1, cc_flag中没有相应的设置,
所以就应当在两者的setup.py中添加两行:
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_60,code=sm_60")
####我估计直接从pip安装,就是这个问题没处理好!

3)
$cd ...causal-conv1d....
$CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install . --no-build-isolation
$cd ...mamba...
$MAMBA_FORCE_BUILD=TRUE pip install . --no-build-isolation


4)  到pointnet2里将pointnet2_ops_lib导出来!!! 
# 我用的是https://github.com/erikwijmans/Pointnet2_PyTorch里的
$cd ...pointnet2_ops_lib..
$pip install .

5) 到pointcept里, 将pointops,pointops2导出来,本地安装
$cd ...pointops2...
$pip install .
$cd ...pointops...
$pip --no-build-isolation install . 

关于pyg的 PyTorch Geometric
# 当然, torch和cuda的版本,自行调整!!!
$pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu118.html
$pip install -U torch_geometric

关于点集操作,有些要注意的问题:
pointnet2_ops_lib里的提供的,对batch的使用是固定.
而torch_geometric里的及pointcept的pointops?里的,是基于Graph视角的.
看./assets/offset_dark.png这张图!!!