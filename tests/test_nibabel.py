## Allows to import .nii files as pure numpy voxel grids, Open3D pointclouds, and Open3D voxel grids.

from typing import List, Optional, Tuple
import open3d as o3d
import nibabel as nib
from nibabel.orientations import axcodes2ornt, ornt_transform

import numpy as np
from skimage.measure import marching_cubes 
from pathlib import Path
import matplotlib.pyplot as plt

source_dir = Path("/home/archie/Projects/data/TYPE D")
       
def test_nibabel():
    img_path = source_dir / "12.nii.gz"
    img_voxels = nib.load(img_path)
    print(img_voxels.affine)
    # 2. 标准化方向到RAS 确保解剖方向一致：R(右), A(前), S(上)
    img_ras = nib.as_closest_canonical(img_voxels)
    data_ras = img_ras.get_fdata()

    # 3. 提取切片
    slice_index = 64  # 切片索引

    # 轴向切片 (Axial: X-Y平面, Z层)
    axial_slice = data_ras[::-1, :, slice_index]  # 形状 (X, Y)

    # 冠状切片 (Coronal: X-Z平面, Y层)
    coronal_slice = data_ras[::-1, slice_index, :]  # 形状 (X, Z)

    # 矢状切片 (Sagittal: Y-Z平面, X层)
    sagittal_slice = data_ras[slice_index, ::-1, :]  # 形状 (Y, Z)

    # 4. 显示切片
    fig, axes = plt.subplots(1, 3) # , figsize=(15, 5))

    # 轴向
    axes[0].imshow(axial_slice.T, cmap='gray', origin='lower')  # 转置并设置原点
    axes[0].set_title(f'Axial Slice (Z={slice_index})')
    axes[0].axis('off')

    # 冠状
    axes[1].imshow(coronal_slice.T, cmap='gray', origin='lower')
    axes[1].set_title(f'Coronal Slice (Y={slice_index})')
    axes[1].axis('off')

    # 矢状
    axes[2].imshow(sagittal_slice.T, cmap='gray', origin='lower')
    axes[2].set_title(f'Sagittal Slice (X={slice_index})')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()
    #marching_cubes


def test_nibabel_orient():
    voxels_path = source_dir / "12.nii.gz"
    voxels = nib.load(voxels_path)
    voxels_data = voxels.get_fdata()
    voxels_affine = voxels.affine.copy()
    
    orientation = nib.aff2axcodes(voxels_affine)  # 如 ('R', 'A', 'S')
    print(orientation)
    
    current_ornt = nib.io_orientation(voxels_affine)
    print(current_ornt)
    
        # 定义目标方向（RAS）
    target_ornt = axcodes2ornt('RAS')
    print(target_ornt)
    
    # 计算转换矩阵
    transform = ornt_transform(current_ornt, target_ornt)

    # 应用转换
    ras_data = nib.apply_orientation(voxels_data, transform)
    print(ras_data[63,64,65])
    ras_affine = nib.orientations.inv_ornt_aff(transform, voxels_data.shape)

    # 创建新图像
    ras_img = nib.Nifti1Image(ras_data, ras_affine)
    
    nib.save(ras_img, 'RAS_converted.nii.gz')
    
    image = ras_data[ras_data.shape[2]//2, :, :]
    # image = np.flip(image,(0, 1))
    # image = image.transpose((1,0))
    plt.imshow(image, cmap='Greys_r')
    plt.show()

def test_masked_image():
    # 加载原始影像和分割掩码
    img_path = source_dir / "12.nii.gz"
    img = nib.load(img_path)  # 替换为你的影像路径
    mask_path = source_dir / "seg-12.nii.gz"
    mask = nib.load(mask_path)  # 替换为你的分割掩码路径

    # 获取数据数组
    img_data = img.get_fdata()
    mask_data = mask.get_fdata()

    # 确保影像和掩码形状相同
    assert img_data.shape == mask_data.shape, "影像和掩码形状不匹配！"

    # 应用掩码：保留分割区域 (mask>0)，其余置0
    masked_data = img_data * (mask_data > 0)

    # 创建新的NIfTI图像（保留原始头信息和仿射矩阵）
    masked_img = nib.Nifti1Image(masked_data, img.affine, img.header)


    # 保存结果
    nib.save(masked_img, 'masked_image.nii.gz')  # 指定输出文件名
        
def test_nipy():
    from nipy import load_image
    
    
    # 加载图像
    path = source_dir / "12.nii.gz"
    img = load_image(path)
 
    # 显示图像
    print(img)

def test_torchio():
    import torchio as tio
    
    img_path = source_dir / "12.nii.gz"
    tio_image = tio.ScalarImage(img_path)
    print(tio_image.data)
    
    colin = tio.datasets.Colin27()
    print(colin)
    
#test_nibabel()
#test_masked_image()    
#test_nibabel_orient()
#test_nipy()
test_torchio()
#