## Allows to import .nii files as pure numpy voxel grids, Open3D pointclouds, and Open3D voxel grids.

import math
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import open3d as o3d
import random


from datasets import load_dataset
from nibabel.filebasedimages import FileBasedImage
from nibabel.orientations import axcodes2ornt, ornt_transform
from pathlib import Path
from scipy import ndimage
from skimage.measure import marching_cubes 
from typing import List, Optional, Tuple



source_dir = Path("/home/archie/Projects/data/TYPE D")

import re
def is_number_start(s):
    return bool(re.match(r'^\d', s))

def get_name(stem:Path):
    return stem.stem.split(".")[-2]

def get_seg_stem(stem:Path):
    temp = list(stem.parts)
    temp[-2] = temp[-2] + "_seg"
    new_stem = Path(*temp) 
    return new_stem

def is_needed_item(stl_item:Path):
    if is_number_start(stl_item.parts[-1]):
        return False
    if stl_item.parts[-2].rfind("_seg") < 0 :
        nii_img = nib.load(stl_item)
        sx, sy, sz = nii_img.header.get_zooms()
        if (sx*sy*sz) < 0.245**3 :
            return False          # 尺度为0.125的，也过滤调！
        if (sx*sy*sz) > 0.255**3: 
            return False          # 尺度为0.5以上的，也过滤调！
        return True
    else:
        return False

def get_croped_Nifti(img:FileBasedImage, s=[32,32,32],l=[64,64,64]):
    cropped_img = img.slicer[:,                   #s[0]:s[0]+l[0],
                             (s[1]+l[1]-64  if l[1] >= 64 else s[1]):s[1]+l[1],  # 这两个纬度耍技巧？
                             (s[2]+l[2]-64  if l[2] >= 64 else s[2]):s[2]+l[2]]  #slicer是nibabel提供的接口.
    return cropped_img

def get_bb_slice(mask_array):
    labeled_image, num_features = ndimage.label(mask_array)
    assert num_features == 1, "Labels error!"
    ttt = ndimage.find_objects(labeled_image)

    s_i = ttt[0][0].start
    s_j = ttt[0][1].start
    s_k = ttt[0][2].start

    l_i = ttt[0][0].stop - ttt[0][0].start
    l_j = ttt[0][1].stop - ttt[0][1].start
    l_k = ttt[0][2].stop - ttt[0][2].start
    
    s = [s_i,s_j,s_k]
    l = [l_i,l_j,l_k]
    return s, l
    
def make_data_to_observe(source_dir:Path =Path("/home/archie/Projects/data/TMJ"), 
                         cls="train",
                         dest_dir:Path = Path("/home/archie/Projects/data/TMJ/observe")):
    #找到原始数据!他们有特定的结构！
    file_path_list = source_dir.glob(f"{cls}/**/*.nii.gz")
    stems=[ stl_item for stl_item in  file_path_list if is_needed_item(stl_item)]
    
    random.shuffle(stems)
    

    
    def get_seg_stem(stem:Path):
        temp = list(stem.parts)
        temp[-2] = temp[-2] + "_seg"
        new_stem = Path(*temp) 
        return new_stem       

    # def get_croped_Nifti(img, s=16,l=96):
    #     cropped_img = img.slicer[s:s+l,s:s+l,s:s+l]  #slicer是nibabel提供的接口.
    #     return cropped_img
                    
    def get_masked_image(i_path,i_seg_path):
        img = nib.load(i_path)  # 替换为你的影像路径
        mask = nib.load(i_seg_path)  # 替换为你的分割掩码路径
        #
        mask_data = mask.get_fdata()

        s, l = get_bb_slice(mask_array=mask_data)
        #ToSee
        img = get_croped_Nifti(img, s, l)
        mask = get_croped_Nifti(mask, s, l)
        print(s)
        print(l)
        # 获取数据数组
        img_data = img.get_fdata()
        mask_data = mask.get_fdata()

        # 确保影像和掩码形状相同
        assert img_data.shape == mask_data.shape, "影像和掩码形状不匹配！"
                
        # 应用掩码：保留分割区域 (mask>0)，其余置0
        masked_data = img_data * (mask_data > 0)

        # 创建新的NIfTI图像（保留原始头信息和仿射矩阵）
        n_img = nib.Nifti1Image(img_data, img.affine, img.header)
        masked_img = nib.Nifti1Image(masked_data, img.affine, img.header)
        
        masked_img = nib.as_closest_canonical(masked_img)
        n_img = nib.as_closest_canonical(n_img)
        mask = nib.as_closest_canonical(mask)
        return  n_img, mask, masked_img, l
    ll = []        
    for stem in stems:
        name = get_name(stem)
        seg_stem = get_seg_stem(stem)
        print(name)        
        img, mask, masked_img, l = get_masked_image(stem, seg_stem)  # 此三者除了data，affine & header 应当是一样的！

        # nib.save(img, str(dest_dir/cls/f"{name}.nii.gz"))  # 指定输出文件名
        # nib.save(mask, str(dest_dir/cls/f"{name}_seg.nii.gz"))
        # nib.save(masked_img, str(dest_dir/cls/f"{name}_masked.nii.gz"))
        ll.append(l)
    print("========")
    print(np.max(np.asarray(ll), axis=0))
    print(np.min(np.asarray(ll), axis=0)) 

def interval_mapping(image, from_min, from_max, to_min, to_max):
    image_cliped = np.clip(image, from_min, from_max)
    from_range = from_max - from_min
    to_range = to_max - to_min
    scaled = np.array((image_cliped - from_min) / float(from_range))
    return (scaled * to_range) + to_min
       
def test_nibabel(source_dir=source_dir):
    img_path = source_dir / "12.nii.gz"
    img_voxels = nib.load(img_path)
    print(img_voxels.affine)
    # 2. 标准化方向到RAS 确保解剖方向一致：R(右), A(前), S(上)
    img_ras = nib.as_closest_canonical(img_voxels)
    data_ras = img_ras.get_fdata()
    data_ras = interval_mapping(data_ras, -90, 1900, 0., 1.)

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


def get_test_image(source_dir=source_dir):
    # 加载原始影像和分割掩码
    img_path = source_dir / "hu su qoing_LCo.nii.gz"
    img = nib.load(img_path)  # 替换为你的影像路径
    mask_path = source_dir / "seg-hu su qoing_LCo.nii.gz"
    mask = nib.load(mask_path)  # 替换为你的分割掩码路径
    return img, mask
    
def test_masked_image(source_dir=source_dir):
    img, mask = get_test_image(source_dir=source_dir)
    #ToSee
    img = get_croped_Nifti(img, s=[16,16,16], l=[96,96,96])
    mask = get_croped_Nifti(mask, s=[16,16,16], l=[96,96,96])
    
    # 获取数据数组
    img_data = img.get_fdata()
    mask_data = mask.get_fdata()

    # 确保影像和掩码形状相同
    assert img_data.shape == mask_data.shape, "影像和掩码形状不匹配！"

    # 应用掩码：保留分割区域 (mask>0)，其余置0
    masked_data = img_data * (mask_data > 0)
    print(type(masked_data), masked_data.shape)
    masked_data_s_0 = np.roll(masked_data, 8, axis=0)
    masked_data_s_1 = np.roll(masked_data, -8, axis=1)
    masked_data_s_2 = np.roll(masked_data, -8, axis=2)
    # 创建新的NIfTI图像（保留原始头信息和仿射矩阵）
    masked_img = nib.Nifti1Image(masked_data, img.affine, img.header)
    masked_img_s_0 = nib.Nifti1Image(masked_data_s_0, img.affine, img.header)
    masked_img_s_1 = nib.Nifti1Image(masked_data_s_1, img.affine, img.header)
    masked_img_s_2 = nib.Nifti1Image(masked_data_s_2, img.affine, img.header)

    masked_img = nib.as_closest_canonical(masked_img)
    masked_img_s_0 = nib.as_closest_canonical(masked_img_s_0)
    masked_img_s_1 = nib.as_closest_canonical(masked_img_s_1)
    masked_img_s_2 = nib.as_closest_canonical(masked_img_s_2)


    # 保存结果
    nib.save(masked_img, 'masked_image.nii.gz')  # 指定输出文件名
    nib.save(masked_img_s_0, 'masked_image_s_0.nii.gz')  # 指定输出文件名
    nib.save(masked_img_s_1, 'masked_image_s_1.nii.gz')  # 指定输出文件名
    nib.save(masked_img_s_2, 'masked_image_s_2.nii.gz')  # 指定输出文件名
    

            
def test_sdf(source_dir=source_dir):
    img, mask = get_test_image(source_dir=source_dir)
    
    mask_data = mask.get_fdata()
    s, l = get_bb_slice(mask_data)
    print(s)
    print(l)
    mask = get_croped_Nifti(mask, s=s, l=l)
    mask_data = mask.get_fdata()
    
    verts, faces, normals, values = marching_cubes(mask_data, spacing=(0.25, 0.25, 0.25))
    #print(verts)
    oral_scan = o3d.geometry.TriangleMesh()
    oral_scan.vertices  = o3d.utility.Vector3dVector(verts)
    oral_scan.triangles = o3d.utility.Vector3iVector(faces)
    #oral_scan.paint_uniform_color([1, 0.706, 0])
    #oral_scan.vertex_normals = o3d.utility.Vector3dVector(normals)
    oral_scan.compute_vertex_normals()
    oral_scan.filter_smooth_laplacian(number_of_iterations=100)
    # o3d.visualization.draw_geometries([oral_scan])
    o3d.io.write_triangle_mesh("xxx.stl", oral_scan)
    
def test_torchio(source_dir=source_dir):
    import torchio as tio
    
    img_path = source_dir / "12.nii.gz"
    tio_image = tio.ScalarImage(img_path)
    print(tio_image.data.device)
    

make_data_to_observe(cls='test')
make_data_to_observe(cls='train')
# test_nibabel()
# test_sdf()
#test_masked_image()    
#test_nibabel_orient()
#test_nipy()
#test_torchio()
#