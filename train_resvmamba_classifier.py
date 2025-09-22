import math
import numpy as np
import nibabel as nib

import random
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

from addict import Dict
from datasets import load_dataset
from datasets import Dataset

from enum import Enum

from functools import partial
from nibabel.filebasedimages import FileBasedImage
from pathlib import Path

from resvmamba.resvmamba3d import ResVMamba3dClassifierModel, make_default_config, ArcFace

from scipy import ndimage

from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

device='cuda'

def is_number_start(s):
    return bool(re.match(r'^\d', s))

def init_something():
    torch.device(device)
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

def interval_mapping(image, from_min=-90., from_max=1900., to_min=0., to_max=1.0):
    image_cliped = np.clip(image, from_min, from_max)
    from_range = from_max - from_min
    to_range = to_max - to_min
    scaled = np.array((image_cliped - from_min) / float(from_range))
    return (scaled * to_range) + to_min

def get_croped_Nifti(img:FileBasedImage, s=[32,32,32],l=[64,64,64]):
    # cropped_img = img.slicer[:,            #s[0]:s[0]+l[0],
    #                          (s[1]+l[1]-64  if l[1] >= 64 else s[1]):s[1]+l[1],  # 这两个纬度耍技巧？
    #                          (s[2]+l[2]-64  if l[2] >= 64 else s[2]):s[2]+l[2]]  #slicer是nibabel提供的接口.
    cropped_img = img
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

def get_masked_image(i_path,i_seg_path):
    # 
    img = nib.load(i_path)  # 替换为你的影像路径
    img = nib.as_closest_canonical(img)
    
    mask = nib.load(i_seg_path)  # 替换为你的分割掩码路径    
    mask = nib.as_closest_canonical(mask)
    # 获取数据数组
    img_data = img.get_fdata()
    mask_data = mask.get_fdata()

    # 确保影像和掩码形状相同
    assert img_data.shape == mask_data.shape, "影像和掩码形状不匹配！"
    
    ##对数据作的处理:
    ##缩放到0-1范围！
    n_img_data = interval_mapping(img_data)    
    # 应用掩码：保留分割区域 (mask>0)，其余置0
    masked_data = n_img_data * np.where(mask_data > 0, 1.0, 0.0)    
    #TODO;TOSEE
    sdf_data = ndimage.distance_transform_edt(mask_data)

    #裁减
    
    s, l = get_bb_slice(mask_array=mask_data)
    #这样折腾,据说能和affine变换兼容！
    masked_img = nib.Nifti1Image(masked_data, img.affine, img.header)
    croped_masked_img = get_croped_Nifti(masked_img, s=s,l=l)
    sdf_img = nib.Nifti1Image(sdf_data, img.affine, img.header)
    croped_sdf_img = get_croped_Nifti(sdf_img, s=s,l=l)
    
    return  croped_masked_img, croped_sdf_img

def make_parque(source_dir:Path =Path("/home/archie/Projects/data/TMJ"), cls="train", parque_size = 100):
    #找到原始数据!
    stems=[ stl_item 
           for stl_item 
           in source_dir.glob(f"{cls}/**/*.nii.gz") 
           if is_needed_item(stl_item)]
    random.shuffle(stems)
    
    class_labels = {
        'N':0,
        'D':1,
        'E':1,
        'F':1,
        'S':1,
    }
    class_num = 2

    def get_name(stem:Path):
        return stem.stem.split(".")[-2]
    
    def get_seg_stem(stem:Path):
        temp = list(stem.parts)
        temp[-2] = temp[-2] + "_seg"
        new_stem = Path(*temp) 
        return new_stem       
    
    def get_class_label_from_stem(stem:Path):
        class_name = stem.parts[-2].removeprefix("TYPE ")
        return class_name
    
    def get_onehot_for_the_class(class_name, class_num=class_num):
        def one_hot(x):
            return np.take(np.eye(class_num), x, axis=0)
        
        x = class_labels[class_name]
        return one_hot(x)
        

    parque_num = math.ceil(len(stems)/parque_size)
    for i in range(parque_num):
        stems_part = stems[parque_size*i:parque_size*(i+1)]
        masked_img_data, sdf, name, label = [], [], [], []
        sdf = []   
        for stem in stems_part:
            seg_stem = get_seg_stem(stem)
            
            masked_img, sdf_img = get_masked_image(stem, seg_stem)  # 
            
            masked_img_data.append(masked_img.get_fdata())
            sdf.append(sdf_img.get_fdata())
            
            name.append(get_name(stem))
            
            cls_name = get_class_label_from_stem(stem)
            cls_onehot= get_onehot_for_the_class(cls_name)
            label.append(cls_onehot)

        
        d = Dataset.from_dict({
            "masked_img_data":masked_img_data, 
            "sdf": sdf,
            "name":name, 
            "label":label
        })

        out_dir = source_dir / "parque" / f"{cls}-part-{i:04}-of-{parque_num:04}.parquet"
        d.to_parquet(out_dir)
                
def jk_padding(x:torch.Tensor):
    # i, j, k = x.shape
    # padding = (
    #     64-k,0,
    #     64-j,0,
    # )
    # x = F.pad(x, pad=padding, value=0.0)
    return x 

def collate_fn(batch, device, is_train=True):
    # "coord":coord_c, "feat":feat_c,"label":label_c,"shape_weight":shape_weight_c,"offset":offset_c,"name":name_c
    masked_img_data, label_data, sdf_data ,name_data = [],[],[],[]
    for example in batch:
        masked_img = example['masked_img_data']
        masked_img = jk_padding(masked_img).unsqueeze(0)
        #mask= example['mask_data'].unsqueeze(0)
        label=example['label'].unsqueeze(0)
        
        sdf =example['sdf']
        sdf = jk_padding(sdf).unsqueeze(0)

        if is_train :
            shift = np.random.randint(-16, 16,(1,))
            masked_img = torch.roll(masked_img,shifts=shift[0],dims = 1)
            sdf = torch.roll(sdf,shifts=shift[0], dims = 1)

        masked_img_data.append(masked_img)
        label_data.append(label)
        sdf_data.append(sdf)
        name_data.append(example["name"])

    masked_img_data = torch.cat(masked_img_data).cuda(device)
    label_data = torch.cat(label_data).cuda(device)
    sdf_data = torch.cat(sdf_data).cuda(device)

    
    return Dict(masked_img_data=masked_img_data,
                label=label_data,
                sdf=sdf_data,
                name=name_data)    

collate_fn_train = partial(collate_fn, device=device, is_train=True)
collate_fn_test = partial(collate_fn, device=device, is_train=False)

def data_loader(source_dir:Path =Path("/home/archie/Projects/data/TMJ"), 
                for_what = "parque",
                split="train",
                batch_size=4, 
                format="torch",
                is_eval = False):
    data_dir = source_dir / for_what
    ds = load_dataset(str(data_dir), split=split)
    ds.set_format(format)
    if not is_eval: 
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_train)
    else:
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_test) 
    return loader

def load_model():
    #
    init_something()
    #
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('resvmamba3d_checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    checkpoints_file = checkpoints_dir.joinpath('model_weights.pth')
    #
    config = make_default_config()
    model = ResVMamba3dClassifierModel(config)
    if checkpoints_file.exists():
        model.load_state_dict(torch.load(checkpoints_file))
        print("Load a saved model")
        
    model = model.to(device)
    return model, checkpoints_file
        
def train(epoches = 200,T_max:int=36,batch_size = 4,learning_rate = 0.0001): # milestones = [100,],

    model, checkpoints_file= load_model()
    
    #Data    
    tra_d_l = data_loader(split="train",batch_size=batch_size)
    test_d_l = data_loader(split="test",batch_size=batch_size, is_eval=True)
    #ttt_d_l = data_loader(split="train",batch_size=1, is_eval=False)

    #Loss and optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1080,], gamma=0.2)
    def train_ep(epoch:int,
                 data_loader:DataLoader,
                 model:nn.Module = model,
                 optimizer:torch.optim.Optimizer = optimizer,
                 scheduler:torch.optim.lr_scheduler.LRScheduler = scheduler):
        model = model.train()
        loss_epoch = []
        correct = []
        with tqdm(data_loader) as t:
            for batch, data in enumerate(t):
                optimizer.zero_grad()
                b_data = torch.cat([data["masked_img_data"].unsqueeze(1).to(device),
                                    data["sdf"].unsqueeze(1).to(device)], dim=1)
                labels = data["label"].to(device)                

                emb, loss, correct_mask = model(b_data, labels)

                loss.backward()                
                optimizer.step()
                loss_epoch.append(loss.item())
                correct.extend(correct_mask.cpu().numpy())
                t.set_description(f"Train: Epoch {epoch+1}/{epoches} Batch {batch+1}  Train_Loss:{loss.item():6f} LearningRate:{optimizer.param_groups[0]['lr']:8f}")
        scheduler.step()
        mean_loss = np.mean(loss_epoch)
        print(f"|- Epoch_{epoch+1}/{epoches} Train's meam_loss:{mean_loss}")
        uni, counts = np.unique(correct, return_counts=True)
        print(dict(zip(uni, counts))) 
                
    def eval_ep(epoch:int,
                data_loader:DataLoader,
                model:nn.Module = model, 
                ):
        model=model.eval()
        loss_epoch = []
        correct = []
        with torch.no_grad():        
            with tqdm(data_loader) as t:
                for batch,data in enumerate(t):
                    b_data = torch.cat([data["masked_img_data"].unsqueeze(1).to(device),
                                        data["sdf"].unsqueeze(1).to(device)], dim=1)
                    labels = data["label"].to(device)
                  
                    emb, loss, correct_mask = model(b_data, labels)

                    loss_epoch.append(loss.item())
                    correct.extend(correct_mask.cpu().numpy())

                    t.set_description(f"Eval: Epoch {epoch+1}/{epoches}: Batch {batch+1} Loss:{loss}") #{labels_cls} {preds}
        mean_loss = np.mean(loss_epoch)
        print(f"|- Epoch_{epoch+1}/{epoches} Eval's meam_loss:{mean_loss}")
        uni, counts = np.unique(correct, return_counts=True)
        print(dict(zip(uni, counts))) 
        
    #        
    for epoch in range(epoches):
        train_ep(epoch,
                 data_loader=tra_d_l,
                 model=model, 
                 optimizer=optimizer, 
                 scheduler=scheduler)
        # eval_ep(epoch,
        #         data_loader=test_d_l,
        #         model=model,)

        torch.save(model.state_dict(), checkpoints_file)        
        torch.cuda.empty_cache()

    # make_a_emds_parque(tra_d_l, model,split="train")    
    # make_a_emds_parque(test_d_l, model,split="test")
    
def make_a_emds_parque(data_loader, 
                    model:ResVMamba3dClassifierModel,
                    for_what:str= "parque_for_embs",
                    source_dir:Path =Path("/home/archie/Projects/data/TMJ"),
                    split="train"): 
    model=model.eval()
    feats, labels, names = [], [], []
    
    with torch.no_grad():        
        with tqdm(data_loader) as t:
            for _ ,data in enumerate(t):
                b_data = torch.cat([data["masked_img_data"].unsqueeze(1).to(device),
                                    data["sdf"].unsqueeze(1).to(device)], dim=1)
                feat, _, _ = model(b_data)
                feats.append(feat)
                                    
                label = data["label"].to(device)
                _ , cls = torch.max(label,dim=1)
                labels.append(cls)
                names.extend(data["name"])
    feats = torch.cat(feats)
    labels = torch.cat(labels)
    print(feats.shape, labels.shape)
    print(names)
    d = Dataset.from_dict({
        "feat":feats, 
        "label":labels,
        "name": names
    })
    out_dir = source_dir / for_what / f"{split}.parquet"
    d.to_parquet(out_dir)

        

def test_cosine_similarity():
    # model, _ = load_model()
    # test_d_l = data_loader(split="test",batch_size=5, is_eval=True)
    embs_train_d_l = load_dataset(str(Path("/home/archie/Projects/data/TMJ/parque_for_embs/")),split = "train")
    embs_train_d_l.set_format('torch')
    embs_test_d_l = load_dataset(str(Path("/home/archie/Projects/data/TMJ/parque_for_embs/")),split = "test")
    embs_test_d_l.set_format('torch')
    
    test, train = embs_test_d_l["feat"], embs_train_d_l["feat"]
    cos_sim = F.linear(F.normalize(test), F.normalize(train))
    sim, indices_of_target = torch.max(cos_sim, dim=-1)

    train_labels = embs_train_d_l["label"]
    test_labels = embs_test_d_l["label"]

    preds = train_labels[indices_of_target]
    correct = (preds == test_labels)
    uni, counts = np.unique(correct, return_counts=True)
    print(dict(zip(uni, counts)))
    
    indices_of_non_normal_in_test = torch.where(embs_test_d_l["label"] > 0)
    indices_of_normal_in_test = torch.where(embs_test_d_l["label"] == 0)
    print(indices_of_non_normal_in_test)
    
    test_non_normal = test[indices_of_non_normal_in_test]
    test_normal = test[indices_of_normal_in_test]
    cos_sim = F.linear(F.normalize(test_normal), F.normalize(test_non_normal))
    print(cos_sim[0])
    
    print(embs_test_d_l["name"][indices_of_normal_in_test[0][0]])
    print(embs_test_d_l["name"][indices_of_non_normal_in_test[0][1]])
    
def test_knn():
    embs_train_d_l = load_dataset(str(Path("/home/archie/Projects/data/TMJ/parque_for_embs/")),split = "train")
    embs_train_d_l.set_format('numpy')
    embs_test_d_l = load_dataset(str(Path("/home/archie/Projects/data/TMJ/parque_for_embs/")),split = "test")
    embs_test_d_l.set_format('numpy')
    from sklearn.neighbors import KNeighborsClassifier
    neigh = KNeighborsClassifier(n_neighbors=2)
    neigh.fit(embs_train_d_l["feat"], embs_train_d_l["label"])
    
    preds = neigh.predict(embs_test_d_l["feat"])
    print(preds)
    test_labels = embs_test_d_l["label"]
    correct = (preds == test_labels)
    #print(correct)
    #print(embs_test_d_l["name"])
    uni, counts = np.unique(correct, return_counts=True)
    print(dict(zip(uni, counts)))    

# make_parque(cls='train')
# make_parque(cls='test')
train(epoches=1080, T_max=18, batch_size=6, learning_rate=0.01 )   # 
# test_cosine_similarity()
#test_knn()















"""
---------------------------------看来没用！---------------------------------    
    make_a_svm_parque(tra_d_l, model=model, for_what="parque_for_svm",split="train")
    make_a_svm_parque(test_d_l, model=model, for_what="parque_for_svm",split="test")

def make_a_svm_parque(data_loader, 
                    model:ResVMamba3dClassifierModel,
                    for_what:str= "parque_for_svm",
                    source_dir:Path =Path("/home/archie/Projects/data/TMJ"),
                    split="train"): 
    model=model.eval()
    feats, labels = [], []
    
    with torch.no_grad():        
        with tqdm(data_loader) as t:
            for _ ,data in enumerate(t):
                b_data = torch.cat([data["masked_img_data"].unsqueeze(1).to(device),
                                    data["mask_data"].unsqueeze(1).to(device)], 
                                    dim=1)
                _ , feat = model(b_data)
                feats.append(feat)
                                    
                label = data["label"].to(device)
                _ , cls = torch.max(label,dim=1)
                labels.append(cls)
    feats = torch.cat(feats)
    labels = torch.cat(labels)
    print(feats.shape, labels.shape)
    
    d = Dataset.from_dict({
        "feat":feats, 
        "label":labels
    })
    out_dir = source_dir / for_what / f"{split}.parquet"
    d.to_parquet(out_dir)

def train_svm():    
    source_dir:Path =Path("/home/archie/Projects/data/TMJ")
    for_what="parque_for_svm"
    data_dir = source_dir / for_what
    split = "train"
    ds_train = load_dataset(str(data_dir), split=split)
    ds_train.set_format("numpy")

    X = ds_train["feat"]
    y = ds_train["label"]
    print(X[0])
    print(y[0])
    model = svm.SVC(kernel='rbf', C=10.0) #, ,gamma=0.1)
    model.fit(X, y)

    split = "test"
    ds_test = load_dataset(str(data_dir), split=split)
    ds_test.set_format("numpy")

    # Evaluate the predictions
    Xt = ds_test["feat"]
    yt = ds_test["label"]
    predictions = model.predict(Xt)
    print(predictions[:20])
    print(yt[:20])
    accuracy = model.score(Xt, yt)
    print("Accuracy of SVM:", accuracy)
""" 