"""
@Author: Tiange Xiang
@Contact: txia7609@uni.sydney.edu.au
@File: curvenet_seg.py
@Time: 2021/01/21 3:10 PM
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from pointnet2_ops import pointnet2_utils
from pm.utils.pointnet_common import knn,sample_and_group
from pm.utils.pointnet_common import PointNetFeaturePropagation
from .walk import Walk


class LPFA(nn.Module):
    def __init__(self, in_channel, out_channel, k, mlp_num=2, initial=False):
        super(LPFA, self).__init__()
        self.k = k
        self.device = torch.device('cuda')
        self.initial = initial

        if not initial:
            self.xyz2feature = nn.Sequential(
                        nn.Conv2d(9, in_channel, kernel_size=1, bias=False),
                        nn.BatchNorm2d(in_channel))

        self.mlp = []
        for _ in range(mlp_num):
            self.mlp.append(nn.Sequential(nn.Conv2d(in_channel, out_channel, 1, bias=False),
                                 nn.BatchNorm2d(out_channel),
                                 nn.LeakyReLU(0.2)))
            in_channel = out_channel
        self.mlp = nn.Sequential(*self.mlp)        

    def forward(self, x, xyz, idx=None):
        x = self.group_feature(x, xyz, idx)
        x = self.mlp(x)

        if self.initial:
            x = x.max(dim=-1, keepdim=False)[0]
        else:
            x = x.mean(dim=-1, keepdim=False)

        return x

    def group_feature(self, x, xyz, idx):
        batch_size, num_dims, num_points = x.size()

        if idx is None:
            idx = knn(xyz, k=self.k)[:,:,:self.k]  # (batch_size, num_points, k)

        idx_base = torch.arange(0, batch_size, device=self.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)

        xyz = xyz.transpose(2, 1).contiguous() # bs, n, 3
        point_feature = xyz.view(batch_size * num_points, -1)[idx, :]
        point_feature = point_feature.view(batch_size, num_points, self.k, -1)  # bs, n, k, 3
        points = xyz.view(batch_size, num_points, 1, 3).expand(-1, -1, self.k, -1)  # bs, n, k, 3

        point_feature = torch.cat((points, point_feature, point_feature - points),
                                dim=3).permute(0, 3, 1, 2).contiguous()

        if self.initial:
            return point_feature

        x = x.transpose(2, 1).contiguous() # bs, n, c
        feature = x.view(batch_size * num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, self.k, num_dims)  #bs, n, k, c
        x = x.view(batch_size, num_points, 1, num_dims)
        feature = feature - x

        feature = feature.permute(0, 3, 1, 2).contiguous()
        point_feature = self.xyz2feature(point_feature)  #bs, c, n, k
        feature = F.leaky_relu(feature + point_feature, 0.2)
        return feature #bs, c, n, k

class CIC(nn.Module):
    def __init__(self, npoint, radius, k, in_channels, output_channels, bottleneck_ratio=2, mlp_num=2, curve_config=None):
        super(CIC, self).__init__()
        self.in_channels = in_channels
        self.output_channels = output_channels
        self.bottleneck_ratio = bottleneck_ratio
        self.radius = radius
        self.k = k
        self.npoint = npoint

        planes = in_channels // bottleneck_ratio

        self.use_curve = curve_config is not None
        if self.use_curve:
            self.curveaggregation = CurveAggregation(planes)
            self.curvegrouping = CurveGrouping(planes, k, curve_config[0], curve_config[1])

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels,
                      planes,
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm1d(in_channels // bottleneck_ratio),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.conv2 = nn.Sequential(
            nn.Conv1d(planes, output_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(output_channels))

        if in_channels != output_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels,
                          output_channels,
                          kernel_size=1,
                          bias=False),
                nn.BatchNorm1d(output_channels))

        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.maxpool = MaskedMaxPool(npoint, radius, k)

        self.lpfa = LPFA(planes, planes, k, mlp_num=mlp_num, initial=False)

    def forward(self, xyz, x):
 
        # max pool
        if xyz.size(-1) != self.npoint:
            xyz, x = self.maxpool(
                xyz.transpose(1, 2).contiguous(), x)
            xyz = xyz.transpose(1, 2)

        shortcut = x
        x = self.conv1(x)  # bs, c', n

        idx = knn(xyz, self.k)

        if self.use_curve:
            # curve grouping
            curves = self.curvegrouping(x, xyz, idx[:,:,1:]) # avoid self-loop

            # curve aggregation
            x = self.curveaggregation(x, curves)

        x = self.lpfa(x, xyz, idx=idx[:,:,:self.k]) #bs, c', n, k

        x = self.conv2(x)  # bs, c, n

        if self.in_channels != self.output_channels:
            shortcut = self.shortcut(shortcut)

        x = self.relu(x + shortcut)

        return xyz, x


class CurveAggregation(nn.Module):
    def __init__(self, in_channel):
        super(CurveAggregation, self).__init__()
        self.in_channel = in_channel
        mid_feature = in_channel // 2
        self.conva = nn.Conv1d(in_channel,
                               mid_feature,
                               kernel_size=1,
                               bias=False)
        self.convb = nn.Conv1d(in_channel,
                               mid_feature,
                               kernel_size=1,
                               bias=False)
        self.convc = nn.Conv1d(in_channel,
                               mid_feature,
                               kernel_size=1,
                               bias=False)
        self.convn = nn.Conv1d(mid_feature,
                               mid_feature,
                               kernel_size=1,
                               bias=False)
        self.convl = nn.Conv1d(mid_feature,
                               mid_feature,
                               kernel_size=1,
                               bias=False)
        self.convd = nn.Sequential(
            nn.Conv1d(mid_feature * 2,
                      in_channel,
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm1d(in_channel))
        self.line_conv_att = nn.Conv2d(in_channel,
                                       1,
                                       kernel_size=1,
                                       bias=False)

    def forward(self, x, curves):
        curves_att = self.line_conv_att(curves)  # bs, 1, c_n, c_l

        curver_inter = torch.sum(curves * F.softmax(curves_att, dim=-1), dim=-1)  #bs, c, c_n
        curves_intra = torch.sum(curves * F.softmax(curves_att, dim=-2), dim=-2)  #bs, c, c_l

        curver_inter = self.conva(curver_inter) # bs, mid, n
        curves_intra = self.convb(curves_intra) # bs, mid ,n

        x_logits = self.convc(x).transpose(1, 2).contiguous()
        x_inter = F.softmax(torch.bmm(x_logits, curver_inter), dim=-1) # bs, n, c_n
        x_intra = F.softmax(torch.bmm(x_logits, curves_intra), dim=-1) # bs, l, c_l
        

        curver_inter = self.convn(curver_inter).transpose(1, 2).contiguous()
        curves_intra = self.convl(curves_intra).transpose(1, 2).contiguous()

        x_inter = torch.bmm(x_inter, curver_inter)
        x_intra = torch.bmm(x_intra, curves_intra)

        curve_features = torch.cat((x_inter, x_intra),dim=-1).transpose(1, 2).contiguous()
        x = x + self.convd(curve_features)

        return F.leaky_relu(x, negative_slope=0.2)


class CurveGrouping(nn.Module):
    def __init__(self, in_channel, k, curve_num, curve_length):
        super(CurveGrouping, self).__init__()
        self.curve_num = curve_num
        self.curve_length = curve_length
        self.in_channel = in_channel
        self.k = k

        self.att = nn.Conv1d(in_channel, 1, kernel_size=1, bias=False)

        self.walk = Walk(in_channel, k, curve_num, curve_length)

    def forward(self, x, xyz, idx):
        # starting point selection in self attention style
        x_att = torch.sigmoid(self.att(x))
        x = x * x_att

        _, start_index = torch.topk(x_att,
                                    self.curve_num,
                                    dim=2,
                                    sorted=False)
        start_index = start_index.squeeze().unsqueeze(2)

        curves = self.walk(xyz, x, idx, start_index)  #bs, c, c_n, c_l
        
        return curves


class MaskedMaxPool(nn.Module):
    def __init__(self, npoint, radius, k):
        super(MaskedMaxPool, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.k = k

    def forward(self, xyz, features):
        sub_xyz, neighborhood_features = sample_and_group(self.npoint, self.radius, self.k, xyz, features.transpose(1,2))

        neighborhood_features = neighborhood_features.permute(0, 3, 1, 2).contiguous()
        sub_features = F.max_pool2d(
            neighborhood_features, kernel_size=[1, neighborhood_features.shape[3]]
        )  # bs, c, n, 1
        sub_features = torch.squeeze(sub_features, -1)  # bs, c, n
        return sub_xyz, sub_features

curve_config = {
        'default': [[100, 5], [100, 5], None, None, None]
    }

class CurveNet(nn.Module):
    def __init__(self, num_classes=50, category=16, k=32, setting='default'):
        super(CurveNet, self).__init__()

        assert setting in curve_config

        additional_channel = 32
        self.lpfa = LPFA(9, additional_channel, k=k, mlp_num=1, initial=True)

        # encoder
        self.cic11 = CIC(npoint=2048, radius=0.2, k=k, in_channels=additional_channel, output_channels=64, bottleneck_ratio=2, curve_config=curve_config[setting][0])
        self.cic12 = CIC(npoint=2048, radius=0.2, k=k, in_channels=64, output_channels=64, bottleneck_ratio=4, curve_config=curve_config[setting][0])

        self.cic21 = CIC(npoint=512, radius=0.4, k=k, in_channels=64, output_channels=128, bottleneck_ratio=2, curve_config=curve_config[setting][1])
        self.cic22 = CIC(npoint=512, radius=0.4, k=k, in_channels=128, output_channels=128, bottleneck_ratio=4, curve_config=curve_config[setting][1])

        self.cic31 = CIC(npoint=128, radius=0.8, k=k, in_channels=128, output_channels=256, bottleneck_ratio=2, curve_config=curve_config[setting][2])
        self.cic32 = CIC(npoint=128, radius=0.8, k=k, in_channels=256, output_channels=256, bottleneck_ratio=4, curve_config=curve_config[setting][2])

        self.cic41 = CIC(npoint=32, radius=1.2, k=31, in_channels=256, output_channels=512, bottleneck_ratio=2, curve_config=curve_config[setting][3])
        self.cic42 = CIC(npoint=32, radius=1.2, k=31, in_channels=512, output_channels=512, bottleneck_ratio=4, curve_config=curve_config[setting][3])

        self.cic51 = CIC(npoint=8, radius=2.0, k=7, in_channels=512, output_channels=1024, bottleneck_ratio=2, curve_config=curve_config[setting][4])
        self.cic52 = CIC(npoint=8, radius=2.0, k=7, in_channels=1024, output_channels=1024, bottleneck_ratio=4, curve_config=curve_config[setting][4])
        self.cic53 = CIC(npoint=8, radius=2.0, k=7, in_channels=1024, output_channels=1024, bottleneck_ratio=4, curve_config=curve_config[setting][4])

        # decoder
        self.fp4 = PointNetFeaturePropagation(in_channel=1024 + 512, mlp=[512, 512], att=[1024, 512, 256])
        self.up_cic5 = CIC(npoint=32, radius=1.2, k=31, in_channels=512, output_channels=512, bottleneck_ratio=4)

        self.fp3 = PointNetFeaturePropagation(in_channel=512 + 256, mlp=[256, 256], att=[512, 256, 128])
        self.up_cic4 = CIC(npoint=128, radius=0.8, k=k, in_channels=256, output_channels=256, bottleneck_ratio=4)

        self.fp2 = PointNetFeaturePropagation(in_channel=256 + 128, mlp=[128, 128], att=[256, 128, 64])
        self.up_cic3 = CIC(npoint=512, radius=0.4, k=k, in_channels=128, output_channels=128, bottleneck_ratio=4)

        self.fp1 = PointNetFeaturePropagation(in_channel=128 + 64, mlp=[64, 64], att=[128, 64, 32])
        self.up_cic2 = CIC(npoint=2048, radius=0.2, k=k, in_channels=128+64+64+category+3, output_channels=256, bottleneck_ratio=4)
        self.up_cic1 = CIC(npoint=2048, radius=0.2, k=k, in_channels=256, output_channels=256, bottleneck_ratio=4)
        

        self.global_conv2 = nn.Sequential(
            nn.Conv1d(1024, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2))
        self.global_conv1 = nn.Sequential(
            nn.Conv1d(512, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2))

        self.conv1 = nn.Conv1d(256, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(256, num_classes, 1)
        self.se = nn.Sequential(nn.AdaptiveAvgPool1d(1),
                                nn.Conv1d(256, 256//8, 1, bias=False),
                                nn.BatchNorm1d(256//8),
                                nn.LeakyReLU(negative_slope=0.2),
                                nn.Conv1d(256//8, 256, 1, bias=False),
                                nn.Sigmoid())
                                
    def forward(self, xyz, l=None):
        batch_size = xyz.size(0)

        l0_points = self.lpfa(xyz, xyz)

        l1_xyz, l1_points = self.cic11(xyz, l0_points)
        l1_xyz, l1_points = self.cic12(l1_xyz, l1_points)

        l2_xyz, l2_points = self.cic21(l1_xyz, l1_points)
        l2_xyz, l2_points = self.cic22(l2_xyz, l2_points)

        l3_xyz, l3_points = self.cic31(l2_xyz, l2_points)
        l3_xyz, l3_points = self.cic32(l3_xyz, l3_points)
 
        l4_xyz, l4_points = self.cic41(l3_xyz, l3_points)
        l4_xyz, l4_points = self.cic42(l4_xyz, l4_points)

        l5_xyz, l5_points = self.cic51(l4_xyz, l4_points)
        l5_xyz, l5_points = self.cic52(l5_xyz, l5_points)
        l5_xyz, l5_points = self.cic53(l5_xyz, l5_points)

        # global features
        emb1 = self.global_conv1(l4_points)
        emb1 = emb1.max(dim=-1, keepdim=True)[0] # bs, 64, 1
        emb2 = self.global_conv2(l5_points)
        emb2 = emb2.max(dim=-1, keepdim=True)[0] # bs, 128, 1

        # Feature Propagation layers
        l4_points = self.fp4(l4_xyz, l5_xyz, l4_points, l5_points)
        l4_xyz, l4_points = self.up_cic5(l4_xyz, l4_points)

        l3_points = self.fp3(l3_xyz, l4_xyz, l3_points, l4_points)
        l3_xyz, l3_points = self.up_cic4(l3_xyz, l3_points)

        l2_points = self.fp2(l2_xyz, l3_xyz, l2_points, l3_points)
        l2_xyz, l2_points = self.up_cic3(l2_xyz, l2_points)

        l1_points = self.fp1(l1_xyz, l2_xyz, l1_points, l2_points)

        if l is not None:
            l = l.view(batch_size, -1, 1)
            emb = torch.cat((emb1, emb2, l), dim=1) # bs, 128 + 16, 1
        l = emb.expand(-1,-1, xyz.size(-1))
        x = torch.cat((l1_xyz, l1_points, l), dim=1)
        print(l1_xyz.shape)
        print(x.shape)
        xyz, x = self.up_cic2(l1_xyz, x)
        xyz, x = self.up_cic1(xyz, x)

        x =  F.leaky_relu(self.bn1(self.conv1(x)), 0.2, inplace=True)
        se = self.se(x)
        x = x * se
        x = self.drop1(x)
        x = self.conv2(x)
        return x

if __name__ == '__main__':

    data = torch.rand(2, 3, 2048).cuda()
    cls_label = torch.rand([2, 16]).cuda()
    print("===> testing modelD ...")
    model = CurveNet().cuda()
    out = model(data,cls_label)  # [2,2048,50]
    print(out.shape)