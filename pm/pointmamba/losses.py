"""
Misc Losses

"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from einops import rearrange,einsum
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Optional, Tuple
from pm.pointmamba.conifuguration_point_sis import  PointSISConfig

class CrossEntropyLoss(nn.Module):
    def __init__(
        self,
        weight=None,
        size_average=None,
        reduce=None,
        reduction="mean",
        label_smoothing=0.0,
        loss_weight=1.0,
        ignore_index=-1,
    ):
        super(CrossEntropyLoss, self).__init__()
        weight = torch.tensor(weight).cuda() if weight is not None else None
        self.loss_weight = loss_weight
        self.loss = nn.CrossEntropyLoss(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )

    def forward(self, pred, target):
        return self.loss(pred, target) * self.loss_weight

class SmoothCELoss(nn.Module):
    def __init__(self, smoothing_ratio=0.1):
        super(SmoothCELoss, self).__init__()
        self.smoothing_ratio = smoothing_ratio

    def forward(self, pred, target):
        eps = self.smoothing_ratio
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prb).total(dim=1)
        loss = loss[torch.isfinite(loss)].mean()
        return loss


class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.5, logits=True, reduce=True, loss_weight=1.0):
        """Binary Focal Loss
        <https://arxiv.org/abs/1708.02002>`
        """
        super(BinaryFocalLoss, self).__init__()
        assert 0 < alpha < 1
        self.gamma = gamma
        self.alpha = alpha
        self.logits = logits
        self.reduce = reduce
        self.loss_weight = loss_weight

    def forward(self, pred, target, **kwargs):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction with shape (N)
            target (torch.Tensor): The ground truth. If containing class
                indices, shape (N) where each value is 0≤targets[i]≤1, If containing class probabilities,
                same shape as the input.
        Returns:
            torch.Tensor: The calculated loss
        """
        if self.logits:
            bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        else:
            bce = F.binary_cross_entropy(pred, target, reduction="none")
        pt = torch.exp(-bce)
        alpha = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_loss = alpha * (1 - pt) ** self.gamma * bce

        if self.reduce:
            focal_loss = torch.mean(focal_loss)
        return focal_loss * self.loss_weight


class FocalLoss(nn.Module):
    def __init__(
        self, gamma=2.0, alpha=0.5, reduction="mean", loss_weight=1.0, ignore_index=-1
    ):
        """Focal Loss
        <https://arxiv.org/abs/1708.02002>`
        """
        super(FocalLoss, self).__init__()
        assert reduction in (
            "mean",
            "sum",
        ), "AssertionError: reduction should be 'mean' or 'sum'"
        assert isinstance(
            alpha, (float, list)
        ), "AssertionError: alpha should be of type float"
        assert isinstance(gamma, float), "AssertionError: gamma should be of type float"
        assert isinstance(
            loss_weight, float
        ), "AssertionError: loss_weight should be of type float"
        assert isinstance(ignore_index, int), "ignore_index must be of type int"
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target, **kwargs):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction with shape (N, C) where C = number of classes.
            target (torch.Tensor): The ground truth. If containing class
                indices, shape (N) where each value is 0≤targets[i]≤C−1, If containing class probabilities,
                same shape as the input.
        Returns:
            torch.Tensor: The calculated loss
        """
        # [B, C, d_1, d_2, ..., d_k] -> [C, B, d_1, d_2, ..., d_k]
        pred = pred.transpose(0, 1)
        # [C, B, d_1, d_2, ..., d_k] -> [C, N]
        pred = pred.reshape(pred.size(0), -1)
        # [C, N] -> [N, C]
        pred = pred.transpose(0, 1).contiguous()
        # (B, d_1, d_2, ..., d_k) --> (B * d_1 * d_2 * ... * d_k,)
        target = target.view(-1).contiguous()
        assert pred.size(0) == target.size(0), "The shape of pred doesn't match the shape of target"
        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        pred = pred[valid_mask]

        if len(target) == 0:
            return 0.0

        num_classes = pred.size(1)
        target = F.one_hot(target, num_classes=num_classes)

        alpha = self.alpha
        if isinstance(alpha, list):
            alpha = pred.new_tensor(alpha)
        pred_sigmoid = pred.sigmoid()
        target = target.type_as(pred)
        one_minus_pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * one_minus_pt.pow(
            self.gamma
        )

        loss = (
            F.binary_cross_entropy_with_logits(pred, target, reduction="none")
            * focal_weight
        )
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.total()
        return self.loss_weight * loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1, exponent=2, loss_weight=1.0, ignore_index=-1):
        """DiceLoss.
        This loss is proposed in `V-Net: Fully Convolutional Neural Networks for
        Volumetric Medical Image Segmentation <https://arxiv.org/abs/1606.04797>`_.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.exponent = exponent
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target, **kwargs):
        # [B, C, d_1, d_2, ..., d_k] -> [C, B, d_1, d_2, ..., d_k]
        pred = pred.transpose(0, 1)
        # [C, B, d_1, d_2, ..., d_k] -> [C, N]
        pred = pred.reshape(pred.size(0), -1)
        # [C, N] -> [N, C]
        pred = pred.transpose(0, 1).contiguous()
        # (B, d_1, d_2, ..., d_k) --> (B * d_1 * d_2 * ... * d_k,)
        target = target.view(-1).contiguous()
        assert pred.size(0) == target.size(
            0
        ), "The shape of pred doesn't match the shape of target"
        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        pred = pred[valid_mask]

        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        target = F.one_hot(
            torch.clamp(target.long(), 0, num_classes - 1), num_classes=num_classes
        )

        total_loss = 0
        for i in range(num_classes):
            if i != self.ignore_index:
                num = torch.sum(torch.mul(pred[:, i], target[:, i])) * 2 + self.smooth
                den = (
                    torch.sum(
                        pred[:, i].pow(self.exponent) + target[:, i].pow(self.exponent)
                    )
                    + self.smooth
                )
                dice_loss = 1 - num / den
                total_loss += dice_loss
        loss = total_loss / num_classes
        return self.loss_weight * loss
    


teeth = {17,16,15,14,13,12,11,
         27,26,25,24,23,22,21,
         37,36,35,34,33,32,31,
         47,46,45,44,43,42,41}

#各种辅助！！ 
def tooth_lables(labels:torch.Tensor) -> List[torch.Tensor]: # b g -> [t,...], [t g,...]  
    b_s = labels.shape[0]
    b_class_labels = []
    b_mask_labels  = []
    for b in range(b_s):
        masks = []
        for i in teeth:
            x = torch.where(labels[b]==i,1,0)
            if x.sum() > 0 :
                x = x.unsqueeze(0)
                masks.append(x)
        num_target = len(masks)   # TODO:需要检查 num_target>0
        class_labels = torch.ones(num_target, device=labels.device).long()       #  t        # TODO: t个目标, 每个目标都是标签1
        b_class_labels.append(class_labels)
        mask_labels = torch.cat(masks, dim=0).float()                             # t g      # TODO: 每个目标, 都有一个掩码！
        b_mask_labels.append(mask_labels)
    return b_class_labels, b_mask_labels

def sigmoid_cross_entropy_loss(inputs: torch.Tensor, labels: torch.Tensor, num_masks: int) -> torch.Tensor:
    r"""
    Args:
        inputs (`torch.Tensor`):
            A float tensor of arbitrary shape.
        labels (`torch.Tensor`):
            A tensor with the same shape as inputs. Stores the binary classification labels for each element in inputs
            (0 for the negative class and 1 for the positive class).

    Returns:
        loss (`torch.Tensor`): The computed loss.
    """
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    cross_entropy_loss = criterion(inputs, labels)

    loss = cross_entropy_loss.mean(1).sum() / num_masks
    return loss

# Copied from transformers.models.maskformer.modeling_maskformer.dice_loss
def dice_loss(inputs: Tensor, labels: Tensor, num_masks: int) -> Tensor:
    r"""
    Args:
        inputs (`torch.Tensor`):
            A tensor representing a mask.
        labels (`torch.Tensor`):
            A tensor with the same shape as inputs. Stores the binary classification labels for each element in inputs
            (0 for the negative class and 1 for the positive class).
        num_masks (`int`):
            The number of masks present in the current batch, used for normalization.

    Returns:
        `torch.Tensor`: The computed loss.
    """
    probs = inputs.sigmoid().flatten(1)
    numerator = 2 * (probs * labels).sum(-1)
    denominator = probs.sum(-1) + labels.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    loss = loss.sum() / num_masks
    return loss

# Copied from transformers.models.maskformer.modeling_maskformer.pair_wise_dice_loss
def pair_wise_dice_loss(inputs: Tensor, labels: Tensor) -> Tensor:
    """
    A pair wise version of the dice loss, see `dice_loss` for usage.
    """
    inputs = inputs.sigmoid().flatten(1)
    numerator = 2 * torch.matmul(inputs, labels.T)
    # using broadcasting to get a [num_queries, NUM_CLASSES] matrix
    denominator = inputs.sum(-1)[:, None] + labels.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss

def pair_wise_sigmoid_cross_entropy_loss(inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    r"""
    A pair wise version of the cross entropy loss, see `sigmoid_cross_entropy_loss` for usage.
    """

    length = inputs.shape[1]

    criterion = nn.BCEWithLogitsLoss(reduction="none")
    cross_entropy_loss_pos = criterion(inputs, torch.ones_like(inputs))
    cross_entropy_loss_neg = criterion(inputs, torch.zeros_like(inputs))

    loss_pos = torch.matmul(cross_entropy_loss_pos, labels.T)
    loss_neg = torch.matmul(cross_entropy_loss_neg, (1 - labels).T)
    loss = loss_pos + loss_neg
    loss = loss / length
    return loss    

class PMHungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 1.0, cost_mask: float = 1.0, cost_dice: float = 1.0):
        """Creates the matcher
        Params:
            cost_class:...            
            cost_mask :...                
            cost_dice :...
            三种loss的权重！
        """
        super().__init__()
        if cost_class == 0 and cost_mask == 0 and cost_dice == 0:
            raise ValueError("All costs cant be 0")

        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

    @torch.no_grad()
    def forward(
        self,
        masks_queries_logits: torch.Tensor,          # b q g
        class_queries_logits: torch.Tensor,          # b q l
        mask_labels: List[torch.Tensor],             # [t g,...],  len_of_list: b
        class_labels: List[torch.Tensor],            # [t,...] , len_of_list: b
    ) -> List[Tuple[Tensor]]:                        # [(t, t),...] , len_of_list: b
        """
        Returns:
            matched_indices (`List[Tuple[Tensor]]`): A list of size batch_size, containing tuples of (index_i, index_j)
            where:
                - index_i 是中选预测的索引
                - index_j 是对应标签的索引
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes).
        """
        indices: List[Tuple[np.array]] = []

        # iterate through batch size
        batch_size = masks_queries_logits.shape[0]
        for i in range(batch_size):
            pred_probs = class_queries_logits[i].softmax(-1)        # -> q l
            pred_mask = masks_queries_logits[i]                     # -> q g
            
            #Class cost
            # Compute the classification cost. Contrary to the loss, we don't use the NLL, but approximate it in 1 - proba[target class]. The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -pred_probs[:, class_labels[i]]               # 理解这里,class_labels[i]作为索引,所起的作用?  q l, t -> q t

            # Mask cost
            target_mask = mask_labels[i].to(pred_mask)                 # t g
            # compute the cross entropy loss between each mask pairs -> shape (num_queries, num_labels)
            cost_mask = pair_wise_sigmoid_cross_entropy_loss(pred_mask, target_mask)  # q g, t g -> q t

            # Dice loss
            # Compute the dice loss betwen each mask pairs -> shape (num_queries, num_labels)
            cost_dice = pair_wise_dice_loss(pred_mask, target_mask)                   # q g, t g -> q t
            # final cost matrix
            cost_matrix = self.cost_mask * cost_mask + self.cost_class * cost_class + self.cost_dice * cost_dice
            # 解决指派问题,用的是scipy里的实现！
            assigned_indices: Tuple[np.array] = linear_sum_assignment(cost_matrix.cpu())   # (t, t)
            indices.append(assigned_indices)

        # It could be stacked in one tensor
        matched_indices = [
            (torch.as_tensor(i, dtype=torch.long), torch.as_tensor(j, dtype=torch.long)) for i, j in indices
        ]
        return matched_indices
    

class PMLoss(nn.Module):
    def __init__(self, config: PointSISConfig):
        """
        The M2F Loss. The loss is computed very similar to DETR. The process happens in two steps: 1) we
        compute hungarian assignment between ground truth masks and the outputs of the model 2) we supervise each pair
        of matched ground-truth / prediction (supervise class and mask)

        Args:
            config (`M2FConfig`):
                The configuration for M2F model also containing loss calculation specific parameters.
            weight_dict (`Dict[str, float]`):
                A dictionary of weights to be applied to the different losses.
        """
        super().__init__()
        self.num_labels = config.num_labels
        self.weight_dict: Dict[str, float] = {
            "loss_cross_entropy": config.class_weight,
            "loss_mask": config.mask_weight,
            "loss_dice": config.dice_weight,
        }

        # Weight to apply to the null class
        self.eos_coef = config.no_object_weight
        empty_weight = torch.ones(self.num_labels + 1)
        empty_weight[0] = self.eos_coef                    # TODO: 到底是0,还是-1.
        #empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)


        self.matcher = PMHungarianMatcher(
            cost_class = config.class_weight,
            cost_dice  = config.dice_weight,
            cost_mask  = config.mask_weight,
        )

    def loss_labels(self, 
                    class_queries_logits: Tensor,     # b q l
                    class_labels: List[Tensor],       # [t,...]       len_of_list: b
                    indices: Tuple[np.array],         # [(t, t),...]  len_of_list: b
                    ) -> Dict[str, Tensor]: 
        """Compute the losses related to the labels using cross entropy.
        Returns:
            `Dict[str, Tensor]`: A dict of `torch.Tensor` containing the following key:
            - **loss_cross_entropy** -- The loss computed using cross entropy on the predicted and ground truth labels.
        """
        pred_logits = class_queries_logits
        batch_size, num_queries, _ = pred_logits.shape
        criterion = nn.CrossEntropyLoss(weight=self.empty_weight)
        # 下面这段,要注意各种索引技巧！！！
        # idx==(batch_indices, prediction_indices), shape为(t_0+t_1+...+t_(b-1), t_0+t_1+...+t_(b-1)).索引出了t_0+t_1+...+t_(b-1)个位置！
        idx = self._get_predictions_permutation_indices(indices)  
        target_classes_o = torch.cat([target[indices_tgt] for target, (_, indices_tgt) in zip(class_labels, indices)])  #  -> t_0+t_1+...+t_(b-1)
        # TODO: fill_value是0还是self.num_labels!
        target_classes = torch.full((batch_size, num_queries), fill_value=0, dtype=target_classes_o.dtype, device=pred_logits.device)  # b q
        target_classes[idx] = target_classes_o     # 将target_classes,在idx索引出的位置上,填入目标值！！！
        
        pred_logits_transposed = pred_logits.transpose(1, 2)         # b q l -> b l q
        loss_ce = criterion(pred_logits_transposed, target_classes)  # b l q, b q
        losses = {"loss_cross_entropy": loss_ce}

        del pred_logits
        del target_classes

        return losses

    def loss_masks(self,
        masks_queries_logits: torch.Tensor,    # b q g
        mask_labels: List[torch.Tensor],       # [t g,...]  len_of_list: b
        indices: Tuple[np.array],
        num_masks: int,
    ) -> Dict[str, torch.Tensor]:
        """Compute the losses related to the masks using sigmoid_cross_entropy_loss and dice loss.
        Returns:
            losses (`Dict[str, Tensor]`): A dict of `torch.Tensor` containing two keys:
            - **loss_mask** -- The loss computed using sigmoid cross entropy loss on the predicted and ground truth.
              masks.
            - **loss_dice** -- The loss computed using dice loss on the predicted on the predicted and ground truth,
              masks.
        """
        # src_idx==(batch_indices, prediction_indices), shape为(t_0+t_1+...+t_(b-1), t_0+t_1+...+t_(b-1)).索引出了t_0+t_1+...+t_(b-1)个位置！
        src_idx = self._get_predictions_permutation_indices(indices)
        # tgt_idx==(batch_indices, target_indices), shape为(t_0+t_1+...+t_(b-1), t_0+t_1+...+t_(b-1)).索引出了t_0+t_1+...+t_(b-1)个位置！
        tgt_idx = self._get_targets_permutation_indices(indices)
        #
        pred_masks = masks_queries_logits[src_idx]  # ->(t_0+t_1+...+t_(b-1), g)
        target_masks = torch.cat([target[target_indices] for target, (_, target_indices) in zip(mask_labels, indices)])
        # print(target_masks.shape)
        # target_masks = mask_labels[tgt_idx]         # 

        losses = {
            "loss_mask": sigmoid_cross_entropy_loss(pred_masks, target_masks, num_masks),
            "loss_dice": dice_loss(pred_masks, target_masks, num_masks),
        }

        del pred_masks
        del target_masks
        return losses

    def _get_predictions_permutation_indices(self, indices):
        # Permute predictions following indices
        # 
        batch_indices = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])   # t_0+t_1+...+t_(b-1) # 所谓对batch的索引
        predictions_indices = torch.cat([src for (src, _) in indices])                               # t_0+t_1+...+t_(b-1) # 
        return batch_indices, predictions_indices

    def _get_targets_permutation_indices(self, indices):
        # Permute labels following indices
        batch_indices = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        target_indices = torch.cat([tgt for (_, tgt) in indices])
        return batch_indices, target_indices

    def get_num_masks(self, 
                      class_labels: torch.Tensor,             # [t,...]    len_of_list: b
                      device: torch.device) -> torch.Tensor:
        """
        计算一批数据中，应当有多少个掩码.
        """
        num_masks = sum([len(classes) for classes in class_labels])
        num_masks_pt = torch.as_tensor([num_masks], dtype=torch.float, device=device)
        return num_masks_pt
    
    def forward(
        self,
        masks_queries_logits: torch.Tensor,                     # b q g
        class_queries_logits: torch.Tensor,                     # b q l
        labels              : torch.Tensor,                     # b g
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
            losses (`Dict[str, Tensor]`): A dict of `torch.Tensor` containing three keys:
            - **loss_cross_entropy** -- The loss computed using cross entropy on the predicted and ground truth labels.
            - **loss_mask** -- The loss computed using sigmoid cross_entropy loss on the predicted and ground truth
              masks.
            - **loss_dice** -- The loss computed using dice loss on the predicted on the predicted and ground truth
              masks.
            if `use_auxiliary_loss` was set to `true` in [`M2FConfig`], the dictionary contains additional
            losses for each auxiliary predictions.
        """
        class_labels, mask_labels = tooth_lables(labels)

        # retrieve the matching between the outputs of the last layer and the labels
        indices = self.matcher(masks_queries_logits, class_queries_logits, mask_labels, class_labels)
        # compute the average number of target masks for normalization purposes
        num_masks = self.get_num_masks(class_labels, device=class_labels[0].device)
        # get all the losses
        losses: Dict[str, Tensor] = {
            **self.loss_masks(masks_queries_logits, mask_labels, indices, num_masks),
            **self.loss_labels(class_queries_logits, class_labels, indices),
        }

        return losses


