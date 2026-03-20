"""Noise generation for Mask-Piloted training."""
import torch
import torch.nn as nn
from typing import List, Tuple


class MPNoiseGenerator(nn.Module):
    """Generate noisy ground-truth masks for MP-Former training."""

    def __init__(self, config):
        super().__init__()
        self.noise_ratio = config.mp_noise_ratio
        self.label_noise_ratio = config.mp_label_noise_ratio
        self.num_classes = config.num_labels

    def forward(
        self,
        mask_labels: List[torch.Tensor],
        class_labels: List[torch.Tensor],
        num_mp_queries: int,
        device: torch.device
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Generate noisy masks and classes for MP queries.

        Args:
            mask_labels: List of ground truth masks for each batch element [t_0, t_1, ...]
            class_labels: List of ground truth classes for each batch element [t_0, t_1, ...]
            num_mp_queries: Number of MP queries to generate per batch
            device: Device to create tensors on

        Returns:
            batch_noisy_masks: List of noisy masks for MP queries [(b*mp_num), g]
            batch_noisy_classes: List of noisy classes for MP queries [(b*mp_num),]
        """
        batch_noisy_masks = []
        batch_noisy_classes = []

        for masks, classes in zip(mask_labels, class_labels):
            num_gt = len(masks)

            # Sample indices for MP queries
            if num_gt <= num_mp_queries:
                # If fewer GT than MP queries, sample with replacement
                indices = torch.cat([
                    torch.arange(num_gt, device=device),
                    torch.randint(0, num_gt, (num_mp_queries - num_gt,), device=device)
                ])
            else:
                # Randomly sample from GT masks
                indices = torch.randperm(num_gt, device=device)[:num_mp_queries]

            sampled_masks = masks[indices]
            sampled_classes = classes[indices]

            # Apply point dropout noise
            noise_mask = torch.rand_like(sampled_masks.float()) > self.noise_ratio
            noisy_masks = sampled_masks * noise_mask.float()

            # Apply label flip noise
            if self.label_noise_ratio > 0:
                flip_mask = torch.rand(num_mp_queries, device=device) < self.label_noise_ratio
                noisy_classes = sampled_classes.clone()
                if flip_mask.sum() > 0:
                    # Flip to random valid class (1 to num_classes)
                    noisy_classes[flip_mask] = torch.randint(
                        1, self.num_classes + 1, (flip_mask.sum(),), device=device
                    )
            else:
                noisy_classes = sampled_classes

            batch_noisy_masks.append(noisy_masks)
            batch_noisy_classes.append(noisy_classes)

        return batch_noisy_masks, batch_noisy_classes
