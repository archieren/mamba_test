"""Convert noisy masks to query embeddings."""
import torch
import torch.nn as nn
from einops import rearrange, einsum
from typing import List


class MPQueryGenerator(nn.Module):
    """Convert noisy GT masks into query embeddings."""

    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.num_classes = config.num_labels
        self.class_embed = nn.Embedding(self.num_classes + 1, config.d_model)

    def forward(
        self,
        noisy_masks: List[torch.Tensor],
        noisy_classes: List[torch.Tensor],
        mask_features: torch.Tensor
    ) -> torch.Tensor:
        """Generate query embeddings from noisy masks.

        Args:
            noisy_masks: List of noisy masks for each batch element [(mp_num), g]
            noisy_classes: List of noisy classes for each batch element [(mp_num),]
            mask_features: Feature tensor from encoder [b, g, d]

        Returns:
            query_emb: MP query embeddings [(b*mp_num), d]
        """
        b = mask_features.shape[0]
        g, d = mask_features.shape[1], mask_features.shape[2]

        # Concatenate all MP queries from batch
        all_masks = torch.cat(noisy_masks, dim=0)  # (b*mp_num) g
        all_classes = torch.cat(noisy_classes, dim=0)  # (b*mp_num)

        # Aggregate features using mask as attention weights
        # all_masks: (b*mp_num) g, mask_features: b g d
        # We need to compute weighted average of features for each mask

        # Split mask_features by batch and compute queries per batch element
        query_embs = []
        mask_idx = 0

        for batch_idx in range(b):
            # Get masks for this batch element
            batch_mp_queries = len(noisy_masks[batch_idx])
            batch_masks = all_masks[mask_idx:mask_idx + batch_mp_queries]  # (mp_num) g

            # Aggregate features using mask as attention weights
            # Normalize masks to sum to 1 for each query
            mask_weights = batch_masks / (batch_masks.sum(dim=-1, keepdim=True) + 1e-8)  # (mp_num) g

            # Compute weighted average of features
            query_emb = einsum(mask_weights, mask_features[batch_idx],
                              "n g, g d -> n d")  # (mp_num) d

            query_embs.append(query_emb)
            mask_idx += batch_mp_queries

        query_emb = torch.cat(query_embs, dim=0)  # (b*mp_num) d

        # Add class embedding
        class_emb = self.class_embed(all_classes)  # (b*mp_num) d
        query_emb = query_emb + class_emb

        return query_emb
