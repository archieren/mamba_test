# MP-Former Training Implementation Summary

## Overview

This document describes the implementation of **Mask-Piloted (MP-Former) training** for the PointSIS point cloud segmentation model. MP-Former is a training technique that injects noisy ground-truth masks as additional queries during training, providing enhanced supervision signals.

## Expected Benefits

- **Accuracy**: +1-2 mIoU improvement
- **Training Speed**: ~50% faster convergence to same accuracy
- **Training Overhead**: ~10-15% increase in memory usage
- **Inference Overhead**: Zero (MP queries only used during training)

## Architecture Changes

### Training Mode Flow

```
Ground Truth Masks + Classes
    ↓
Noise Injection (MPNoiseGenerator)
    ↓
MP Query Generator (MPQueryGenerator)
    ↓
6 Additional MP Queries per Batch
    ↓
Concatenate with 24 Original Queries = 30 Total
    ↓
MaskDecoder with Enhanced Supervision
    ↓
    ├─→ Original Queries (24) → Hungarian Matching → Loss
    └─→ MP Queries (6) → Direct Supervision → Loss (No Matching)
```

### Inference Mode Flow

```
Original 24 Queries Only (MP Queries Disabled)
    ↓
MaskDecoder
    ↓
Predictions (Same as Before)
```

## Implementation Details

### Files Modified

1. **`pm/pointmamba/configuration_point_sis.py`**
   - Added MP training configuration parameters:
     - `use_mp_training`: Enable/disable MP training (default: False)
     - `mp_num_queries`: Number of MP queries per batch (default: 6)
     - `mp_noise_ratio`: Point dropout noise ratio (default: 0.2)
     - `mp_label_noise_ratio`: Class label noise ratio (default: 0.1)

2. **`pm/pointmamba/mp_noise.py`** (NEW)
   - `MPNoiseGenerator`: Generates noisy GT masks by:
     - Randomly sampling GT masks
     - Applying point dropout (20% by default)
     - Applying label flip noise (10% by default)

3. **`pm/pointmamba/mp_query_generator.py`** (NEW)
   - `MPQueryGenerator`: Converts noisy masks to query embeddings by:
     - Aggregating features using masks as attention weights
     - Adding class embeddings
     - Producing query embeddings in same space as original queries

4. **`pm/pointmamba/pointmask.py`**
   - Modified `MaskDecoder.__init__` to:
     - Initialize MP noise generator and query generator
     - Store MP training configuration
   - Modified `MaskDecoder.forward` to:
     - Accept optional GT masks and classes
     - Generate MP queries during training
     - Concatenate with original queries
     - Return `is_mp_query` indicator for loss computation

5. **`pm/pointmamba/losses.py`**
   - Modified `PMLoss.forward` to:
     - Accept `is_mp_query` parameter
     - Separate MP and original query predictions
     - Apply Hungarian matching only to original queries
     - Apply direct supervision to MP queries
   - Added `_compute_mp_losses` method:
     - Computes direct losses for MP queries without matching
     - Returns `mp_loss_mask`, `mp_loss_dice`, `mp_loss_cross_entropy`

6. **`pm/pointmamba/point_sis_masked_former.py`**
   - Modified `PointSIS_Seg.forward` to:
     - Prepare GT masks/classes when MP training enabled
     - Pass to `mask_decoder`
     - Pass `is_mp_query` to loss computation
   - Added import for `tooth_lables` function

7. **`train_point_mask.py`**
   - Enabled MP training with configuration:
     ```python
     m_config.use_mp_training = True
     m_config.mp_num_queries = 6
     m_config.mp_noise_ratio = 0.2
     m_config.mp_label_noise_ratio = 0.1
     ```

## Usage

### Training with MP-Former

MP training is now enabled by default in `train_point_mask.py`. To disable it:

```python
# In train_point_mask.py
m_config = make_default_config()
m_config.use_mp_training = False  # Disable MP training
```

### Customizing MP Training

You can adjust MP training parameters:

```python
m_config.mp_num_queries = 8  # Use 8 MP queries instead of 6
m_config.mp_noise_ratio = 0.15  # Less aggressive noise
m_config.mp_label_noise_ratio = 0.05  # Less label noise
```

### Monitoring Training

When MP training is enabled, you'll see additional loss components:

```python
# Original losses (always present)
loss['loss_cross_entropy']
loss['loss_mask']
loss['loss_dice']
loss['loss_geo']

# MP losses (only during MP training)
loss['mp_loss_cross_entropy']
loss['mp_loss_mask']
loss['mp_loss_dice']
```

## Testing

Run the unit tests to verify the implementation:

```bash
cd /home/archie/Projects/AI_algorithm/mamba_test
python -m pytest tests/test_mp_training.py -v
```

Or run directly:

```bash
python tests/test_mp_training.py
```

## Verification Checklist

- [x] Configuration parameters added
- [x] MP noise generator implemented
- [x] MP query generator implemented
- [x] Mask decoder modified to support MP queries
- [x] Loss computation modified to handle MP queries
- [x] Main model modified to pass GT data
- [x] Training script updated to enable MP training
- [x] Unit tests created
- [ ] All unit tests pass (needs to be run)
- [ ] Training completes without errors (needs to be verified)
- [ ] Memory overhead < 20% (needs to be measured)
- [ ] mIoU improvement > 0.5% over baseline (needs to be measured)

## Technical Notes

### Noise Generation Strategy

1. **Point Dropout**: Randomly sets 20% of mask values to 0
   - Helps model learn robust features
   - Prevents overfitting to exact mask boundaries

2. **Label Flip**: Randomly changes 10% of class labels
   - Encourages model to use mask features, not just class priors
   - Improves generalization

3. **Query Sampling**: Samples GT masks with replacement if needed
   - Ensures consistent number of MP queries per batch
   - Handles variable number of GT objects

### Direct Supervision Strategy

MP queries bypass Hungarian matching and use direct supervision:
- Each MP query is paired with a corresponding GT mask
- Loss is computed directly without the matching step
- Provides stronger gradient signal for training

### Memory Considerations

- Training: +10-15% memory (30 queries vs 24)
- Inference: Zero overhead (MP queries not generated)

## Troubleshooting

### Issue: Out of Memory

**Solution**: Reduce `mp_num_queries` or `batch_size`:
```python
m_config.mp_num_queries = 4  # Instead of 6
# or
batch_size = 1  # Already small
```

### Issue: Training Slower Than Expected

**Solution**: This is expected initially. MP training should converge faster overall.

### Issue: No Improvement in Accuracy

**Possible causes**:
1. Noise ratios too high - try reducing `mp_noise_ratio` to 0.15
2. Not enough MP queries - try increasing to 8
3. Training not long enough - MP training needs fewer epochs total

## References

- [MP-Former Paper](https://arxiv.org/abs/2303.07336) - CVPR 2023
- Original implementation based on PointSIS architecture

## Next Steps

1. Run unit tests to verify implementation
2. Train model with MP training enabled
3. Compare with baseline (MP training disabled)
4. Measure memory overhead and training speed
5. Evaluate accuracy improvement (mIoU)
6. Fine-tune hyperparameters if needed

## Contact

For questions or issues, please refer to the main project documentation.
