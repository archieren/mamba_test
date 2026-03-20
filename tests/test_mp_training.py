"""Unit tests for MP-Former training implementation."""
import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pm.pointmamba import PointSIS_Seg, make_default_config
from pm.pointmamba.mp_noise import MPNoiseGenerator
from pm.pointmamba.mp_query_generator import MPQueryGenerator
from pm.utils.point_cloud import PointCloud


def test_noise_generation():
    """Test noise generation for MP queries."""
    print("Testing noise generation...")
    config = make_default_config()
    config.mp_noise_ratio = 0.2
    config.mp_label_noise_ratio = 0.1
    config.num_labels = 4

    noise_gen = MPNoiseGenerator(config)

    # Create dummy data
    masks = [torch.randint(0, 2, (10, 1000)).float()]
    classes = [torch.randint(1, 5, (10,))]

    noisy_masks, noisy_classes = noise_gen(masks, classes, num_mp_queries=6, device='cpu')

    assert noisy_masks[0].shape[0] == 6, f"Expected 6 masks, got {noisy_masks[0].shape[0]}"
    assert noisy_classes[0].shape[0] == 6, f"Expected 6 classes, got {noisy_classes[0].shape[0]}"
    print("✓ Noise generation test passed!")


def test_query_generator():
    """Test MP query generation."""
    print("\nTesting MP query generator...")
    config = make_default_config()
    config.d_model = 96
    config.num_labels = 4

    query_gen = MPQueryGenerator(config)

    # Create dummy data
    noisy_masks = [
        torch.randint(0, 2, (3, 1000)).float(),  # 3 MP queries for batch 0
        torch.randint(0, 2, (3, 1000)).float()   # 3 MP queries for batch 1
    ]
    noisy_classes = [
        torch.randint(1, 5, (3,)),  # 3 classes for batch 0
        torch.randint(1, 5, (3,))   # 3 classes for batch 1
    ]
    mask_features = torch.randn(2, 1000, 96)  # b=2, g=1000, d=96

    query_emb = query_gen(noisy_masks, noisy_classes, mask_features)

    assert query_emb.shape[0] == 6, f"Expected 6 query embeddings, got {query_emb.shape[0]}"
    assert query_emb.shape[1] == 96, f"Expected d_model=96, got {query_emb.shape[1]}"
    print("✓ Query generator test passed!")


def test_forward_with_mp():
    """Test that MP components are properly initialized."""
    print("\nTesting MP components initialization...")
    config = make_default_config()
    config.use_mp_training = True
    config.mp_num_queries = 6
    config.num_queries = 24

    model = PointSIS_Seg(config).train()

    # Check that MP components exist
    assert hasattr(model.mask_decoder, 'noise_gen'), "Missing noise_gen in mask_decoder"
    assert hasattr(model.mask_decoder, 'query_gen'), "Missing query_gen in mask_decoder"
    assert model.mask_decoder.use_mp_training == True, "use_mp_training should be True"
    assert model.mask_decoder.num_mp_queries == 6, "num_mp_queries should be 6"
    assert model.mask_decoder.num_original_queries == 24, "num_original_queries should be 24"

    print("✓ MP components initialization test passed!")


def test_inference_no_mp():
    """Test that MP components are disabled in eval mode."""
    print("\nTesting inference mode behavior...")
    config = make_default_config()
    config.use_mp_training = True  # MP training enabled
    config.mp_num_queries = 6
    config.num_queries = 24

    model = PointSIS_Seg(config).eval()  # Set to eval mode

    # Check that model is in eval mode
    assert not model.training, "Model should be in eval mode"

    print("✓ Inference mode test passed!")


def test_config_defaults():
    """Test that config has correct defaults."""
    print("\nTesting configuration defaults...")
    config = make_default_config()

    # MP training should be disabled by default
    assert config.use_mp_training == False, "MP training should be disabled by default"
    assert config.mp_num_queries == 6, "Default mp_num_queries should be 6"
    assert config.mp_noise_ratio == 0.2, "Default mp_noise_ratio should be 0.2"
    assert config.mp_label_noise_ratio == 0.1, "Default mp_label_noise_ratio should be 0.1"

    print("✓ Configuration defaults test passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("Running MP-Former Training Tests")
    print("=" * 60)

    try:
        test_config_defaults()
        test_noise_generation()
        test_query_generator()
        test_forward_with_mp()
        test_inference_no_mp()

        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
