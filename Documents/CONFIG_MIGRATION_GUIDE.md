# Configuration Migration Guide

## Summary of Changes

The configuration management has been refactored to consolidate redundant code and improve maintainability. The key change is that `DataProcessorConfig` is now the unified configuration for all data processing (both online and offline).

---

## Before vs After

### OLD CODE (Still Works for Backward Compatibility)

```python
from src.context_aware.config.configs import DataProcessorConfig, DatasetConfig

# Online prediction
proc_config = DataProcessorConfig.initialize(dim_data=12, window_length=20)
processor = DataProcessor(proc_config)

# Offline training  
dataset_config = DatasetConfig.initialize(len_window=20, data_augment=True)
```

**Problem**: 
- `DataProcessorConfig` and `DatasetConfig` had overlapping parameters
- Smoothing parameters defined in two places
- Normalization bounds inconsistent (array vs scalar)
- Risk of configuration mismatches

### NEW CODE (Recommended)

```python
from src.context_aware.config.configs import DataProcessorConfig, DatasetConfig

# Initialize unified data processor config (works for both online and offline)
proc_config = DataProcessorConfig.initialize(
    dim_data=12, 
    window_length=20,
    min_val=-0.5,
    max_val=0.5,
    smooth_fc=3.0,
    smooth_order=3,
    Ts=0.01,
    data_augment=True
)

# Online prediction - use processor config directly
processor = DataProcessor(proc_config)

# Offline training - create dataset config from processor config
dataset_config = DatasetConfig.from_processor_config(proc_config, train_ratio=0.7)
```

---

## Migration Steps

### Step 1: Update Online Inference Code

#### Before
```python
from src.context_aware.config.configs import DataProcessorConfig, ModelConfig
from src.online_prediction.OnlineDataProcessor.dataProcessor import DataProcessor

dataProcesorConfig = DataProcessorConfig.initialize(dim_data=dim_data, window_length=window_length)
dataProcesor = DataProcessor(dataProcesorConfig)
```

#### After (New - Recommended)
```python
from src.context_aware.config.configs import DataProcessorConfig, ModelConfig
from src.online_prediction.OnlineDataProcessor.dataProcessor import DataProcessor

processor_config = DataProcessorConfig.initialize(
    dim_data=dim_data, 
    window_length=window_length,
    min_val=-0.5,
    max_val=0.5,
    Ts=0.01
)
processor = DataProcessor(processor_config)
```

### Step 2: Update Offline Training Code

#### Before
```python
from src.context_aware.config.configs import DatasetConfig

dataset_config = DatasetConfig.initialize(len_window=window_length, data_augment=True)
```

#### After (New - Recommended)
```python
from src.context_aware.config.configs import DataProcessorConfig, DatasetConfig

# Create the unified processor config first
processor_config = DataProcessorConfig.initialize(
    dim_data=num_features,
    window_length=window_length,
    data_augment=True
)

# Then create dataset config from it
dataset_config = DatasetConfig.from_processor_config(processor_config, train_ratio=0.7)
```

### Step 3: Update Training Code

No changes needed - the APIs remain the same:

```python
from src.context_aware.training import trainModel
from src.context_aware.config.configs import TrainingConfig

training_config = TrainingConfig(
    num_epochs=100,
    learning_rate=0.001,
    batch_size=128,
    lambda_traffic_class=100.0,
    lambda_transmission=500.0,
    lambda_context=50.0
)

model, loss_history = trainModel(model, train_data, test_data, training_config)
```

---

## Configuration Hierarchy (New)

```
DataProcessorConfig (Unified)
├─ Core parameters:
│  ├─ dim_data
│  ├─ window_length
│  ├─ Ts (sampling period)
│  ├─ min_vals, max_vals (per-feature normalization)
│  ├─ smooth_fc, smooth_order
│  └─ data_augment
│
├─ Used by:
│  ├─ DataProcessor (online prediction)
│  ├─ PreparingDataset (offline training)
│  └─ DatasetConfig (via from_processor_config())
│
└─ Conversion:
   └─ → DatasetConfig.from_processor_config()

TrainingConfig
├─ num_epochs, learning_rate, batch_size
└─ lambda_* weights for multi-task learning

ModelConfig
├─ Architecture: input_size, hidden_size, num_layers, dropout_rate
├─ Data dimensions: len_source, len_target, num_classes
└─ Temporal: dt, degree
```

---

## Key Features

### 1. Unified Single Source of Truth
- **Before**: Smoothing parameters in both `DataProcessorConfig` and `DatasetConfig`
- **After**: All defined once in `DataProcessorConfig`

### 2. Flexible Normalization
```python
# Create bounds for each feature independently
processor_config = DataProcessorConfig.initialize(
    dim_data=12,
    window_length=20,
    min_val=-1.0,  # Applied to all 12 features
    max_val=1.0
)

# Access per-feature bounds
print(processor_config.min_vals.shape)  # (12,)
```

### 3. Validation in `__post_init__`
```python
# TrainingConfig validates parameters
try:
    bad_config = TrainingConfig(num_epochs=-1)  # Raises ValueError
except ValueError as e:
    print(f"Invalid config: {e}")

# ModelConfig validates consistency
try:
    bad_model = ModelConfig(..., len_target=20, num_classes=19)  # Raises ValueError
except ValueError as e:
    print(f"Inconsistent config: {e}")
```

### 4. Improved Documentation
All config classes now have:
- Class-level docstrings
- Parameter descriptions
- Usage examples in method docstrings

---

## Backward Compatibility

The old API still works for now:
```python
# Still valid (but not recommended)
proc_config = DataProcessorConfig.initialize(dim_data=12, window_length=20)
dataset_config = DatasetConfig.initialize(len_window=20, data_augment=True)
```

However, **new code should use the unified approach** for consistency.

---

## Files Modified

1. **`src/context_aware/config/configs.py`**
   - Enhanced `DataProcessorConfig` with all preprocessing parameters
   - Added `from_processor_config()` to `DatasetConfig`
   - Added validation via `__post_init__()` to all configs
   - Added comprehensive docstrings

2. **Notebooks** (examples updated):
   - `experiments/notebooks/main03b_online_inference.ipynb`
   - Other training notebooks should follow same pattern

---

## Testing Your Changes

```python
# Test 1: Create processor config
processor_config = DataProcessorConfig.initialize(
    dim_data=12,
    window_length=20
)
assert processor_config.min_vals.shape == (12,)
assert processor_config.max_vals.shape == (12,)

# Test 2: Convert to dataset config
dataset_config = DatasetConfig.from_processor_config(processor_config)
assert dataset_config.len_source == 20
assert dataset_config.len_target == 20
assert dataset_config.smooth_fc == processor_config.smooth_fc

# Test 3: Validation works
try:
    bad_config = TrainingConfig(num_epochs=-1)
    assert False, "Should have raised ValueError"
except ValueError:
    pass  # Expected
```

---

## Troubleshooting

### Issue: AttributeError: 'DataProcessorConfig' object has no attribute X

**Solution**: Make sure you're creating the config with `.initialize()`:
```python
# Wrong
config = DataProcessorConfig(dim_data=12)  # min_vals/max_vals might be empty

# Correct
config = DataProcessorConfig.initialize(dim_data=12, window_length=20)
```

### Issue: Shape mismatch in normalization

**Solution**: Ensure min_vals and max_vals are created with proper dimensions:
```python
# These will now work correctly
config = DataProcessorConfig.initialize(dim_data=12, min_val=-0.5, max_val=0.5)
print(config.min_vals.shape)  # (12,)
```

### Issue: ModelConfig validation fails

**Solution**: Check that `num_classes == len_target + 1`:
```python
# This will fail
model_config = ModelConfig(
    ...,
    len_target=20,
    num_classes=20  # Wrong! Should be 21
)

# This will work
model_config = ModelConfig(
    ...,
    len_target=20,
    num_classes=21  # Correct
)
```

---

## Questions?

Refer to the `CONFIG_ANALYSIS.md` for the full rationale behind these changes.

