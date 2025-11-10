# Quick Start Guide - Configuration System

## TL;DR - What Changed?

✅ **DataProcessorConfig** is now unified for ALL data processing (online & offline)
✅ **DatasetConfig** can be created from **DataProcessorConfig**
✅ **Validation** is automatic (catches errors early)
✅ **100% backward compatible** (old code still works)

---

## 3-Minute Setup

### For Online Prediction

```python
from src.context_aware.config.configs import DataProcessorConfig
from src.online_prediction.OnlineDataProcessor.dataProcessor import DataProcessor

# Create config (all params optional)
config = DataProcessorConfig.initialize(
    dim_data=12,
    window_length=20,
    min_val=-0.5,
    max_val=0.5,
    Ts=0.01
)

# Use processor
processor = DataProcessor(config)
processor.add_data_point(data)
context, sources, no_smooth, _ = processor.get_window_features()
predictions = model.inference(context, sources, no_smooth)
```

### For Offline Training

```python
from src.context_aware.config.configs import (
    DataProcessorConfig, DatasetConfig, TrainingConfig, ModelConfig
)

# 1. Processor config (source of truth)
proc_config = DataProcessorConfig.initialize(
    dim_data=12,
    window_length=20,
    data_augment=True
)

# 2. Dataset config (converted from processor)
dataset_config = DatasetConfig.from_processor_config(proc_config)

# 3. Training config (with validation)
training_config = TrainingConfig(
    num_epochs=100,
    learning_rate=0.001,  # Auto-validated: 0 < x < 1
    batch_size=128
)

# 4. Use in pipeline
model, device = createModel(ModelConfig(...))
model, loss_hist = trainModel(model, train_data, test_data, training_config)
```

---

## Common Parameters

### DataProcessorConfig.initialize()

```python
DataProcessorConfig.initialize(
    dim_data=12,           # Number of features
    window_length=20,      # Context window size
    min_val=-0.5,          # Min normalization (all features)
    max_val=0.5,           # Max normalization (all features)
    smooth_fc=3.0,         # Smoothing cutoff frequency
    smooth_order=3,        # Smoothing filter order
    Ts=0.01,               # Sampling period (seconds)
    data_augment=True      # Use data augmentation?
)
```

### TrainingConfig()

```python
TrainingConfig(
    num_epochs=100,        # > 0
    learning_rate=0.001,   # 0 < x < 1
    batch_size=128,        # > 0, power of 2 recommended
    lambda_traffic_class=100.0,
    lambda_transmission=500.0,
    lambda_context=50.0
)
```

### ModelConfig()

```python
ModelConfig(
    input_size=12,         # Features per timestep
    output_size=1,         # Prediction output size
    hidden_size=128,       # Hidden units
    num_layers=3,          # Stacked layers
    dropout_rate=0.5,      # 0 <= x < 1
    len_source=20,         # Source window
    len_target=20,         # Target window
    num_classes=21         # len_target + 1
)
```

---

## Common Errors & Fixes

### ❌ Learning rate too high
```python
TrainingConfig(learning_rate=2.0)
# ValueError: learning_rate must be in (0, 1), got 2.0
```

✅ **Fix**: Use value between 0 and 1
```python
TrainingConfig(learning_rate=0.001)
```

### ❌ num_classes mismatch
```python
ModelConfig(..., len_target=20, num_classes=20)
# ValueError: num_classes (20) should equal len_target + 1 (21)
```

✅ **Fix**: Set num_classes = len_target + 1
```python
ModelConfig(..., len_target=20, num_classes=21)
```

### ❌ Empty min/max arrays
```python
config = DataProcessorConfig.initialize(dim_data=12)
# Config created but min_vals/max_vals are arrays of shape (12,)
```

✅ **Use**: This is correct behavior now!
```python
print(config.min_vals.shape)  # (12,)
```

---

## Migration Checklist

If you're upgrading existing code:

- [ ] Replace `DataProcessorConfig.initialize(dim_data, window_length)` 
  - Add all parameters you need
  - Old signature still works but incomplete

- [ ] For training, use `DatasetConfig.from_processor_config()`
  - Don't create DatasetConfig separately
  - Ensures parameter consistency

- [ ] Remove try/except around config creation
  - Validation is now automatic in `__post_init__()`
  - Errors caught immediately

- [ ] Test error messages
  - ValidationErrors are now clear and specific
  - Use them to debug issues

---

## Best Practices

### ✅ DO

```python
# 1. Create unified processor config
processor_config = DataProcessorConfig.initialize(
    dim_data=features,
    window_length=window_len,
    data_augment=use_augment
)

# 2. Let validation catch errors
try:
    bad_training = TrainingConfig(learning_rate=-0.1)
except ValueError as e:
    print(f"Configuration error: {e}")

# 3. Use from_processor_config() for conversions
dataset_config = DatasetConfig.from_processor_config(processor_config)

# 4. Display config for debugging
processor_config.display()
```

### ❌ DON'T

```python
# 1. Create separate configs that might diverge
proc_config = DataProcessorConfig.initialize(dim_data=12, window_length=20)
dataset_config = DatasetConfig.initialize(len_window=20, data_augment=True)
# ❌ Smoothing params now different!

# 2. Create configs with invalid parameters
TrainingConfig(num_epochs=-1)  # ❌ Will raise ValueError

# 3. Ignore validation errors
try:
    bad_config = ModelConfig(..., len_target=20, num_classes=20)
except ValueError:
    pass  # ❌ Silently ignoring errors

# 4. Create ModelConfig manually when you have dataset
model_config = ModelConfig(...)  # ❌ Should use .from_dataset()
```

---

## Performance Notes

- ✅ No performance penalty
- ✅ Validation happens once at initialization
- ✅ Numpy array creation is O(n) where n=dim_data (fast)
- ✅ Configuration is immutable (no copying overhead)

---

## Backward Compatibility

✅ All old code still works:

```python
# Old way (still valid)
config = DataProcessorConfig.initialize(dim_data=12, window_length=20)
processor = DataProcessor(config)

# New way (recommended)
config = DataProcessorConfig.initialize(
    dim_data=12,
    window_length=20,
    min_val=-0.5,
    max_val=0.5
)
processor = DataProcessor(config)

# Both work! New way is more explicit.
```

---

## Key Changes Summary

| What | Before | After |
|------|--------|-------|
| Smoothing | 2 places | 1 place ✅ |
| Bounds | Inconsistent | Consistent ✅ |
| Validation | None | Automatic ✅ |
| Docs | Minimal | Comprehensive ✅ |
| API | Confusing | Clear ✅ |

---

## Need Help?

1. **Quick answers**: Check "Common Errors & Fixes" above
2. **How to migrate**: Read `CONFIG_MIGRATION_GUIDE.md`
3. **Understanding design**: Read `CONFIG_ARCHITECTURE.md`
4. **Full details**: Read `CONFIG_ANALYSIS.md`

---

## Example: Complete Training Pipeline

```python
import numpy as np
from src.context_aware.config.configs import (
    DataProcessorConfig, DatasetConfig, TrainingConfig, ModelConfig
)
from src.context_aware.models import createModel
from src.context_aware.training import trainModel
from src.context_aware.preprocessing import PreparingDataset

# ========== CONFIGURATION ==========
processor_config = DataProcessorConfig.initialize(
    dim_data=12,
    window_length=20,
    min_val=-0.5,
    max_val=0.5,
    Ts=0.01,
    data_augment=True,
    smooth_fc=3.0,
    smooth_order=3
)

dataset_config = DatasetConfig.from_processor_config(processor_config)

training_config = TrainingConfig(
    num_epochs=100,
    learning_rate=0.001,
    batch_size=128,
    lambda_traffic_class=100.0,
    lambda_transmission=500.0,
    lambda_context=50.0
)

# ========== PREPARE DATA ==========
train_dataset, test_dataset = PreparingDataset(data_unit, dataset_config)

# ========== CREATE MODEL ==========
model_config = ModelConfig.from_dataset(dataset_config, train_dataset)
model, device = createModel(model_config)

# ========== TRAIN ==========
model, train_losses, test_losses = trainModel(
    model,
    train_dataset,
    test_dataset,
    training_config,
    verbose=True
)

print("✅ Training complete!")
```

---

**Last Updated**: November 2025
**Status**: Ready for Production ✅

