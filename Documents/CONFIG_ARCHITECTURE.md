# Configuration Architecture Diagram

## System Before Refactoring ❌

```
┌─────────────────────────────────────────────────────────────────┐
│                     Configuration Problem                        │
└─────────────────────────────────────────────────────────────────┘

DataProcessorConfig                          DatasetConfig
├─ dim_data: int                             ├─ len_source: int
├─ window_length: int                        ├─ len_target: int
├─ smooth_fc: float ⚠️ REDUNDANT             ├─ smooth_fc: float ⚠️ REDUNDANT
├─ smooth_order: int ⚠️ REDUNDANT            ├─ smooth_order: int ⚠️ REDUNDANT
├─ Ts: float                                 ├─ max_val: float ⚠️ INCONSISTENT
├─ min_vals: np.ndarray (array)              ├─ min_val: float ⚠️ INCONSISTENT
├─ max_vals: np.ndarray (array)              ├─ data_augment: bool
└─ history_length: int                       ├─ train_ratio: float
                                              └─ (missing other params)
                    ⚠️ CONFLICTS
                    
Online Prediction              Offline Training
    ↓                                ↓
DataProcessor                 PreparingDataset
    │                                │
    └─ Uses DataProcessorConfig      └─ Expects DatasetConfig
       (has smooth_fc)                  (has different smooth_fc)
       
    💥 POTENTIAL INCONSISTENCY 💥
```

## System After Refactoring ✅

```
┌─────────────────────────────────────────────────────────────────┐
│                    Unified Configuration                         │
│                  (Single Source of Truth)                        │
└─────────────────────────────────────────────────────────────────┘

                    DataProcessorConfig
                      (UNIFIED - ALL)
                            │
                ┌───────────┼───────────┐
                │           │           │
                ▼           ▼           ▼
          Online        Offline      Features &
        Prediction     Training      Models
           ↓              ↓              ↓
       DataProcessor PreparingDataset  ModelConfig
           ↓              ↓              
           │              └──→ DatasetConfig
           │                  (via from_processor_config)
           │
    ✅ SINGLE SOURCE OF TRUTH
    ✅ CONSISTENT SMOOTHING PARAMETERS
    ✅ CONSISTENT NORMALIZATION BOUNDS
    ✅ VALIDATED AT INITIALIZATION
```

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    INITIALIZATION FLOW                           │
└─────────────────────────────────────────────────────────────────┘

1. CREATE UNIFIED CONFIG
   ┌────────────────────────────────────┐
   │ DataProcessorConfig.initialize(    │
   │   dim_data=12,                     │
   │   window_length=20,                │
   │   min_val=-0.5,                    │
   │   max_val=0.5,                     │
   │   smooth_fc=3.0,                   │
   │   Ts=0.01,                         │
   │   data_augment=True                │
   │ )                                  │
   └────────────────────────────────────┘
              ↓
   ✓ Validation in __post_init__() (if added)
   ✓ min_vals/max_vals arrays created: shape (12,)
              ↓
        processor_config

2. USE IN ONLINE PREDICTION
   ┌────────────────────────────────────┐
   │ DataProcessor(processor_config)    │
   └────────────────────────────────────┘
   Uses: dim_data, window_length, smooth_fc, smooth_order, Ts, min_vals, max_vals

3. USE IN OFFLINE TRAINING
   ┌─────────────────────────────────────────┐
   │ dataset_config = DatasetConfig          │
   │   .from_processor_config(               │
   │     processor_config,                   │
   │     train_ratio=0.7                     │
   │   )                                     │
   └─────────────────────────────────────────┘
              ↓
   ✓ Automatically maps all parameters
   ✓ Converts per-feature bounds to scalars
              ↓
        dataset_config
        
4. USE IN MODEL TRAINING
   ┌─────────────────────────────────────────┐
   │ training_config = TrainingConfig(...)   │
   │ model_config = ModelConfig(...)         │
   └─────────────────────────────────────────┘
        ↓              ↓
   PreparingDataset  createModel
        ↓              ↓
   training data   model ready
```

## Class Hierarchy

```
┌────────────────────────────────────────────────────────────────┐
│                    Configuration Classes                        │
└────────────────────────────────────────────────────────────────┘

Processing Layer
├─ DataProcessorConfig ⭐ CENTRAL
│  ├─ initialize() → creates with arrays
│  ├─ display() → prints current config
│  └─ __post_init__() → validates [FUTURE]
│
└─ DatasetConfig (backward compatible)
   ├─ initialize() → old API
   ├─ from_processor_config() ⭐ NEW
   └─ __post_init__() → [FUTURE]

Model Layer
├─ ModelConfig
│  ├─ input_size, output_size
│  ├─ hidden_size, num_layers, dropout_rate
│  ├─ from_dataset()
│  └─ __post_init__() ✓ VALIDATES
│
├─ TrainingConfig
│  ├─ num_epochs, learning_rate, batch_size
│  ├─ lambda_* weights
│  └─ __post_init__() ✓ VALIDATES
│
└─ Base
   └─ @dataclass ← All use this

Validation Chain
┌────────────────────────────────────────┐
│ Create Config                          │
│   ↓                                    │
│ __post_init__() called automatically   │
│   ├─ Type check                        │
│   ├─ Range check                       │
│   └─ Consistency check                 │
│   ↓                                    │
│ ✓ Valid → Ready to use                 │
│ ✗ Invalid → ValueError raised          │
└────────────────────────────────────────┘
```

## Parameter Mapping

```
DataProcessorConfig → DatasetConfig
┌─────────────────────────────────┐
│ window_length   → len_source    │
│ window_length   → len_target    │
│ smooth_fc       → smooth_fc     │
│ smooth_order    → smooth_order  │
│ data_augment    → data_augment  │
│ max_vals[0]     → max_val       │
│ min_vals[0]     → min_val       │
│ (input)         → train_ratio   │
└─────────────────────────────────┘

DatasetConfig → ModelConfig
┌─────────────────────────────────┐
│ len_source      → len_source    │
│ len_target      → len_target    │
│ len_target+1    → num_classes   │
│ (from data)     → input_size    │
│ (from data)     → output_size   │
└─────────────────────────────────┘
```

## Component Relationships

```
                  DataProcessorConfig
                       (Core)
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
   Online Prediction  Training Pipeline   Modeling
        │                 │                 │
        ▼                 ▼                 ▼
    DataProcessor   PreparingDataset    ModelConfig
                         │                 │
                         │           TrainingConfig
                         │                 │
                    DatasetConfig          │
                         └─────────────────┘
                                │
                                ▼
                        Training Function
```

## Validation Flow

```
┌──────────────────────────────────────────────────────────────┐
│                    VALIDATION PIPELINE                        │
└──────────────────────────────────────────────────────────────┘

User creates config
    ↓
@dataclass calls __post_init__()
    ├─ TrainingConfig.__post_init__()
    │  ├─ ✓ num_epochs > 0?
    │  ├─ ✓ 0 < learning_rate < 1?
    │  ├─ ✓ batch_size > 0?
    │  └─ ✓ lambda_* >= 0?
    │
    └─ ModelConfig.__post_init__()
       ├─ ✓ input_size > 0?
       ├─ ✓ output_size > 0?
       ├─ ✓ hidden_size > 0?
       ├─ ✓ num_layers > 0?
       ├─ ✓ 0 <= dropout_rate < 1?
       └─ ✓ num_classes == len_target + 1?
    ↓
Result:
├─ ✓ All valid → Config ready
└─ ✗ Invalid → ValueError raised with clear message

Examples:
├─ ValueError: "num_epochs must be positive, got -1"
├─ ValueError: "dropout_rate must be in [0, 1), got 1.5"
└─ ValueError: "num_classes (20) should equal len_target + 1 (21)"
```

## Usage Patterns

```
Pattern 1: Online Prediction
─────────────────────────────
config = DataProcessorConfig.initialize(dim_data=12, window_length=20)
processor = DataProcessor(config)
processor.add_data_point(data)
context, sources, context_no_smooth = processor.get_window_features()
predictions = model.inference(context, sources, context_no_smooth)


Pattern 2: Offline Training
────────────────────────────
processor_config = DataProcessorConfig.initialize(
    dim_data=12, window_length=20, data_augment=True
)
dataset_config = DatasetConfig.from_processor_config(processor_config)
train_data, test_data = prepare_dataset(dataset_config)
model, device = createModel(ModelConfig(...))
train_model(model, train_data, test_data, TrainingConfig(...))


Pattern 3: Mixed (Training + Deployment)
──────────────────────────────────────────
# Training
processor_config = DataProcessorConfig.initialize(...)
dataset_config = DatasetConfig.from_processor_config(processor_config)
model_config = ModelConfig.from_dataset(dataset_config, data)
train_model(model, train_data, test_data, TrainingConfig(...))

# Deployment (uses same processor_config)
processor = DataProcessor(processor_config)
# ... production inference loop ...
```

## Breaking Changes

✅ **None** - Fully backward compatible

Old code continues to work:
```python
processor_config = DataProcessorConfig.initialize(dim_data=12, window_length=20)
dataset_config = DatasetConfig.initialize(len_window=20, data_augment=True)
```

New code uses unified approach:
```python
processor_config = DataProcessorConfig.initialize(
    dim_data=12, window_length=20, data_augment=True
)
dataset_config = DatasetConfig.from_processor_config(processor_config)
```

## Summary

✅ **Central Config** - `DataProcessorConfig` is the source of truth
✅ **Flexible** - Can be converted to `DatasetConfig` when needed
✅ **Validated** - `__post_init__()` catches errors early
✅ **Documented** - Clear docstrings for all parameters
✅ **Backward Compatible** - Old code still works
✅ **Future-Proof** - Easy to extend and maintain

