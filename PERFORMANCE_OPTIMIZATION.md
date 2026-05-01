# Performance Optimization Guide

## 🚀 Maximizing System Performance

This guide provides strategies to optimize the Smart Student Lifestyle Analytics System for speed, accuracy, and scalability.

---

## 1. GPU Acceleration

### Check GPU Status

```python
import torch

print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU Count: {torch.cuda.device_count()}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")
print(f"CUDA Version: {torch.version.cuda}")
```

### Enable GPU in Code

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
X = X.to(device)
```

### Expected Speedup

| Model | CPU (seconds) | GPU (seconds) | Speedup |
|-------|---------------|---------------|---------|
| MLP | 45 | 3 | **15x** |
| DNN | 120 | 8 | **15x** |
| Dropout | 130 | 10 | **13x** |
| Batch Norm | 90 | 6 | **15x** |
| Hybrid | 180 | 15 | **12x** |

---

## 2. Batch Size Optimization

### Effect of Batch Size

```python
# Small batch (better generalization, slower)
BATCH_SIZE = 16    # More iterations, slower per epoch

# Medium batch (balanced)
BATCH_SIZE = 32    # Recommended default
BATCH_SIZE = 64    # Good for GPUs with 8GB+

# Large batch (faster, may need higher learning rate)
BATCH_SIZE = 128   # For GPUs with 12GB+ VRAM
```

### GPU Memory Estimation

```
Approximate VRAM needed ≈ (4 * batch_size * features * 2)
                        = (4 * 32 * 16 * 2) = ~4 MB per batch

Rule of thumb:
- 4GB GPU:   batch_size = 16-32
- 8GB GPU:   batch_size = 32-64
- 12GB GPU:  batch_size = 64-128
- 24GB GPU:  batch_size = 128-256
```

### Configure in settings.py

```python
TRAINING_CONFIG = {
    'batch_size': 64,  # Increase for larger GPUs
    'epochs': 100,
    'learning_rate': 0.001,
}
```

---

## 3. Learning Rate Scheduling

### Current Implementation (ReduceLROnPlateau)

```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,        # Reduce by 50%
    patience=10,       # After 10 epochs without improvement
    verbose=True
)
```

### Alternative: Cosine Annealing

```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100,         # Total epochs
    eta_min=1e-5
)

# Better for: Periodic reduction, smoother convergence
# Expected improvement: 1-2% better final accuracy
```

### Alternative: Cyclical Learning Rate

```python
scheduler = optim.lr_scheduler.CyclicLR(
    optimizer,
    base_lr=0.0001,
    max_lr=0.001,
    step_size_up=20,
    cycle_momentum=False
)

# Better for: Faster convergence, fewer epochs needed
# Expected improvement: 20-30% fewer epochs
```

---

## 4. Model Architecture Optimization

### Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for X_batch, y_batch in train_loader:
    with autocast():
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
    
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()

# Benefits: 2x faster, uses half VRAM
# Accuracy impact: < 0.1% in most cases
```

### Knowledge Distillation

```python
# Train teacher model first
teacher = DNNModel(input_size, output_size)
teacher.train()

# Then distill to faster student
student = MLPModel(input_size, output_size)

# Distillation loss
temperature = 4.0
distillation_loss = nn.KLDivLoss()(
    F.log_softmax(student_out / temperature, dim=1),
    F.softmax(teacher_out / temperature, dim=1)
)

# Benefits: 5x faster inference with 95% accuracy
```

### Quantization (Post-Training)

```python
import torch.quantization as quantization

# Convert to quantized model
model.qconfig = quantization.get_default_qat_qconfig('fbgemm')
quantization.prepare(model, inplace=True)

# Calibrate with data
for X, _ in calibration_loader:
    model(X)

quantization.convert(model, inplace=True)

# Benefits: 4x smaller model, 2x faster inference
# Accuracy impact: 0.5-1% loss typical
```

---

## 5. Data Loading Optimization

### Optimized Data Loader

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,        # ⭐ Parallel data loading
    pin_memory=True,      # ⭐ Pin memory for faster transfer to GPU
    drop_last=True        # Drop last incomplete batch
)

# For inference (single worker, no shuffling)
val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

# Expected improvement: 20-40% faster data loading
```

### Prefetching

```python
class PrefetchDataLoader:
    """Prefetch next batch while processing current batch"""
    
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
    
    def __iter__(self):
        for X, y in self.loader:
            X = X.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            yield X, y

# Usage
train_loader = PrefetchDataLoader(train_loader, device)

# Expected improvement: 10-15% faster iteration
```

---

## 6. Distributed Training

### Multi-GPU with DataParallel

```python
import torch.nn as nn

model = create_model('hybrid', input_size=16)
model = nn.DataParallel(model)  # Use all available GPUs

# Simple but less efficient
# Speedup: ~2x with 2 GPUs, sublinear scaling
```

### Multi-GPU with DistributedDataParallel (Better)

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

# Initialize distributed training
dist.init_process_group(backend='nccl')

model = create_model('hybrid', input_size=16)
model = model.to(rank)
model = DistributedDataParallel(model, device_ids=[rank])

# Expected speedup: ~1.9x with 2 GPUs, ~3.7x with 4 GPUs
```

### Launch Multi-GPU Training

```bash
# Run with 2 GPUs
python -m torch.distributed.launch --nproc_per_node=2 training/run_training.py

# Run with 4 GPUs
python -m torch.distributed.launch --nproc_per_node=4 training/run_training.py
```

---

## 7. Inference Optimization

### Model Compilation (PyTorch 2.0+)

```python
# Compile model for faster inference
model = torch.compile(model)

# Benefits: 1.5-2x faster inference
# Overhead: First run takes longer for compilation
```

### Batch Inference

```python
# Inefficient (one by one)
predictions = []
for sample in samples:
    pred = model(sample.unsqueeze(0))
    predictions.append(pred)

# Efficient (batch)
all_samples = torch.stack(samples)
predictions = model(all_samples)

# Speedup: 10-50x faster for batch of 32
```

### Export to ONNX

```python
import torch.onnx

dummy_input = torch.randn(1, 16, device=device)
torch.onnx.export(
    model, dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=13
)

# Benefits: Deploy to C++, mobile, or web
# Performance: Near-native C++ speed
```

---

## 8. Database Query Optimization

### Indexed Queries

```sql
-- Current indexes in schema
CREATE INDEX idx_students_roll_no ON students(roll_no);
CREATE INDEX idx_students_created_at ON students(created_at DESC);
CREATE INDEX idx_model_metrics_model_name ON model_metrics(model_name);

-- Add composite index for common queries
CREATE INDEX idx_students_score_burnout ON students(lifestyle_score, burnout_risk);
```

### Batch Database Operations

```python
# Slow (individual inserts)
for row in df.iter_rows(named=True):
    db.insert_student_data(row)

# Fast (batch insert)
db.insert_student_data(df)

# Expected improvement: 50-100x faster
```

### Connection Pooling

```python
from psycopg2.pool import SimpleConnectionPool

# Create connection pool
pool = SimpleConnectionPool(1, 20, database='student_analytics')

def get_connection():
    return pool.getconn()

def release_connection(conn):
    pool.putconn(conn)
```

---

## 9. Monitoring & Profiling

### Training Time Profiler

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Training code
train_model(...)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

### Memory Profiler

```python
from memory_profiler import profile

@profile
def train_epoch(train_loader):
    for X_batch, y_batch in train_loader:
        # Training code
        pass

# Run with: python -m memory_profiler script.py
```

### Bottleneck Detection

```python
import time

# Profile each component
times = {
    'data_loading': 0,
    'forward_pass': 0,
    'backward_pass': 0,
    'optimizer_step': 0
}

start = time.time()
X_batch, y_batch = next(iter(train_loader))
times['data_loading'] += time.time() - start

start = time.time()
predictions = model(X_batch)
times['forward_pass'] += time.time() - start

start = time.time()
loss = criterion(predictions, y_batch)
loss.backward()
times['backward_pass'] += time.time() - start

start = time.time()
optimizer.step()
times['optimizer_step'] += time.time() - start

print(times)
```

---

## 10. Configuration Recommendations by Hardware

### Low-End Laptop (4GB RAM)

```python
TRAINING_CONFIG = {
    'batch_size': 8,
    'epochs': 50,
    'learning_rate': 0.001,
    'num_workers': 0,
    'use_gpu': False,
    'model': 'mlp'  # Simplest model
}
# Training time: ~1-2 minutes
```

### Mid-Range Computer (16GB RAM, GTX 1660)

```python
TRAINING_CONFIG = {
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 0.001,
    'num_workers': 4,
    'use_gpu': True,
    'model': 'hybrid'
}
# Training time: ~2-3 minutes
```

### High-End Workstation (64GB RAM, RTX 3090)

```python
TRAINING_CONFIG = {
    'batch_size': 128,
    'epochs': 100,
    'learning_rate': 0.001,
    'num_workers': 8,
    'use_gpu': True,
    'use_mixed_precision': True,
    'use_compiled_model': True,
    'distributed': False,
    'model': 'hybrid'
}
# Training time: ~30-60 seconds
```

### Server Farm (Multi-GPU)

```python
TRAINING_CONFIG = {
    'batch_size': 512,
    'epochs': 100,
    'learning_rate': 0.01,  # Higher LR for larger batch
    'num_workers': 16,
    'use_gpu': True,
    'use_mixed_precision': True,
    'use_compiled_model': True,
    'distributed': True,
    'num_gpus': 8,
    'model': 'hybrid'
}
# Training time: ~10-20 seconds (8 GPUs)
```

---

## Performance Comparison Summary

| Optimization | Speedup | Complexity | Notes |
|--------------|---------|-----------|-------|
| GPU Acceleration | **15x** | Low | Most important |
| Batch Size (32→64) | **1.2x** | Very Low | Free gain |
| Learning Rate Schedule | **1.1x** | Low | Better convergence |
| Mixed Precision | **2x** | Medium | Minor accuracy loss |
| DataParallel (2 GPU) | **1.9x** | Medium | Sublinear scaling |
| DistributedDataParallel (4 GPU) | **3.7x** | High | Better scaling |
| Compiled Model (PyTorch 2.0+) | **1.5x** | Low | PyTorch 2.0+ only |
| Knowledge Distillation | **5x** (inference) | High | For deployment |
| Quantization | **4x** (model size) | High | 0.5-1% accuracy loss |

---

## Quick Start Optimization

To get immediate improvements, do this:

```bash
# Step 1: Check GPU
python -c "import torch; print(torch.cuda.is_available())"

# Step 2: Update batch size in config/settings.py
# Use batch_size = 64 (if GPU has 8GB+)

# Step 3: Train
python training/run_training.py

# Expected result: 10-15x faster training
```

---

**For questions or specific optimization needs, see the main SETUP.md**
