---
dataset_info:
  features:
  - name: case_id
    dtype: string
  - name: file
    dtype: string
  - name: num_nodes
    dtype: int64
  - name: num_edges
    dtype: int64
  - name: metadata
    dtype: string
  splits:
  - name: train
    num_bytes: 171422
    num_examples: 1100
  - name: test
    num_bytes: 44934
    num_examples: 290
  download_size: 108913
  dataset_size: 216356
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
---
# Laser Powder Bed Fusion (LPBF) Additive Manufacturing Dataset

As part of our paper **[FLARE: Fast Low-Rank Attention Routing Engine](https://huggingface.co/papers/2508.12594)** ([arXiv:2508.12594](https://arxiv.org/abs/2508.12594)), we release a new **3D field prediction benchmark** derived from numerical simulations of the **Laser Powder Bed Fusion (LPBF)** additive manufacturing process.

This dataset is designed for evaluating neural surrogate models on **3D field prediction tasks** over complex geometries with up to **50,000 nodes**. We believe this benchmark will be useful for researchers working on graph neural networks, mesh-based learning, surrogate PDE modeling, or 3D foundation models.

---

## Dataset Overview

In metal additive manufacturing (AM), subtle variations in design geometry can cause residual stresses and shape distortion during the build process, leading to part inaccuracies or failures. We simulate the LPBF process on a set of complex 3D CAD geometries to generate a benchmark dataset where the goal is to **predict the vertical (Z) displacement field** of the printed part.

| Split        | # Samples | Max # Nodes / sample |
|--------------|-----------|----------------------|
| Train        | 1,100     | ~50,000              |
| Test         | 290       | ~50,000              |

Each sample consists of:

- `points`: array of shape `(N, 3)` (x, y, z coordinates of mesh nodes)
- optionally connectivity: `edge_index` array specifying axis-aligned hexahedral elements
- 3D `displacement` filed: array of shape `(N, 3)`
- Von mises `stress` field: array of shape `(N, 1)`

---

## Usage

### Quick Start

Load the LPBF dataset using the optimized PyTorch Geometric interface:

```python
from pdebench.dataset.utils import LPBFDataset

# Load train and test datasets
train_dataset = LPBFDataset(split='train')
test_dataset = LPBFDataset(split='test')

print(f"Train samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Access a sample
sample = train_dataset[0]
print(f"Nodes: {sample.x.shape}")           # [N, 3] - node coordinates
print(f"Edges: {sample.edge_index.shape}")  # [2, E] - edge connectivity 
print(f"Target: {sample.y.shape}")          # [N] - Z-displacement values
print(f"Elements: {sample.elems.shape}")    # [M, 8] - hex element connectivity
```

### PyTorch DataLoader Integration

```python
from torch_geometric.loader import DataLoader

# Create DataLoader for training
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Training loop example
for batch in train_loader:
    # batch.x: [batch_size*N, 3] - node coordinates
    # batch.y: [batch_size*N] - target Z-displacements
    # batch.edge_index: [2, batch_size*E] - edges
    # batch.batch: [batch_size*N] - batch assignment
    
    # Your model forward pass here
    pred = model(batch.x, batch.edge_index, batch.batch)
    loss = loss_fn(pred, batch.y)
```

### Performance Features

- **⚡ Fast initialization**: ~0.8s (vs 18s+ for naive approaches)
- **🚀 Efficient loading**: ~8ms per sample access
- **💾 Smart caching**: Downloads once, cached locally
- **🔄 Lazy loading**: Files downloaded only when first accessed

### Data Fields

Each sample contains:
- `x` (pos): Node coordinates [N, 3]
- `edge_index`: Edge connectivity [2, E] 
- `y`: Target Z-displacement [N]
- `elems`: Element connectivity [M, 8]
- `temp`: Temperature field [N]
- `disp`: Full displacement field [N, 3] 
- `vmstr`: Von Mises stress [N]
- `metadata`: Simulation metadata

## Implementation

### LPBFDataset Class

The optimized `LPBFDataset` implementation with lazy loading and efficient caching:

```python
import os
import json
import numpy as np
import torch
import torch_geometric as pyg
import datasets
import huggingface_hub

class LPBFDataset(pyg.data.Dataset):
    def __init__(self, split='train', transform=None):
        assert split in ['train', 'test'], f"Invalid split: {split}. Must be one of: 'train', 'test'."

        self.repo_id = 'vedantpuri/LPBF_FLARE'
        
        print(f"Initializing {split} dataset...")
        
        # Fast initialization: Load dataset index first (lightweight)
        import time
        start_time = time.time()
        self.dataset = datasets.load_dataset(self.repo_id, split=split, keep_in_memory=True)
        dataset_time = time.time() - start_time
        print(f"Dataset index load: {dataset_time:.2f}s")

        # Lazy cache initialization - only download when needed
        self._cache_dir = None
        
        print(f"✅ Loaded {len(self.dataset)} samples for {split} split")

        super().__init__(None, transform=transform)
    
    @property
    def cache_dir(self):
        """Lazy loading of cache directory - only download when first sample is accessed."""
        if self._cache_dir is None:
            print("Downloading repository files on first access...")
            import time
            start_time = time.time()
            self._cache_dir = huggingface_hub.snapshot_download(self.repo_id, repo_type="dataset")
            download_time = time.time() - start_time
            print(f"Repository download/cache: {download_time:.2f}s")
            print(f"Cache directory: {self._cache_dir}")
        return self._cache_dir

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        # Get file path from index
        entry = self.dataset[idx]
        rel_path = entry["file"]
        npz_path = os.path.join(self.cache_dir, rel_path)

        # Load NPZ file (main bottleneck check)
        data = np.load(npz_path, allow_pickle=True)
        graph = pyg.data.Data()

        # Convert to tensors efficiently
        for key, value in data.items():
            if key == "_metadata":
                graph["metadata"] = json.loads(value[0])["metadata"]
            else:
                # Use torch.from_numpy for faster conversion when possible
                if value.dtype.kind == "f":
                    tensor = torch.from_numpy(value.astype(np.float32))
                else:
                    tensor = torch.from_numpy(value.astype(np.int64)) if value.dtype != np.int64 else torch.from_numpy(value)
                graph[key] = tensor

        # Set standard attributes
        graph.x = graph.pos
        graph.y = graph.disp[:, 2]

        return graph
```

### Key Implementation Features

#### 🚀 **Lazy Loading Strategy**
- **Fast initialization** (~0.8s): Only loads lightweight parquet index
- **Deferred downloads**: Heavy NPZ files downloaded on first sample access
- **Property-based caching**: `@property cache_dir` ensures files download only when needed

#### ⚡ **Efficient Tensor Conversion**
```python
# Optimized: Direct numpy->torch conversion (zero-copy when possible)
tensor = torch.from_numpy(value.astype(np.float32))

# vs. Slower: torch.tensor() creates new copy
tensor = torch.tensor(value, dtype=torch.float32)
```

#### 💾 **Smart Caching**
- Uses HuggingFace's built-in caching system
- Files downloaded once, reused across all dataset instances
- Automatic cache validation and updates

#### 🎯 **Memory Efficiency**
- No preloading of samples
- On-demand loading with `np.load()`
- Minimal memory footprint during initialization

---

## Source & Generation

- Geometries are taken from the **Fusion 360 segmentation dataset**.
- Simulations performed using **Autodesk NetFabb** with Ti-6Al-4V material on a Renishaw AM250 machine.
- Full thermomechanical simulation producing residual stress and displacement fields.
- We applied subsampling and aspect-ratio filtering to select ~1,390 usable simulations.
- The dataset focuses on **steady-state residual deformation prediction**.

---

## Dataset Gallery

We simulate the LPBF process on selected geometries from the Autodesk segementation dataset (Lambourne et al., 2021) to generate a benchmark dataset for AM calculations. Several geometries are presented in this gallery. The color indicates Z (vertical) displacement field.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/6729087b934501c4f242f768/bGSdeMmf4iSWTVflVg2p5.png)

---

## Dataset Statistics

Summary of LPBF dataset statistics.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/6729087b934501c4f242f768/179s7b3TT03tvt9oNFg-O.png)

---

## Benchmark Task

**Task**: Given the 3D mesh coordinates of a part, predict the Z-displacement at each node after the LPBF build process (final state).

This surrogate modeling task is highly relevant to the additive manufacturing field, where fast prediction of distortion can save time and cost compared to full-scale FEM simulation.

---

## Citation

If you use this dataset in your work, please cite:

```
@misc{puri2025flare,
      title={{FLARE}: {F}ast {L}ow-rank {A}ttention {R}outing {E}ngine}, 
      author={Vedant Puri and Aditya Joglekar and Kevin Ferguson and Yu-hsuan Chen and Yongjie Jessica Zhang and Levent Burak Kara},
      year={2025},
      eprint={2508.12594},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2508.12594}, 
}
```

---

## Future Work & Extensions

We plan to expand this dataset toward larger-scale **3D shape foundation models**, and potentially include dynamic time-history fields (stress, temperature, etc.) in future releases.

---

## License

MIT License

---

## Contact

For questions about the dataset or related research, feel free to reach out via email or the GitHub repository linked in the paper: [`https://github.com/vpuri3/FLARE.py`](https://github.com/vpuri3/FLARE.py).
