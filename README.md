**Structure-First Backpropagation**

This repo explores a simple training rule:

Train a dense directed graph spanning inputs and outputs with standard backprop, and interleave discrete rounding steps that snap all weights to {0, 1}.

The graph’s structure is learned during training, not fixed in advance.

**The Setup**

We start with a dense directed graph.

The node set includes:

input nodes (I),

output nodes (O),

optional internal nodes.

Directed edges are allowed between nodes at the chosen granularity.

There is no predefined architecture unless you impose one.

**Parameters**

All parameters are weights.

Weights may live:

on directed edges,

on nodes (self-weights / gates).

There is no semantic distinction between node weights and edge weights.

If a weight is zero, the corresponding connection or node is inactive.

---

## Implementation

This repository includes:

1. **Architecture Diagram** (`ARCHITECTURE.md`) - Visual representation of the Structure-First Backpropagation concept
2. **Training Loop Pseudocode** (`TRAINING_PSEUDOCODE.md`) - Detailed algorithmic description
3. **PyTorch Prototype** (`structure_backprop.py`) - Working implementation
4. **Demo Examples** (`example.py`) - Demonstrations on XOR and addition tasks

### Quick Start

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the demo:
```bash
python example.py
```

This will train the Structure-First Backpropagation network on two tasks:
- **XOR Problem**: Non-linear task requiring hidden nodes
- **Addition Problem**: Linear task that can use direct connections

The demo shows how the network learns which connections to keep (weight=1) or remove (weight=0) through training.

### Key Features

- **Dense Graph Initialization**: Starts with fully connected network
- **Gradient-Based Learning**: Uses standard backpropagation
- **Structure Discovery**: Periodic weight rounding to {0, 1} discovers sparse topology
- **No Predefined Architecture**: Network learns its own structure

### Usage Example

```python
from structure_backprop import StructureBackpropNetwork, train_structure_backprop
import torch

# Create network
model = StructureBackpropNetwork(
    n_input=2,
    n_hidden=4,
    n_output=1,
    rounding_threshold=0.3
)

# Prepare data
X = torch.randn(100, 2)
y = torch.randn(100, 1)

# Train
history = train_structure_backprop(
    model=model,
    train_data=(X, y),
    n_epochs=100,
    learning_rate=0.01,
    rounding_frequency=10
)

# Inspect learned structure
print(f"Sparsity: {model.get_sparsity():.2%}")
print(f"Active edges: {model.get_active_edges()}")
edges = model.get_structure()
```

### Files

- `ARCHITECTURE.md` - Detailed architecture diagrams and information flow
- `TRAINING_PSEUDOCODE.md` - Step-by-step algorithmic pseudocode
- `structure_backprop.py` - Core implementation
- `example.py` - Demonstration scripts with visualizations
- `requirements.txt` - Python dependencies
