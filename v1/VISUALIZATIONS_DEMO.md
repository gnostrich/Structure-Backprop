# Structure-First Backpropagation - v1 Visualizations Demo

This document demonstrates that the v1 visualizations are working correctly.

## Generated Outputs

Running `python example.py` successfully generated all visualizations for both demonstration tasks:

### XOR Problem (Non-linear Task)

**Location:** `outputs/v1/xor/`

Generated files:
- ✅ `xor_structure.png` - Heatmap showing learned network structure
- ✅ `xor_history.png` - Training metrics over time (loss, sparsity, active edges)
- ✅ `xor_embeddings_umap.png` - UMAP visualization of hidden layer embeddings
- ✅ `xor_summary.txt` - Text summary of training results

**Key Results:**
- Network: 2 inputs → 4 hidden → 1 output (7 total nodes)
- Started with 30 edges (fully connected)
- Learned structure with 14 active edges (53.33% sparsity)
- Final accuracy: 74.50%
- Network discovered it needs hidden nodes for the non-linear XOR task

**Structure Breakdown:**
- Input → Hidden: 8 edges (all inputs connect to hidden nodes)
- Input → Output: 2 edges (direct connections also learned)
- Hidden → Output: 4 edges (all hidden nodes contribute to output)
- Hidden → Hidden: 0 edges (no recurrent connections needed)

### Addition Problem (Linear Task)

**Location:** `outputs/v1/addition/`

Generated files:
- ✅ `addition_structure.png` - Heatmap showing learned network structure
- ✅ `addition_history.png` - Training metrics over time
- ✅ `addition_embeddings_umap.png` - UMAP visualization of hidden layer embeddings
- ✅ `addition_summary.txt` - Text summary of training results

**Key Results:**
- Network: 2 inputs → 3 hidden → 1 output (6 total nodes)
- Started with 20 edges (fully connected)
- Learned structure with 11 active edges (45.00% sparsity)
- Final loss: 0.3782
- Network learned both direct input→output paths and hidden node connections

**Structure Breakdown:**
- Input → Hidden: 6 edges
- Input → Output: 2 edges (strong direct connections for linear task)
- Hidden → Output: 3 edges (all hidden nodes used)
- Hidden → Hidden: 0 edges (no recurrent connections needed)

## Visualization Features Demonstrated

### 1. Structure Heatmaps
- Color-coded weight matrices showing learned connections
- Blue = positive weights (active connections)
- White/light = zero weights (pruned connections)
- Clear visualization of which edges are kept vs. removed

### 2. Training History
Shows three key metrics over training:
- **Training Loss**: Convergence behavior with characteristic spikes at rounding steps
- **Network Sparsity**: Percentage of weights set to zero (increases during training)
- **Active Edges**: Count of non-zero connections (decreases as structure is pruned)

The spikes in the loss plot occur at rounding steps (every 50 epochs for XOR, every 30 for addition) when weights are discretized to {0, 1}.

### 3. UMAP Embeddings
- 2D projection of hidden layer activations
- Points colored by target values
- Shows how the network learns to represent data internally
- XOR shows clear clustering of the 4 XOR classes
- Addition shows a continuous manifold structure (linear relationship)

## How to Run

```bash
cd v1
pip install -r requirements.txt
python example.py
```

All outputs are automatically saved to `outputs/v1/{task_name}/` directories.

## Key Observations

1. **Structure Discovery Works**: Both networks successfully pruned connections during training, discovering task-appropriate architectures automatically.

2. **Task-Appropriate Learning**: 
   - XOR (non-linear) maintains more hidden node connections
   - Addition (linear) learns strong direct input→output weights

3. **Sparsification**: Networks start fully connected and progressively prune unnecessary connections through the rounding process.

4. **Interpretable Results**: The structure heatmaps clearly show which connections the network chose to keep, making the learned architecture interpretable.

## Conclusion

✅ All v1 visualizations are working correctly and demonstrate the Structure-First Backpropagation algorithm in action!
