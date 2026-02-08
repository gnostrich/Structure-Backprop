# Recurrent Topology: Training Data Analysis

## Problem Statement

**Question**: In the context of the Structure-First Backpropagation paper, what training data would shape the topology into a unitary recurrent/cyclic structure?

## Summary Answer

Training data that encourages **recurrent/cyclic structures** has these key characteristics:

1. **Sequential/Temporal Nature**: Data points must be ordered in time or sequence
2. **Memory Requirements**: Current outputs depend on previous inputs or states
3. **Stateful Computation**: An internal state evolves over time
4. **Temporal Dependencies**: Context from the past affects current predictions

### Specific Examples

**Best candidates for encouraging unitary recurrent structures:**

1. **Cumulative Operations**
   - Task: Cumulative sum, running average, counter
   - Why: Single state variable to maintain → encourages single recurrent loop
   - Example: Input `[2, 3, 1, 4]` → Output `[2, 5, 6, 10]`

2. **Simple State Machines**
   - Task: Parity checking, binary counter, circular state
   - Why: Clean state transitions → minimal cyclic structure
   - Example: Parity of bits seen so far (alternates between even/odd state)

3. **Delayed Echo/Memory**
   - Task: Repeat input after N steps
   - Why: Short-term memory buffer → focused recurrent connections
   - Example: Echo with 3-step delay

## Implementation

The implementation includes:

### 1. Theoretical Analysis (`v1/RECURRENT_TOPOLOGY_ANALYSIS.md`)
- Comprehensive explanation of what makes training data encourage cycles
- Types of sequential/temporal patterns
- Design guidelines for creating such datasets
- Comparison with feedforward-encouraging data

### 2. Extended Architecture (`v1/structure_backprop_recurrent.py`)
- `RecurrentStructureBackpropNetwork`: Allows hidden→hidden connections including cycles
- Support for sequential data processing with hidden state
- Cycle detection and analysis methods
- Tracks recurrent edge emergence during training

### 3. Example Training Data (`v1/recurrent_data_examples.py`)
Seven example tasks demonstrating different patterns:
- Sequence summation (cumulative operations)
- Parity checking (2-state machine)
- Delayed echo (memory buffer)
- Running average (sliding window)
- State counter (cyclic states)
- Binary counter (incrementing state)
- Pattern repeat detection (pattern memory)

### 4. Full Demonstration (`v1/demo_recurrent_learning.py`)
- Trains models on sequential tasks
- Visualizes emergence of recurrent connections
- Shows that cycles appear automatically through gradient descent
- Includes analysis of learned structures

## Key Findings

When trained on appropriate sequential data:

1. **Recurrent connections emerge automatically** through the Structure-First Backpropagation algorithm
2. **Sparse recurrent structures** form (not densely connected)
3. **Task-specific cycles** develop based on computational needs
4. **Unitary structures** (clean, minimal cycles) emerge from simple sequential tasks

### Example Result

Training on cumulative sum (50 sequences, length 4):
```
Initial: 16 active edges, 0 recurrent edges
After training (200 epochs):
  - Final loss: 0.2414
  - Total active edges: 16
  - Recurrent edges: 9 (hidden→hidden connections)
  - Has cycles: True ✓
```

The network discovered that maintaining state through recurrent connections helps solve the cumulative sum task.

## Contrast: What Training Data Does NOT Encourage Cycles?

Training data that encourages **feedforward structures**:
- ✗ Independent, i.i.d. samples (no temporal order)
- ✗ Static pattern recognition (XOR, image classification)
- ✗ Instantaneous input-output mapping (no memory needed)
- ✗ Randomly shuffled sequential data (breaks temporal structure)

## How to Use

### Quick Start
```bash
cd v1

# See examples of sequential data patterns
python recurrent_data_examples.py

# Run full training demonstration
python demo_recurrent_learning.py
```

### Create Your Own Sequential Task
```python
from structure_backprop_recurrent import (
    RecurrentStructureBackpropNetwork,
    train_recurrent_structure_backprop
)

# Create sequential task (e.g., cumulative sum)
X = torch.randn(100, 8, 1)  # (n_sequences, seq_len, n_features)
y = torch.cumsum(X, dim=1)  # Cumulative sum

# Create recurrent network
model = RecurrentStructureBackpropNetwork(
    n_input=1,
    n_hidden=4,
    n_output=1,
    activation='tanh'
)

# Train - recurrent structure will emerge
history = train_recurrent_structure_backprop(
    model=model,
    train_data=(X, y),
    n_epochs=500,
    sequence_mode=True
)

# Check learned structure
print(f"Recurrent edges: {len(model.get_recurrent_edges())}")
print(f"Has cycles: {model.has_cycles()}")
```

## Connection to Paper

This work extends the paper's future direction (Section 5.3):
> "**Future Directions**  
> - Extension to recurrent connections"

We've demonstrated:
1. What modifications enable recurrent connections (structure mask changes)
2. What training data naturally encourages their emergence
3. How the same gradient-based structure learning applies to temporal tasks

The key insight remains: **network topology emerges from training data patterns through gradient descent**. Sequential/temporal data patterns naturally lead to recurrent topologies, just as static patterns lead to feedforward topologies.

## Files Added

- **`v1/RECURRENT_TOPOLOGY_ANALYSIS.md`**: Full theoretical analysis (8KB)
- **`v1/structure_backprop_recurrent.py`**: Recurrent network implementation (14KB)
- **`v1/recurrent_data_examples.py`**: Seven example task generators (10KB)
- **`v1/demo_recurrent_learning.py`**: Complete training demonstrations (11KB)
- **`v1/README.md`**: Updated with recurrent section

Total: ~40KB of new documentation and code.

## Conclusion

**Answer to the original question:**

Training data should have **sequential/temporal dependencies** where:
- Output at time *t* depends on inputs at times *t-1*, *t-2*, etc.
- An internal state must be maintained across time steps
- Memory of past observations is required

Examples include: time series prediction, cumulative operations (sum, count), state machines (parity, counters), delayed echo, and any task where context from the past informs current predictions.

The Structure-First Backpropagation algorithm, when presented with such data, will automatically discover that recurrent connections provide computational advantages and will learn sparse cyclic topologies through gradient descent and weight rounding.
