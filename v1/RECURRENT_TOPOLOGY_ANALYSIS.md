# Training Data for Recurrent/Cyclic Topology

## Context

This document addresses the question: **"What training data would shape the topology into a unitary recurrent/cyclic structure?"**

In the context of Structure-First Backpropagation, where network topology is learned during training through gradient descent and weight rounding, certain types of training data naturally encourage the emergence of cyclic (recurrent) connections.

## What is a Recurrent/Cyclic Structure?

A **recurrent or cyclic structure** in a neural network contains at least one cycle in the directed graph, where information can flow in a loop:
- Hidden node H₁ → Hidden node H₂ → Hidden node H₁ (forming a cycle)
- Or self-connections: Hidden node H₁ → Hidden node H₁

This differs from feedforward structures where information flows strictly forward without loops.

## Types of Training Data That Encourage Recurrent Structures

### 1. **Sequential/Temporal Data with Memory Requirements**

**Characteristic**: Data where the output at time *t* depends on previous inputs or states.

**Examples**:
- **Time series prediction**: Stock prices, weather patterns, sensor readings
- **Language modeling**: Next word prediction requires remembering previous context
- **Control sequences**: Robot actions where current action depends on previous state

**Why it encourages cycles**:
Recurrent connections allow the network to maintain an internal state/memory. When trained on sequences, the gradient descent will favor structures that can "remember" previous information through cyclic connections rather than having to recompute everything from scratch.

**Specific Pattern**:
```
Input: x₁, x₂, x₃, x₄, ...  (sequence)
Output: y₁, y₂, y₃, y₄, ...  (where yᵢ depends on x₁...xᵢ)
```

### 2. **State Machine / Finite Automata Tasks**

**Characteristic**: Tasks that require tracking internal state across multiple steps.

**Examples**:
- **Parity checking**: Output 1 if number of 1's seen so far is odd
- **Balanced parentheses**: Check if parentheses are balanced
- **Pattern recognition**: Detect specific sequences like "010" in binary stream

**Why it encourages cycles**:
These tasks explicitly require memory of past observations. A cyclic structure can encode the current state in hidden node activations, with cycles updating the state based on new inputs.

**Specific Pattern**:
```
Input sequence: [0, 1, 0, 1, 0]
Output at each step depends on cumulative history:
  After [0]: state = S₀
  After [0,1]: state = S₁  
  After [0,1,0]: state = S₂
  etc.
```

### 3. **Fixed-Point / Equilibrium Problems**

**Characteristic**: Problems where the answer is a fixed point that emerges from iterative computation.

**Examples**:
- **Graph algorithms**: Shortest paths, PageRank (iterative until convergence)
- **Constraint satisfaction**: Sudoku, SAT problems
- **Dynamical systems**: Finding stable equilibria

**Why it encourages cycles**:
The network can learn to iteratively refine its solution through recurrent cycles until reaching a stable state. Each cycle through the network performs one iteration of refinement.

**Specific Pattern**:
```
Input: Problem specification (graph, constraints)
Output: Solution that satisfies convergence criterion
Process: Network iterates internally through cycles until stable
```

### 4. **Contextual Dependencies / Long-Range Dependencies**

**Characteristic**: Output depends on relationships between distant parts of the input.

**Examples**:
- **Sentence understanding**: "The cat, which was sleeping, woke up" (cat-woke relationship)
- **Music generation**: Maintaining theme across measures
- **Code generation**: Matching opening/closing brackets across lines

**Why it encourages cycles**:
Recurrent connections allow information to persist and be accessed later, enabling the network to maintain context without requiring exponentially growing feedforward paths.

**Specific Pattern**:
```
Input: [w₁, w₂, ..., w₁₀₀]  (long sequence)
Output: Depends on relationship between w₅ and w₉₅
Requirement: Network must "remember" w₅ until processing w₉₅
```

### 5. **Oscillatory or Periodic Patterns**

**Characteristic**: Data exhibiting cyclical behavior or repetition.

**Examples**:
- **Circadian rhythms**: 24-hour cycles in biological data
- **Seasonal patterns**: Yearly cycles in sales, weather
- **Waveform generation**: Sine waves, musical tones

**Why it encourages cycles**:
The network can learn to generate or track periodic patterns through cyclic connections that naturally create oscillatory behavior in activations.

**Specific Pattern**:
```
Input: Time t
Output: sin(ωt) or periodic function
Network learns: Cyclic structure that oscillates at frequency ω
```

## Minimal Example: Sequence Summation

A simple task that demonstrates the need for recurrent structure:

**Task**: Given a sequence of numbers, output the cumulative sum at each step.

```
Input:  [2, 3, 1, 4]
Output: [2, 5, 6, 10]
```

**Why recurrent**:
- At each step, network needs previous sum
- Cyclic connection from hidden node to itself can maintain running sum
- Pure feedforward would need separate path for each sequence position

## Unitary Recurrent Structure

A "unitary" recurrent structure typically means:
1. **Single cycle**: One primary recurrent loop (not many overlapping cycles)
2. **Simple connectivity**: Clean, minimal structure (not densely recurrent)
3. **Functional cycle**: The cycle serves a clear computational purpose

To encourage a **unitary** (single, clean) recurrent structure rather than many complex cycles:
- Use tasks with **single state variable** to track
- Prefer **short-term memory** needs (avoid complex long-term dependencies)
- Use **regular, predictable** patterns (not complex irregular sequences)

## Training Data Design Guidelines

To encourage recurrent/cyclic topology emergence:

### DO:
✓ Use sequential data with temporal dependencies  
✓ Require memory of past inputs/states  
✓ Present data in temporal order during training  
✓ Use tasks where state naturally evolves over time  
✓ Start with simple, regular patterns before complex ones  

### DON'T:
✗ Use independent, i.i.d. samples (encourages feedforward)  
✗ Shuffle sequential data randomly (breaks temporal structure)  
✗ Use tasks solvable by instantaneous input mapping  
✗ Provide full context in each input (removes need for memory)  

## Implementation Requirements

To enable learning of recurrent structures, the architecture must:

1. **Allow cyclic connections** in the structure mask:
   - Hidden → Hidden (including earlier layers)
   - Self-connections (node → itself)

2. **Handle temporal processing**:
   - Process sequences step-by-step
   - Maintain hidden state across time steps
   - Backpropagate through time (BPTT)

3. **Regularization**:
   - Still apply L1 regularization to encourage sparsity
   - Prevent trivial "all connected" solutions

## Expected Outcomes

When trained on appropriate sequential/temporal data, the Structure-First Backpropagation algorithm should learn:

- **Sparse recurrent connections** (not dense recurrence)
- **Task-specific cycles** (cycles that serve the computational need)
- **Minimal topology** (only necessary recurrent connections)

The exact structure depends on the task:
- **Simple counting**: Single self-loop on one hidden node
- **State tracking**: Small cycles between 2-3 hidden nodes
- **Complex sequences**: Multiple specialized cycles for different aspects

## Conclusion

Training data that encourages recurrent/cyclic topology has these key characteristics:
1. **Sequential/temporal nature** - order matters
2. **Memory requirement** - past affects future
3. **Stateful computation** - internal state evolves
4. **Dependencies across time** - current output needs previous context

The most straightforward examples are:
- **Sequence prediction** (next element depends on history)
- **Cumulative operations** (sum, count, state updates)
- **Pattern recognition in sequences** (detecting temporal patterns)

By presenting such data during training, the Structure-First Backpropagation algorithm will discover that cyclic connections provide computational advantages, leading to the emergence of recurrent topology through gradient descent and weight rounding.
