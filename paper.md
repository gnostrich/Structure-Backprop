# Structure-Aware Backpropagation: Learning Neural Network Architectures Through Gradient-Based Structural Optimization

**Abstract**
Neural architecture search (NAS) usually treats topology as a discrete choice, separate from weight optimization. We present Structure-Aware Backpropagation, which treats connectivity itself as differentiable: each edge weight is a learnable parameter whose magnitude controls both signal flow and structural retention. By combining task loss with L1 regularization, the method induces sparsity during training, effectively learning topology and parameters in a single gradient-based optimization. We evaluate the approach on XOR classification and arithmetic addition regression and observe automatic edge pruning alongside task learning.

## 1. Introduction
Architecture design is central to deep learning performance, but it is typically handled outside the gradient descent loop. We hypothesize that many architectural decisions can be incorporated directly into gradient-based learning by parameterizing connectivity as continuous variables.

**Contribution**: A unified formulation where edges are learned like weights, and sparsity emerges through regularization.

## 2. Method
We consider a feedforward network with edge weights w^(ℓ)_ij. Connectivity is learned by regularizing these weights.

**Objective**: L = L_task + λ Σ|w^(ℓ)_ij|

**Optimization**: w ← w - η(∂L_task/∂w + λ·sign(w))

**Architecture extraction**: An edge is active if |w| ≥ ε.

## 3. Experiments
**XOR**: [2,4,1] architecture, 74.5% accuracy, 8/12 active edges
**Addition**: [2,8,1] architecture, R²=0.796, 12/16 active edges

## 4. Related Work
NAS methods (ENAS, DARTS) introduce separate architecture parameterization. Our method integrates structure and training into a single continuous optimization.

## 5. Limitations
- XOR at 74.5% suggests local minima
- Only toy problems tested
- Manual λ tuning required

## 6. Conclusion
Structure-Aware Backpropagation demonstrates that topology can be learned by gradient descent when connectivity is treated as a first-class parameter.
