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
