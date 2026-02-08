"""
Structure-First Backpropagation with Recurrent Connections

This module extends the base Structure-First Backpropagation algorithm
to support recurrent/cyclic connections, enabling the learning of
topologies with temporal dependencies and memory.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List


class RecurrentStructureBackpropNetwork(nn.Module):
    """
    A neural network that learns recurrent/cyclic structures through training.
    
    Unlike the feedforward version, this allows:
    - Hidden → Hidden connections (including cycles)
    - Self-connections (node → itself)
    - Temporal processing of sequences
    """
    
    def __init__(
        self,
        n_input: int,
        n_hidden: int,
        n_output: int,
        rounding_threshold: float = 0.5,
        activation: str = 'tanh',
        allow_self_connections: bool = True
    ):
        """
        Initialize a dense directed graph with recurrent connections allowed.
        
        Args:
            n_input: Number of input nodes
            n_hidden: Number of hidden nodes
            n_output: Number of output nodes
            rounding_threshold: Threshold for binary rounding (0-1)
            activation: Activation function ('relu', 'tanh', 'sigmoid')
            allow_self_connections: Allow nodes to connect to themselves
        """
        super().__init__()
        
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_total = n_input + n_hidden + n_output
        self.rounding_threshold = rounding_threshold
        self.allow_self_connections = allow_self_connections
        
        # Define node ranges
        self.input_range = (0, n_input)
        self.hidden_range = (n_input, n_input + n_hidden)
        self.output_range = (n_input + n_hidden, self.n_total)
        
        # Initialize dense adjacency matrix
        self.weights = nn.Parameter(
            torch.randn(self.n_total, self.n_total) * 0.1
        )
        
        # Mask for enforcing recurrent graph structure
        self.register_buffer('structure_mask', self._create_recurrent_mask())
        
        # Set activation function (tanh is better for recurrent nets)
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def _create_recurrent_mask(self) -> torch.Tensor:
        """
        Create a mask for recurrent connections.
        
        Allows connections:
        - Input -> Hidden
        - Input -> Output
        - Hidden -> Hidden (INCLUDING CYCLES - this is the key difference!)
        - Hidden -> Output
        
        Prevents:
        - Output -> anything (terminal nodes)
        - Anything -> Input (inputs are set externally)
        """
        mask = torch.zeros(self.n_total, self.n_total)
        
        # Input nodes can connect to hidden and output
        mask[self.input_range[0]:self.input_range[1], 
             self.hidden_range[0]:self.output_range[1]] = 1
        
        # Hidden nodes can connect to ALL hidden nodes (enables cycles!)
        # and to output nodes
        mask[self.hidden_range[0]:self.hidden_range[1],
             self.hidden_range[0]:self.output_range[1]] = 1
        
        # Optionally disable self-connections
        if not self.allow_self_connections:
            mask.fill_diagonal_(0)
        
        # Output nodes don't connect to anything
        # Nothing connects to input nodes
        
        return mask
    
    def forward_step(
        self, 
        x: torch.Tensor, 
        hidden_state: Optional[torch.Tensor] = None,
        use_rounded: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single forward step through the recurrent network.
        
        Args:
            x: Input tensor of shape (batch_size, n_input)
            hidden_state: Previous hidden state (batch_size, n_hidden)
            use_rounded: If True, use rounded binary weights
            
        Returns:
            (output, new_hidden_state)
        """
        batch_size = x.shape[0]
        
        # Initialize hidden state if not provided
        if hidden_state is None:
            hidden_state = torch.zeros(
                batch_size, self.n_hidden, device=x.device
            )
        
        # Get effective weights
        if use_rounded:
            effective_weights = self.get_rounded_weights()
        else:
            effective_weights = self.weights * self.structure_mask
        
        # Construct full activation vector
        activations = torch.cat([
            x,
            hidden_state,
            torch.zeros(batch_size, self.n_output, device=x.device)
        ], dim=1)
        
        # Compute new hidden state
        # Hidden nodes receive input from: inputs + previous hidden state
        preactivation_hidden = torch.matmul(activations, effective_weights[:, self.hidden_range[0]:self.hidden_range[1]])
        new_hidden = self.activation(preactivation_hidden)
        
        # Update activations with new hidden state
        activations = torch.cat([
            x,
            new_hidden,
            torch.zeros(batch_size, self.n_output, device=x.device)
        ], dim=1)
        
        # Compute output
        preactivation_output = torch.matmul(activations, effective_weights[:, self.output_range[0]:self.output_range[1]])
        output = preactivation_output  # No activation on output (for regression/logits)
        
        return output, new_hidden
    
    def forward(
        self, 
        x: torch.Tensor,
        use_rounded: bool = False,
        return_all_outputs: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through sequences.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_input)
               or (batch_size, n_input) for single step
            use_rounded: If True, use rounded binary weights
            return_all_outputs: If True, return outputs for all time steps
            
        Returns:
            Output tensor
        """
        # Handle single-step input
        if x.dim() == 2:
            output, _ = self.forward_step(x, use_rounded=use_rounded)
            return output
        
        # Handle sequence input
        batch_size, seq_len, _ = x.shape
        hidden_state = None
        outputs = []
        
        for t in range(seq_len):
            output, hidden_state = self.forward_step(
                x[:, t, :], 
                hidden_state,
                use_rounded=use_rounded
            )
            outputs.append(output)
        
        if return_all_outputs:
            return torch.stack(outputs, dim=1)  # (batch, seq, output)
        else:
            return outputs[-1]  # Return only last output
    
    def round_weights(self, method: str = 'threshold') -> None:
        """Round weights to binary values {0, 1}."""
        with torch.no_grad():
            if method == 'threshold':
                mask = torch.abs(self.weights) >= self.rounding_threshold
                self.weights.data = mask.float()
            elif method == 'sigmoid':
                probs = torch.sigmoid(self.weights)
                self.weights.data = (probs >= 0.5).float()
            elif method == 'hard':
                self.weights.data = torch.clamp(torch.round(self.weights), 0, 1)
            else:
                raise ValueError(f"Unknown rounding method: {method}")
            
            # Re-apply structure mask
            self.weights.data *= self.structure_mask
    
    def get_rounded_weights(self) -> torch.Tensor:
        """Get current weights rounded to binary without modifying them."""
        with torch.no_grad():
            mask = torch.abs(self.weights) >= self.rounding_threshold
            rounded = mask.float()
            return rounded * self.structure_mask
    
    def get_sparsity(self) -> float:
        """Calculate the current sparsity (percentage of zero weights)."""
        with torch.no_grad():
            possible_weights = self.structure_mask.sum()
            active_weights = self.weights * self.structure_mask
            zero_weights = ((active_weights == 0) & (self.structure_mask == 1)).sum()
            return (zero_weights / possible_weights).item()
    
    def get_active_edges(self) -> int:
        """Count the number of active (non-zero) edges."""
        with torch.no_grad():
            return ((self.weights * self.structure_mask) != 0).sum().item()
    
    def get_recurrent_edges(self) -> List[Tuple[int, int]]:
        """
        Extract recurrent (cyclic) edges in the hidden layer.
        
        Returns:
            List of (source, target) tuples for hidden-to-hidden connections
        """
        with torch.no_grad():
            active_weights = self.weights * self.structure_mask
            edges = []
            
            # Check hidden-to-hidden connections
            for i in range(self.hidden_range[0], self.hidden_range[1]):
                for j in range(self.hidden_range[0], self.hidden_range[1]):
                    if active_weights[i, j] != 0:
                        edges.append((i, j))
            
            return edges
    
    def has_cycles(self) -> bool:
        """
        Check if the current structure has cycles.
        
        Returns:
            True if cycles exist in the hidden layer
        """
        recurrent_edges = self.get_recurrent_edges()
        if not recurrent_edges:
            return False
        
        # Build adjacency list for hidden nodes only
        n_hidden = self.hidden_range[1] - self.hidden_range[0]
        adj = {i: [] for i in range(n_hidden)}
        
        for src, tgt in recurrent_edges:
            # Convert to 0-indexed hidden node IDs
            src_h = src - self.hidden_range[0]
            tgt_h = tgt - self.hidden_range[0]
            adj[src_h].append(tgt_h)
        
        # DFS to detect cycles
        def has_cycle_dfs(node, visited, rec_stack):
            visited[node] = True
            rec_stack[node] = True
            
            for neighbor in adj[node]:
                if not visited[neighbor]:
                    if has_cycle_dfs(neighbor, visited, rec_stack):
                        return True
                elif rec_stack[neighbor]:
                    return True
            
            rec_stack[node] = False
            return False
        
        visited = {i: False for i in range(n_hidden)}
        rec_stack = {i: False for i in range(n_hidden)}
        
        for node in range(n_hidden):
            if not visited[node]:
                if has_cycle_dfs(node, visited, rec_stack):
                    return True
        
        return False


def train_recurrent_structure_backprop(
    model: RecurrentStructureBackpropNetwork,
    train_data: Tuple[torch.Tensor, torch.Tensor],
    n_epochs: int,
    learning_rate: float = 0.01,
    rounding_frequency: int = 10,
    rounding_method: str = 'threshold',
    sequence_mode: bool = True,
    verbose: bool = True
) -> dict:
    """
    Train a recurrent Structure-First Backpropagation model.
    
    Args:
        model: RecurrentStructureBackpropNetwork instance
        train_data: Tuple of (X_train, y_train) tensors
        n_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        rounding_frequency: Round weights every N steps
        rounding_method: Method for rounding
        sequence_mode: If True, expect sequential data
        verbose: Print training progress
        
    Returns:
        Dictionary with training history
    """
    X_train, y_train = train_data
    
    # Determine loss function
    if y_train.dim() == 3 or (y_train.dim() == 2 and sequence_mode):
        criterion = nn.MSELoss()  # For sequences
    elif y_train.shape[-1] == 1:
        criterion = nn.MSELoss()  # Regression
    else:
        criterion = nn.CrossEntropyLoss()  # Classification
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    history = {
        'loss': [],
        'sparsity': [],
        'active_edges': [],
        'recurrent_edges': [],
        'has_cycles': []
    }
    
    step_count = 0
    
    for epoch in range(n_epochs):
        # Forward pass
        if sequence_mode and X_train.dim() == 3:
            predictions = model(X_train, return_all_outputs=True)
        else:
            predictions = model(X_train)
        
        # Compute loss
        loss = criterion(predictions, y_train)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Periodic weight rounding
        if step_count % rounding_frequency == 0 and step_count > 0:
            model.round_weights(method=rounding_method)
        
        step_count += 1
        
        # Record metrics
        history['loss'].append(loss.item())
        history['sparsity'].append(model.get_sparsity())
        history['active_edges'].append(model.get_active_edges())
        history['recurrent_edges'].append(len(model.get_recurrent_edges()))
        history['has_cycles'].append(model.has_cycles())
        
        # Print progress
        if verbose and (epoch % max(1, n_epochs // 10) == 0 or epoch == n_epochs - 1):
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.4f} | "
                  f"Sparsity: {model.get_sparsity():.2%} | "
                  f"Active Edges: {model.get_active_edges()} | "
                  f"Recurrent Edges: {len(model.get_recurrent_edges())} | "
                  f"Has Cycles: {model.has_cycles()}")
    
    return history


if __name__ == "__main__":
    print("Recurrent Structure-First Backpropagation module loaded successfully.")
    print("This module supports learning of cyclic/recurrent topologies.")
