"""
Examples: Training Data That Encourages Recurrent/Cyclic Topology

This script demonstrates various types of training data that naturally
encourage the emergence of recurrent/cyclic structures in the
Structure-First Backpropagation algorithm.
"""

import torch
import numpy as np
from structure_backprop_recurrent import (
    RecurrentStructureBackpropNetwork,
    train_recurrent_structure_backprop
)


def create_sequence_sum_dataset(n_sequences: int = 100, seq_len: int = 5) -> tuple:
    """
    Task: Cumulative sum over sequences.
    
    Given sequence [x₁, x₂, x₃, ...], output cumulative sum at each step.
    Example: Input [2, 3, 1] -> Output [2, 5, 6]
    
    This requires memory (recurrent connection) to maintain running sum.
    
    Args:
        n_sequences: Number of sequences
        seq_len: Length of each sequence
        
    Returns:
        (X, y) tensors of shape (n_sequences, seq_len, 1)
    """
    X = torch.randn(n_sequences, seq_len, 1)  # Random values
    y = torch.cumsum(X, dim=1)  # Cumulative sum
    
    return X, y


def create_parity_dataset(n_sequences: int = 100, seq_len: int = 8) -> tuple:
    """
    Task: Count parity (odd/even) of 1's seen so far.
    
    Given binary sequence [0, 1, 1, 0, 1], output parity at each step.
    Example: [0, 1, 1, 0, 1] -> [0, 1, 0, 0, 1]
    
    This is a classic finite state machine requiring memory of cumulative count.
    
    Args:
        n_sequences: Number of sequences
        seq_len: Length of each sequence
        
    Returns:
        (X, y) tensors
    """
    X = torch.randint(0, 2, (n_sequences, seq_len, 1)).float()
    y = (torch.cumsum(X, dim=1) % 2).long().squeeze(-1)  # Parity
    
    return X, y


def create_delayed_echo_dataset(n_sequences: int = 100, seq_len: int = 10, delay: int = 3) -> tuple:
    """
    Task: Echo input with a delay.
    
    Given sequence [a, b, c, d, e], output [0, 0, 0, a, b, c, d, e]
    (delays by 'delay' steps)
    
    This requires memory to store past inputs.
    
    Args:
        n_sequences: Number of sequences
        seq_len: Length of each sequence
        delay: Number of steps to delay
        
    Returns:
        (X, y) tensors
    """
    X = torch.randn(n_sequences, seq_len, 1)
    y = torch.zeros_like(X)
    
    if delay < seq_len:
        y[:, delay:, :] = X[:, :-delay, :]
    
    return X, y


def create_running_average_dataset(n_sequences: int = 100, seq_len: int = 8, window: int = 3) -> tuple:
    """
    Task: Compute running average over a window.
    
    At each time step, output the average of the last 'window' inputs.
    
    This requires maintaining a short-term memory buffer.
    
    Args:
        n_sequences: Number of sequences
        seq_len: Length of each sequence
        window: Size of averaging window
        
    Returns:
        (X, y) tensors
    """
    X = torch.randn(n_sequences, seq_len, 1)
    y = torch.zeros_like(X)
    
    for i in range(seq_len):
        start = max(0, i - window + 1)
        y[:, i, :] = X[:, start:i+1, :].mean(dim=1)
    
    return X, y


def create_state_counter_dataset(n_sequences: int = 100, seq_len: int = 10, n_states: int = 3) -> tuple:
    """
    Task: Circular state counter.
    
    Count through states 0, 1, 2, ..., n_states-1, 0, 1, 2, ...
    Each input increments the state counter.
    
    This is a simple finite state machine with cyclic behavior.
    
    Args:
        n_sequences: Number of sequences
        seq_len: Length of each sequence
        n_states: Number of states in the cycle
        
    Returns:
        (X, y) tensors
    """
    # Input doesn't matter much - we just want state progression
    X = torch.ones(n_sequences, seq_len, 1)
    
    # Output is the state at each time step
    y = torch.zeros(n_sequences, seq_len, 1)
    for i in range(seq_len):
        y[:, i, 0] = i % n_states
    
    return X, y


def create_binary_counter_dataset(n_sequences: int = 100, seq_len: int = 8) -> tuple:
    """
    Task: Binary counter.
    
    Each time step, increment a binary counter.
    Input: trigger signal (all 1s)
    Output: current counter value
    
    Example: 0, 1, 2, 3, 4, 5, 6, 7, ...
    
    Requires maintaining state across time steps.
    
    Args:
        n_sequences: Number of sequences
        seq_len: Length of each sequence
        
    Returns:
        (X, y) tensors
    """
    X = torch.ones(n_sequences, seq_len, 1)  # Trigger at each step
    y = torch.arange(seq_len).unsqueeze(0).unsqueeze(-1).repeat(n_sequences, 1, 1).float()
    
    # Normalize to [0, 1] range
    y = y / seq_len
    
    return X, y


def create_sequence_repeat_detection_dataset(
    n_sequences: int = 100, 
    seq_len: int = 10,
    pattern_len: int = 3
) -> tuple:
    """
    Task: Detect when a pattern repeats.
    
    Given a sequence, output 1 when the last 'pattern_len' elements
    match the previous 'pattern_len' elements.
    
    Example with pattern_len=2:
    Input:  [1, 2, 3, 1, 2, ...]
    Output: [0, 0, 0, 0, 1, ...]  (1 when [1,2] repeats)
    
    Requires memory of previous patterns.
    
    Args:
        n_sequences: Number of sequences
        seq_len: Length of each sequence
        pattern_len: Length of pattern to detect
        
    Returns:
        (X, y) tensors
    """
    # Generate random sequences from limited vocabulary
    X = torch.randint(0, 5, (n_sequences, seq_len, 1)).float()
    y = torch.zeros(n_sequences, seq_len, 1)
    
    # Detect repeating patterns
    for i in range(pattern_len * 2, seq_len):
        prev_pattern = X[:, i-pattern_len*2:i-pattern_len, :]
        curr_pattern = X[:, i-pattern_len:i, :]
        
        # Check if patterns match
        matches = (prev_pattern == curr_pattern).all(dim=1).all(dim=1)
        y[:, i, 0] = matches.float()
    
    return X, y


# Example usage and demonstrations
if __name__ == "__main__":
    print("=" * 70)
    print("Training Data Examples for Recurrent/Cyclic Topology")
    print("=" * 70)
    print()
    
    # Example 1: Sequence Summation
    print("1. SEQUENCE SUMMATION (Cumulative Sum)")
    print("-" * 70)
    print("Task: Output cumulative sum at each time step")
    print("Why recurrent: Needs to remember running sum")
    print()
    X, y = create_sequence_sum_dataset(n_sequences=5, seq_len=4)
    print("Example input sequence:")
    print(X[0].squeeze().tolist())
    print("Example output (cumulative sum):")
    print(y[0].squeeze().tolist())
    print()
    
    # Example 2: Parity Checking
    print("2. PARITY CHECKING")
    print("-" * 70)
    print("Task: Output parity of number of 1's seen so far")
    print("Why recurrent: Classic finite state machine (2 states: even/odd)")
    print()
    X, y = create_parity_dataset(n_sequences=5, seq_len=6)
    print("Example input sequence:")
    print(X[0].squeeze().long().tolist())
    print("Example output (parity):")
    print(y[0].tolist())
    print()
    
    # Example 3: Delayed Echo
    print("3. DELAYED ECHO")
    print("-" * 70)
    print("Task: Echo input with a 3-step delay")
    print("Why recurrent: Needs memory buffer to store past inputs")
    print()
    X, y = create_delayed_echo_dataset(n_sequences=5, seq_len=8, delay=3)
    print("Example input sequence:")
    print([f"{x:.2f}" for x in X[0].squeeze().tolist()])
    print("Example output (delayed):")
    print([f"{x:.2f}" for x in y[0].squeeze().tolist()])
    print()
    
    # Example 4: Running Average
    print("4. RUNNING AVERAGE")
    print("-" * 70)
    print("Task: Output average of last 3 values")
    print("Why recurrent: Maintains short-term memory window")
    print()
    X, y = create_running_average_dataset(n_sequences=5, seq_len=6, window=3)
    print("Example input sequence:")
    print([f"{x:.2f}" for x in X[0].squeeze().tolist()])
    print("Example output (running avg):")
    print([f"{x:.2f}" for x in y[0].squeeze().tolist()])
    print()
    
    # Example 5: State Counter
    print("5. CIRCULAR STATE COUNTER")
    print("-" * 70)
    print("Task: Count through states in a cycle (0, 1, 2, 0, 1, 2, ...)")
    print("Why recurrent: Cyclical state progression - natural for cycles!")
    print()
    X, y = create_state_counter_dataset(n_sequences=5, seq_len=9, n_states=3)
    print("Example output (state sequence):")
    print(y[0].squeeze().long().tolist())
    print()
    
    # Example 6: Binary Counter
    print("6. BINARY COUNTER")
    print("-" * 70)
    print("Task: Increment counter at each step (0, 1, 2, 3, ...)")
    print("Why recurrent: Maintains and updates internal counter state")
    print()
    X, y = create_binary_counter_dataset(n_sequences=5, seq_len=8)
    print("Example output (counter):")
    print([f"{x:.2f}" for x in y[0].squeeze().tolist()])
    print()
    
    # Example 7: Pattern Repeat Detection
    print("7. PATTERN REPEAT DETECTION")
    print("-" * 70)
    print("Task: Detect when a pattern repeats")
    print("Why recurrent: Must remember previous pattern to compare")
    print()
    X, y = create_sequence_repeat_detection_dataset(n_sequences=5, seq_len=10, pattern_len=2)
    print("Example input sequence:")
    print(X[0].squeeze().long().tolist())
    print("Example output (1 when pattern repeats):")
    print(y[0].squeeze().long().tolist())
    print()
    
    print("=" * 70)
    print("Key Insight:")
    print("All these tasks share a common property: the output at time t")
    print("depends on PREVIOUS inputs/states, not just the current input.")
    print("This temporal dependency encourages the network to learn")
    print("recurrent connections that can maintain state/memory.")
    print("=" * 70)
