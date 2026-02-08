"""
Demonstration: Learning Recurrent Topology from Sequential Data

This script trains the Structure-First Backpropagation algorithm on
sequential tasks that require memory, demonstrating how recurrent/cyclic
connections naturally emerge from the training data patterns.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from structure_backprop_recurrent import (
    RecurrentStructureBackpropNetwork,
    train_recurrent_structure_backprop
)
from recurrent_data_examples import (
    create_sequence_sum_dataset,
    create_parity_dataset,
    create_state_counter_dataset,
    create_binary_counter_dataset
)


def visualize_recurrent_structure(model, title="Learned Recurrent Structure"):
    """
    Visualize the learned structure with emphasis on recurrent connections.
    
    Args:
        model: Trained RecurrentStructureBackpropNetwork
        title: Plot title
    """
    weights = (model.weights * model.structure_mask).detach().cpu().numpy()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Full weight matrix
    im1 = ax1.imshow(weights, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    ax1.set_title(f"{title}\nFull Weight Matrix")
    ax1.set_xlabel("Target Node")
    ax1.set_ylabel("Source Node")
    
    # Add grid lines to separate node types
    ax1.axhline(model.input_range[1] - 0.5, color='black', linewidth=2)
    ax1.axhline(model.hidden_range[1] - 0.5, color='black', linewidth=2)
    ax1.axvline(model.input_range[1] - 0.5, color='black', linewidth=2)
    ax1.axvline(model.hidden_range[1] - 0.5, color='black', linewidth=2)
    
    # Add labels for node types
    node_types = [
        ('Input', model.input_range[0], model.n_input),
        ('Hidden', model.hidden_range[0], model.n_hidden),
        ('Output', model.output_range[0], model.n_output)
    ]
    for label, start, size in node_types:
        ax1.text(-1, start + size/2, label, ha='right', va='center', fontsize=10)
    
    plt.colorbar(im1, ax=ax1, label='Weight Value')
    
    # Hidden-to-Hidden (recurrent) connections only
    h_start, h_end = model.hidden_range
    hidden_weights = weights[h_start:h_end, h_start:h_end]
    
    im2 = ax2.imshow(hidden_weights, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    ax2.set_title(f"Hidden-to-Hidden (Recurrent) Connections\n{len(model.get_recurrent_edges())} active edges")
    ax2.set_xlabel("Target Hidden Node")
    ax2.set_ylabel("Source Hidden Node")
    
    # Annotate non-zero connections
    for i in range(hidden_weights.shape[0]):
        for j in range(hidden_weights.shape[1]):
            if abs(hidden_weights[i, j]) > 0.01:
                color = 'white' if abs(hidden_weights[i, j]) > 0.5 else 'black'
                ax2.text(j, i, f'{hidden_weights[i, j]:.2f}', 
                        ha='center', va='center', color=color, fontsize=8)
    
    plt.colorbar(im2, ax=ax2, label='Weight Value')
    
    plt.tight_layout()
    return fig


def plot_recurrent_training_history(history, title="Training History"):
    """
    Plot training metrics including recurrent edge information.
    
    Args:
        history: Training history dictionary
        title: Plot title
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss
    axes[0, 0].plot(history['loss'])
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Sparsity
    axes[0, 1].plot([s * 100 for s in history['sparsity']])
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Sparsity (%)')
    axes[0, 1].set_title('Network Sparsity')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Active edges
    axes[1, 0].plot(history['active_edges'], label='Total Active')
    axes[1, 0].plot(history['recurrent_edges'], label='Recurrent Only', linestyle='--')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Edge Count')
    axes[1, 0].set_title('Active Edges Over Time')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Cycle detection
    has_cycles_int = [int(h) for h in history['has_cycles']]
    axes[1, 1].plot(has_cycles_int, marker='o', markersize=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Has Cycles (0/1)')
    axes[1, 1].set_title('Cycle Detection Over Training')
    axes[1, 1].set_ylim([-0.1, 1.1])
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def demonstrate_task(task_name, X, y, n_hidden=4, n_epochs=500):
    """
    Train and demonstrate structure learning on a sequential task.
    
    Args:
        task_name: Name of the task
        X: Input sequences
        y: Target sequences
        n_hidden: Number of hidden nodes
        n_epochs: Training epochs
        
    Returns:
        (model, history) tuple
    """
    print(f"\n{'='*70}")
    print(f"TASK: {task_name}")
    print(f"{'='*70}")
    
    # Determine dimensions
    if X.dim() == 3:
        n_input = X.shape[2]
        if y.dim() == 3:
            n_output = y.shape[2]
        else:
            n_output = 1
    else:
        n_input = X.shape[1]
        n_output = 1 if y.dim() == 1 else y.shape[1]
    
    print(f"Input dim: {n_input}, Hidden: {n_hidden}, Output: {n_output}")
    print(f"Training samples: {X.shape[0]}")
    print(f"Sequence length: {X.shape[1] if X.dim() == 3 else 'N/A (single step)'}")
    
    # Create model
    model = RecurrentStructureBackpropNetwork(
        n_input=n_input,
        n_hidden=n_hidden,
        n_output=n_output,
        rounding_threshold=0.3,
        activation='tanh'
    )
    
    print(f"\nInitial structure: {model.get_active_edges()} active edges")
    print(f"Possible recurrent connections: {model.n_hidden * model.n_hidden}")
    
    # Train
    print(f"\nTraining for {n_epochs} epochs...")
    history = train_recurrent_structure_backprop(
        model=model,
        train_data=(X, y),
        n_epochs=n_epochs,
        learning_rate=0.01,
        rounding_frequency=50,
        rounding_method='threshold',
        sequence_mode=(X.dim() == 3),
        verbose=True
    )
    
    # Analyze learned structure
    print(f"\n{'='*70}")
    print("LEARNED STRUCTURE ANALYSIS")
    print(f"{'='*70}")
    print(f"Final loss: {history['loss'][-1]:.4f}")
    print(f"Final sparsity: {history['sparsity'][-1]:.2%}")
    print(f"Total active edges: {history['active_edges'][-1]}")
    print(f"Recurrent edges (hidden→hidden): {history['recurrent_edges'][-1]}")
    print(f"Has cycles: {history['has_cycles'][-1]}")
    
    if history['recurrent_edges'][-1] > 0:
        print("\n✓ RECURRENT STRUCTURE LEARNED!")
        print("  The network discovered that cyclic connections help solve this task.")
    else:
        print("\n✗ No recurrent structure learned")
        print("  The network found a feedforward solution sufficient.")
    
    # Show recurrent edges
    recurrent_edges = model.get_recurrent_edges()
    if recurrent_edges:
        print(f"\nRecurrent edges: {len(recurrent_edges)}")
        for src, tgt in recurrent_edges[:10]:  # Show first 10
            h_src = src - model.hidden_range[0]
            h_tgt = tgt - model.hidden_range[0]
            print(f"  Hidden {h_src} → Hidden {h_tgt}")
        if len(recurrent_edges) > 10:
            print(f"  ... and {len(recurrent_edges) - 10} more")
    
    return model, history


if __name__ == "__main__":
    print("="*70)
    print("DEMONSTRATION: LEARNING RECURRENT TOPOLOGY FROM SEQUENTIAL DATA")
    print("="*70)
    print("\nThis demonstration shows how different types of sequential training")
    print("data naturally encourage the emergence of recurrent/cyclic structures")
    print("in the Structure-First Backpropagation algorithm.")
    print()
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Task 1: Sequence Summation
    X1, y1 = create_sequence_sum_dataset(n_sequences=200, seq_len=6)
    model1, history1 = demonstrate_task(
        "Cumulative Sequence Sum",
        X1, y1,
        n_hidden=4,
        n_epochs=800
    )
    
    # Visualize
    fig1 = visualize_recurrent_structure(model1, "Sequence Sum: Learned Structure")
    fig2 = plot_recurrent_training_history(history1, "Sequence Sum: Training History")
    
    # Task 2: Binary Counter
    X2, y2 = create_binary_counter_dataset(n_sequences=200, seq_len=8)
    model2, history2 = demonstrate_task(
        "Binary Counter",
        X2, y2,
        n_hidden=3,
        n_epochs=800
    )
    
    # Visualize
    fig3 = visualize_recurrent_structure(model2, "Binary Counter: Learned Structure")
    fig4 = plot_recurrent_training_history(history2, "Binary Counter: Training History")
    
    # Task 3: State Counter (Cyclic States)
    X3, y3 = create_state_counter_dataset(n_sequences=200, seq_len=10, n_states=3)
    model3, history3 = demonstrate_task(
        "Circular State Counter (0→1→2→0→...)",
        X3, y3,
        n_hidden=3,
        n_epochs=600
    )
    
    # Visualize
    fig5 = visualize_recurrent_structure(model3, "State Counter: Learned Structure")
    fig6 = plot_recurrent_training_history(history3, "State Counter: Training History")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\nKey Findings:")
    print(f"1. Sequence Sum: {history1['recurrent_edges'][-1]} recurrent edges learned")
    print(f"   → Needed to maintain running sum state")
    print(f"\n2. Binary Counter: {history2['recurrent_edges'][-1]} recurrent edges learned")
    print(f"   → Needed to increment and maintain counter state")
    print(f"\n3. State Counter: {history3['recurrent_edges'][-1]} recurrent edges learned")
    print(f"   → Needed for cyclic state transitions")
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("\nTraining data characteristics that encourage recurrent structures:")
    print("✓ Sequential/temporal nature (not i.i.d. samples)")
    print("✓ Memory requirements (output depends on history)")
    print("✓ State evolution (internal state changes over time)")
    print("✓ Temporal dependencies (current output needs past context)")
    print("\nThe Structure-First Backpropagation algorithm automatically discovers")
    print("that recurrent connections provide computational advantages for these")
    print("tasks, leading to the emergence of cyclic topology through gradient")
    print("descent and weight rounding.")
    print("="*70)
    
    # Show all plots
    plt.show()
