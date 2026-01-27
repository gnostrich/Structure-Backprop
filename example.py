"""
Example: Structure-First Backpropagation Demo

This script demonstrates the Structure-First Backpropagation algorithm
on a simple task (XOR problem) to show how structure emerges during training.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from structure_backprop import StructureBackpropNetwork, train_structure_backprop


def create_xor_dataset(n_samples: int = 100) -> tuple:
    """
    Create XOR dataset for testing structure learning.
    
    XOR is a classic non-linearly separable problem that requires
    at least one hidden layer to solve.
    
    Args:
        n_samples: Number of samples per class
        
    Returns:
        (X, y) tensors
    """
    # Create XOR data: [0,0]->0, [0,1]->1, [1,0]->1, [1,1]->0
    X = []
    y = []
    
    for _ in range(n_samples // 4):
        X.extend([[0, 0], [0, 1], [1, 0], [1, 1]])
        y.extend([[0], [1], [1], [0]])
    
    # Add some noise
    X = np.array(X, dtype=np.float32)
    X += np.random.randn(*X.shape) * 0.1
    y = np.array(y, dtype=np.float32)
    
    return torch.tensor(X), torch.tensor(y)


def create_addition_dataset(n_samples: int = 200) -> tuple:
    """
    Create a simple addition dataset: output = input1 + input2
    
    This is a linear problem that shouldn't require hidden nodes,
    so we expect the structure to learn direct input->output connections.
    
    Args:
        n_samples: Number of samples
        
    Returns:
        (X, y) tensors
    """
    X = torch.randn(n_samples, 2)
    y = X.sum(dim=1, keepdim=True)
    return X, y


def visualize_structure(model: StructureBackpropNetwork, title: str = "Learned Structure"):
    """
    Visualize the learned graph structure.
    
    Args:
        model: Trained StructureBackpropNetwork
        title: Plot title
    """
    with torch.no_grad():
        weights = (model.weights * model.structure_mask).numpy()
    
    plt.figure(figsize=(10, 8))
    plt.imshow(weights, cmap='RdBu', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(label='Weight Value')
    
    # Add grid lines to separate node types
    plt.axhline(y=model.input_range[1] - 0.5, color='black', linewidth=2)
    plt.axhline(y=model.hidden_range[1] - 0.5, color='black', linewidth=2)
    plt.axvline(x=model.input_range[1] - 0.5, color='black', linewidth=2)
    plt.axvline(x=model.hidden_range[1] - 0.5, color='black', linewidth=2)
    
    # Add labels
    plt.ylabel('Source Node')
    plt.xlabel('Target Node')
    plt.title(title)
    
    # Add text annotations for node types
    y_labels = ['Input'] * model.n_input + ['Hidden'] * model.n_hidden + ['Output'] * model.n_output
    x_labels = y_labels
    
    plt.yticks(range(model.n_total), 
               [f"{y_labels[i]}_{i}" for i in range(model.n_total)],
               fontsize=8)
    plt.xticks(range(model.n_total),
               [f"{x_labels[i]}_{i}" for i in range(model.n_total)],
               rotation=90, fontsize=8)
    
    plt.tight_layout()
    return plt.gcf()


def plot_training_history(history: dict, title: str = "Training History"):
    """
    Plot training metrics over time.
    
    Args:
        history: Dictionary with 'loss', 'sparsity', 'active_edges'
        title: Plot title
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss
    axes[0].plot(history['loss'])
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True, alpha=0.3)
    
    # Sparsity
    axes[1].plot(history['sparsity'])
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Sparsity (%)')
    axes[1].set_title('Network Sparsity')
    axes[1].grid(True, alpha=0.3)
    
    # Active edges
    axes[2].plot(history['active_edges'])
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Number of Edges')
    axes[2].set_title('Active Edges')
    axes[2].grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def print_structure_summary(model: StructureBackpropNetwork):
    """Print a summary of the learned structure."""
    print("\n" + "="*60)
    print("LEARNED STRUCTURE SUMMARY")
    print("="*60)
    
    edges = model.get_structure()
    
    # Categorize edges
    input_to_hidden = []
    input_to_output = []
    hidden_to_hidden = []
    hidden_to_output = []
    
    for src, tgt in edges:
        src_type = 'input' if src < model.n_input else ('hidden' if src < model.n_input + model.n_hidden else 'output')
        tgt_type = 'input' if tgt < model.n_input else ('hidden' if tgt < model.n_input + model.n_hidden else 'output')
        
        if src_type == 'input' and tgt_type == 'hidden':
            input_to_hidden.append((src, tgt - model.n_input))
        elif src_type == 'input' and tgt_type == 'output':
            input_to_output.append((src, tgt - model.n_input - model.n_hidden))
        elif src_type == 'hidden' and tgt_type == 'hidden':
            hidden_to_hidden.append((src - model.n_input, tgt - model.n_input))
        elif src_type == 'hidden' and tgt_type == 'output':
            hidden_to_output.append((src - model.n_input, tgt - model.n_input - model.n_hidden))
    
    print(f"Total active edges: {len(edges)}")
    print(f"Sparsity: {model.get_sparsity():.2%}")
    print()
    
    print(f"Input → Hidden: {len(input_to_hidden)} edges")
    for src, tgt in input_to_hidden[:5]:  # Show first 5
        print(f"  Input_{src} → Hidden_{tgt}")
    if len(input_to_hidden) > 5:
        print(f"  ... and {len(input_to_hidden) - 5} more")
    print()
    
    print(f"Input → Output: {len(input_to_output)} edges")
    for src, tgt in input_to_output:
        print(f"  Input_{src} → Output_{tgt}")
    print()
    
    print(f"Hidden → Hidden: {len(hidden_to_hidden)} edges")
    for src, tgt in hidden_to_hidden[:5]:
        print(f"  Hidden_{src} → Hidden_{tgt}")
    if len(hidden_to_hidden) > 5:
        print(f"  ... and {len(hidden_to_hidden) - 5} more")
    print()
    
    print(f"Hidden → Output: {len(hidden_to_output)} edges")
    for src, tgt in hidden_to_output:
        print(f"  Hidden_{src} → Output_{tgt}")
    
    print("="*60 + "\n")


def demo_xor():
    """Demonstrate structure learning on XOR problem."""
    print("\n" + "="*60)
    print("DEMO 1: XOR PROBLEM")
    print("="*60)
    print("XOR is non-linearly separable and requires hidden nodes.")
    print("Expected: Network should learn to use hidden nodes.\n")
    
    # Create dataset
    X, y = create_xor_dataset(n_samples=200)
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Create model
    model = StructureBackpropNetwork(
        n_input=2,
        n_hidden=4,
        n_output=1,
        rounding_threshold=0.3,
        activation='relu'
    )
    
    print(f"Model: {model.n_input} input, {model.n_hidden} hidden, {model.n_output} output nodes")
    print(f"Initial active edges: {model.get_active_edges()}")
    
    # Train
    history = train_structure_backprop(
        model=model,
        train_data=(X, y),
        n_epochs=500,
        learning_rate=0.01,
        rounding_frequency=50,
        rounding_method='threshold',
        verbose=True
    )
    
    # Evaluate
    with torch.no_grad():
        predictions = model(X)
        final_loss = torch.nn.functional.mse_loss(predictions, y)
        accuracy = ((predictions > 0.5) == (y > 0.5)).float().mean()
    
    print(f"\nFinal Loss: {final_loss.item():.4f}")
    print(f"Accuracy: {accuracy.item():.2%}")
    
    # Show structure
    print_structure_summary(model)
    
    # Visualize
    try:
        fig1 = visualize_structure(model, "XOR Problem - Learned Structure")
        fig1.savefig('xor_structure.png', dpi=150, bbox_inches='tight')
        print("Structure visualization saved to: xor_structure.png")
        
        fig2 = plot_training_history(history, "XOR Problem - Training History")
        fig2.savefig('xor_history.png', dpi=150, bbox_inches='tight')
        print("Training history saved to: xor_history.png")
        
        plt.close('all')
    except Exception as e:
        print(f"Visualization skipped (display not available): {e}")
    
    return model, history


def demo_addition():
    """Demonstrate structure learning on simple addition problem."""
    print("\n" + "="*60)
    print("DEMO 2: ADDITION PROBLEM (LINEAR)")
    print("="*60)
    print("Task: output = input1 + input2 (linear, no hidden nodes needed)")
    print("Expected: Network should learn direct input→output connections.\n")
    
    # Create dataset
    X, y = create_addition_dataset(n_samples=200)
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Create model
    model = StructureBackpropNetwork(
        n_input=2,
        n_hidden=3,
        n_output=1,
        rounding_threshold=0.3,
        activation='relu'
    )
    
    print(f"Model: {model.n_input} input, {model.n_hidden} hidden, {model.n_output} output nodes")
    print(f"Initial active edges: {model.get_active_edges()}")
    
    # Train
    history = train_structure_backprop(
        model=model,
        train_data=(X, y),
        n_epochs=300,
        learning_rate=0.01,
        rounding_frequency=30,
        rounding_method='threshold',
        verbose=True
    )
    
    # Evaluate
    with torch.no_grad():
        predictions = model(X)
        final_loss = torch.nn.functional.mse_loss(predictions, y)
    
    print(f"\nFinal Loss: {final_loss.item():.4f}")
    
    # Show structure
    print_structure_summary(model)
    
    # Visualize
    try:
        fig1 = visualize_structure(model, "Addition Problem - Learned Structure")
        fig1.savefig('addition_structure.png', dpi=150, bbox_inches='tight')
        print("Structure visualization saved to: addition_structure.png")
        
        fig2 = plot_training_history(history, "Addition Problem - Training History")
        fig2.savefig('addition_history.png', dpi=150, bbox_inches='tight')
        print("Training history saved to: addition_history.png")
        
        plt.close('all')
    except Exception as e:
        print(f"Visualization skipped (display not available): {e}")
    
    return model, history


if __name__ == "__main__":
    print("\n" + "="*60)
    print("STRUCTURE-FIRST BACKPROPAGATION - DEMONSTRATIONS")
    print("="*60)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run demonstrations
    xor_model, xor_history = demo_xor()
    addition_model, addition_history = demo_addition()
    
    print("\n" + "="*60)
    print("ALL DEMONSTRATIONS COMPLETED")
    print("="*60)
    print("\nKey Observations:")
    print("1. XOR (non-linear) requires hidden nodes - structure should reflect this")
    print("2. Addition (linear) may learn direct input→output paths")
    print("3. Structure sparsity increases during training as weak connections are pruned")
    print("4. The algorithm discovers problem-appropriate architectures automatically")
