#!/usr/bin/env python3
# ###########################################################
#   ###   2418521        Antony O'Neill                   ###
#   ###   SNAKE RL - PLOT GENERATOR                       ###
#   ###   Last Updated: 08-12-2025                        ###
# ###########################################################

# ===============================================
#   SNAKE RL - generate_plots.py - Results Analysis
# ===============================================


# ===============================================
# 1. IMPORTS
# ===============================================
# Libraries for data loading, visualization, and numerical operations.

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ===============================================
# 2. CONFIGURATION
# ===============================================
# Define directories, experiment parameters, color schemes, and labels for visualization.

RESULTS_DIR = './experiment_results'  # Source directory for all experimental data
PLOTS_DIR = './plots'  # Output directory for generated visualizations

os.makedirs(PLOTS_DIR, exist_ok=True)  # Create plots directory if it doesn't exist

# Non-wall experiments: testing different network architectures and memory sizes
EXPERIMENTS = {
    'baseline': {'label': 'Baseline [256]', 'color': '#2ecc71'},  # Single layer, 100k memory
    'nn_wide': {'label': 'Wide [512]', 'color': '#3498db'},  # Wider single layer
    'nn_increased': {'label': 'Increased [256,128,64]', 'color': '#e74c3c'},  # Three layers
    'nn_deeper': {'label': 'Deeper [512,256,128,64,32]', 'color': '#9b59b6'},  # Five layers
    'memory_small': {'label': 'Small Memory (10k)', 'color': '#f39c12'},  # Reduced memory buffer
    'memory_large': {'label': 'Large Memory (500k)', 'color': '#1abc9c'},  # Expanded memory buffer
}

# Wall experiments: same configurations but with added environmental obstacle
WALL_EXPERIMENTS = {
    'baseline_wall': {'label': 'Baseline + Wall', 'color': '#2ecc71'},  # Standard setup with wall
    'nn_wide_wall': {'label': 'Wide + Wall', 'color': '#3498db'},  # Wider single layer with wall
    'nn_increased_wall': {'label': 'Increased + Wall', 'color': '#e74c3c'},  # Three-layer network with wall
    'nn_deeper_wall': {'label': 'Deeper + Wall', 'color': '#9b59b6'},  # Five-layer network with wall
    'memory_small_wall': {'label': 'Small Memory + Wall', 'color': '#f39c12'},  # 10k memory with wall
    'memory_large_wall': {'label': 'Large Memory + Wall', 'color': '#1abc9c'},  # 500k memory with wall
}


# ===============================================
# 3. HELPER FUNCTIONS
# ===============================================
# Utility functions for loading experimental data and calculating smoothed metrics.

def load_scores(experiment_name):
    """Load scores from experiment folder"""
    path = os.path.join(RESULTS_DIR, experiment_name, 'scores.csv')
    if os.path.exists(path):
        df = pd.read_csv(path)  # Read CSV containing episode-by-episode scores
        return df['Score'].values  # Extract score column as numpy array
    return None  # Return None if file doesn't exist


def load_summary(experiment_name):
    """Load summary metrics from experiment folder"""
    path = os.path.join(RESULTS_DIR, experiment_name, 'summary.csv')
    if os.path.exists(path):
        df = pd.read_csv(path)  # Read summary statistics CSV
        metrics = dict(zip(df['Metric'], df['Value']))  # Convert to dictionary for easy lookup
        return metrics
    return None  # Return None if file doesn't exist


def moving_average(data, window=50):
    """Calculate moving average for smoother curves"""
    # Convert to pandas Series and apply rolling average with specified window size
    # min_periods=1 ensures we get values even at the start where full window isn't available
    return pd.Series(data).rolling(window=window, min_periods=1).mean().values


# ===============================================
# 4. PLOTTING FUNCTIONS
# ===============================================
# Generate comparative visualizations for training curves, performance metrics, and experimental results.

def plot_training_curves_comparison():
    """Plot training curves comparing all non-wall experiments"""
    plt.figure(figsize=(14, 8))

    # Iterate through all standard (non-wall) experiments
    for exp_name, config in EXPERIMENTS.items():
        scores = load_scores(exp_name)
        if scores is not None:
            episodes = range(1, len(scores) + 1)  # Create x-axis values (episode numbers)
            smoothed = moving_average(scores, window=50)  # Apply 50-episode smoothing
            # Plot smoothed learning curve with experiment-specific color and label
            plt.plot(episodes, smoothed, label=config['label'],
                    color=config['color'], linewidth=2, alpha=0.9)

    # Configure plot aesthetics and labels
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Score (50-episode moving average)', fontsize=12)
    plt.title('Training Curves: Neural Network & Memory Variations', fontsize=14)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)  # Add subtle grid for readability
    plt.xlim(0, 500)  # Set x-axis range to full training duration
    plt.tight_layout()  # Optimize spacing to prevent label cutoff
    plt.savefig(os.path.join(PLOTS_DIR, '1_training_curves_comparison.png'), dpi=150)
    plt.close()
    print("Saved: 1_training_curves_comparison.png")


def plot_wall_comparison():
    """Plot training curves for wall experiments"""
    plt.figure(figsize=(14, 8))

    # Iterate through all wall-environment experiments
    for exp_name, config in WALL_EXPERIMENTS.items():
        scores = load_scores(exp_name)
        if scores is not None:
            episodes = range(1, len(scores) + 1)  # Create episode sequence
            smoothed = moving_average(scores, window=50)  # Smooth the data for clarity
            # Plot each wall experiment with matching colors to non-wall counterparts
            plt.plot(episodes, smoothed, label=config['label'],
                    color=config['color'], linewidth=2, alpha=0.9)

    # Configure plot for wall experiments comparison
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Score (50-episode moving average)', fontsize=12)
    plt.title('Training Curves: Wall Environment Experiments', fontsize=14)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 500)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '2_training_curves_wall.png'), dpi=150)
    plt.close()
    print("Saved: 2_training_curves_wall.png")


def plot_nn_architecture_comparison():
    """Compare neural network architectures side by side"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))  # Create side-by-side subplots

    # Left subplot: compare different network architectures without wall
    nn_experiments = ['baseline', 'nn_wide', 'nn_increased', 'nn_deeper']

    ax1 = axes[0]
    for exp_name in nn_experiments:
        config = EXPERIMENTS[exp_name]
        scores = load_scores(exp_name)
        if scores is not None:
            smoothed = moving_average(scores, window=50)  # Apply smoothing
            # Plot each network architecture variant
            ax1.plot(range(1, len(scores)+1), smoothed,
                    label=config['label'], color=config['color'], linewidth=2)

    # Configure left subplot showing network architecture impact in standard environment
    ax1.set_xlabel('Episode', fontsize=11)
    ax1.set_ylabel('Score (50-ep MA)', fontsize=11)
    ax1.set_title('Part B: Neural Network Architectures', fontsize=12)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 500)

    # Right subplot: compare network architectures with wall obstacle
    ax2 = axes[1]
    wall_nn = ['baseline_wall', 'nn_wide_wall', 'nn_increased_wall', 'nn_deeper_wall']
    for exp_name in wall_nn:
        config = WALL_EXPERIMENTS[exp_name]
        scores = load_scores(exp_name)
        if scores is not None:
            smoothed = moving_average(scores, window=50)  # Apply same smoothing
            # Plot wall experiments to show how architectures handle increased complexity
            ax2.plot(range(1, len(scores)+1), smoothed,
                    label=config['label'], color=config['color'], linewidth=2)

    # Configure right subplot showing architecture performance with environmental obstacle
    ax2.set_xlabel('Episode', fontsize=11)
    ax2.set_ylabel('Score (50-ep MA)', fontsize=11)
    ax2.set_title('Part D: NN Architectures + Wall', fontsize=12)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 500)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '3_nn_architecture_comparison.png'), dpi=150)
    plt.close()
    print("Saved: 3_nn_architecture_comparison.png")


def plot_memory_comparison():
    """Compare memory buffer sizes"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))  # Side-by-side comparison

    # Left subplot: memory buffer comparison without wall
    ax1 = axes[0]
    memory_exp = ['baseline', 'memory_small', 'memory_large']
    colors = ['#2ecc71', '#f39c12', '#1abc9c']  # Consistent color scheme
    labels = ['Baseline (100k)', 'Small (10k)', 'Large (500k)']  # Memory sizes

    for exp_name, color, label in zip(memory_exp, colors, labels):
        scores = load_scores(exp_name)
        if scores is not None:
            smoothed = moving_average(scores, window=50)  # Smooth scores
            # Plot each memory configuration to show impact of replay buffer size
            ax1.plot(range(1, len(scores)+1), smoothed,
                    label=label, color=color, linewidth=2)

    # Configure left subplot showing memory buffer size effects
    ax1.set_xlabel('Episode', fontsize=11)
    ax1.set_ylabel('Score (50-ep MA)', fontsize=11)
    ax1.set_title('Part C: Memory Buffer Size', fontsize=12)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 500)

    # Right subplot: memory buffer comparison with wall
    ax2 = axes[1]
    wall_memory = ['baseline_wall', 'memory_small_wall', 'memory_large_wall']
    labels_wall = ['Baseline + Wall', 'Small + Wall', 'Large + Wall']

    for exp_name, color, label in zip(wall_memory, colors, labels_wall):
        scores = load_scores(exp_name)
        if scores is not None:
            smoothed = moving_average(scores, window=50)  # Apply smoothing
            # Plot wall experiments to test if memory effects change with complexity
            ax2.plot(range(1, len(scores)+1), smoothed,
                    label=label, color=color, linewidth=2)

    # Configure right subplot showing memory effects in complex environment
    ax2.set_xlabel('Episode', fontsize=11)
    ax2.set_ylabel('Score (50-ep MA)', fontsize=11)
    ax2.set_title('Part D: Memory Buffer + Wall', fontsize=12)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 500)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '4_memory_comparison.png'), dpi=150)
    plt.close()
    print("Saved: 4_memory_comparison.png")


def plot_final_performance_bar():
    """Bar chart comparing final performance metrics"""
    # Combine all experiments (both standard and wall) for comprehensive comparison
    all_experiments = list(EXPERIMENTS.keys()) + list(WALL_EXPERIMENTS.keys())

    # Initialize lists to collect performance metrics
    records = []  # Maximum score achieved in any episode
    avg_last_100 = []  # Average performance over final 100 episodes
    labels = []  # Experiment names for x-axis
    colors = []  # Bar colors to distinguish wall vs non-wall

    # Extract metrics from each experiment's summary file
    for exp in all_experiments:
        summary = load_summary(exp)
        if summary:
            # Get peak performance metric
            records.append(int(summary.get('Final Record', 0)))
            # Get convergence performance metric
            avg_str = summary.get('Avg Last 100 Episodes', '0')
            avg_last_100.append(float(avg_str))

            # Format experiment names for display (add line breaks for readability)
            if '_wall' in exp:
                labels.append(exp.replace('_wall', '\n+Wall'))
            else:
                labels.append(exp.replace('_', '\n'))

            # Color: blue for no-wall, orange for wall
            colors.append('#3498db' if '_wall' not in exp else '#e67e22')

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))  # Create two bar charts side-by-side

    x = np.arange(len(labels))  # Create numeric positions for bars

    # Left bar chart: peak performance (record scores)
    ax1 = axes[0]
    bars1 = ax1.bar(x, records, color=colors, edgecolor='black', alpha=0.8)
    ax1.set_xlabel('Experiment', fontsize=11)
    ax1.set_ylabel('Record Score', fontsize=11)
    ax1.set_title('Maximum Score Achieved', fontsize=12)  # Shows best single episode
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)  # Horizontal grid lines for easier reading

    # Add value labels on top of each bar for exact numbers
    for bar, val in zip(bars1, records):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(val), ha='center', va='bottom', fontsize=9)

    # Right bar chart: converged performance (last 100 episodes average)
    ax2 = axes[1]
    bars2 = ax2.bar(x, avg_last_100, color=colors, edgecolor='black', alpha=0.8)
    ax2.set_xlabel('Experiment', fontsize=11)
    ax2.set_ylabel('Average Score', fontsize=11)
    ax2.set_title('Average Score (Last 100 Episodes)', fontsize=12)  # Shows stability
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels showing average to one decimal place
    for bar, val in zip(bars2, avg_last_100):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '5_final_performance_bar.png'), dpi=150)
    plt.close()
    print("Saved: 5_final_performance_bar.png")


def plot_baseline_vs_wall():
    """Direct comparison: baseline vs baseline with wall"""
    plt.figure(figsize=(12, 6))

    # Load and plot standard baseline (no obstacles)
    scores = load_scores('baseline')
    if scores is not None:
        smoothed = moving_average(scores, window=50)  # Apply smoothing
        # Green line shows performance in simple environment
        plt.plot(range(1, len(scores)+1), smoothed,
                label='Baseline (No Wall)', color='#2ecc71', linewidth=2.5)

    # Load and plot baseline with wall obstacle
    scores_wall = load_scores('baseline_wall')
    if scores_wall is not None:
        smoothed_wall = moving_average(scores_wall, window=50)  # Apply smoothing
        # Red line shows same configuration but with added environmental complexity
        plt.plot(range(1, len(scores_wall)+1), smoothed_wall,
                label='Baseline (With Wall)', color='#e74c3c', linewidth=2.5)

    # Configure plot to highlight environmental impact on learning
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Score (50-episode moving average)', fontsize=12)
    plt.title('Part D: Impact of Environmental Complexity', fontsize=14)
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 500)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '6_baseline_vs_wall.png'), dpi=150)
    plt.close()
    print("Saved: 6_baseline_vs_wall.png")


def create_summary_table():
    """Create a summary table of all results"""
    # Combine all experiments for comprehensive table
    all_experiments = list(EXPERIMENTS.keys()) + list(WALL_EXPERIMENTS.keys())

    rows = []  # Collect data for each experiment
    for exp in all_experiments:
        summary = load_summary(exp)
        if summary:
            # Extract key metrics and configuration details into table row
            rows.append({
                'Experiment': exp,
                'Architecture': summary.get('Hidden Layers', 'N/A'),  # Network structure
                'Memory': summary.get('Memory Size', 'N/A'),  # Replay buffer size
                'Wall': 'Yes' if 'True' in str(summary.get('Has Wall', '')) else 'No',  # Environment
                'Record': summary.get('Final Record', 'N/A'),  # Best score achieved
                'Mean Score': summary.get('Final Mean Score', 'N/A'),  # Overall average
                'Avg Last 100': summary.get('Avg Last 100 Episodes', 'N/A'),  # Final performance
            })

    # Convert to DataFrame for easy manipulation and export
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(PLOTS_DIR, 'results_summary_table.csv'), index=False)
    print("Saved: results_summary_table.csv")
    print("\n" + df.to_string(index=False))  # Display table in console


# ===============================================
# 5. MAIN
# ===============================================
# Execute all plotting functions to generate complete visualization suite.

if __name__ == '__main__':
    print("Generating plots from experiment results...\n")

    # Generate all visualization outputs
    plot_training_curves_comparison()  # Overall learning curves for non-wall experiments
    plot_wall_comparison()  # Learning curves for wall experiments
    plot_nn_architecture_comparison()  # Side-by-side NN architecture comparisons
    plot_memory_comparison()  # Side-by-side memory buffer comparisons
    plot_final_performance_bar()  # Bar charts showing peak and converged performance
    plot_baseline_vs_wall()  # Direct comparison showing environmental impact
    create_summary_table()  # CSV table with all metrics for reference

    print(f"\nAll plots saved to: {PLOTS_DIR}/")
