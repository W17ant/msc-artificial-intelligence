# ###########################################################
#   ###   2418521        Antony O'Neill                   ###
#   ###   SNAKE RL - HELPER FUNCTIONS                     ###
#   ###   Last Updated: 08-12-2025                        ###
# ###########################################################

# ===============================================
#   SNAKE RL - helper.py - Plotting Utilities
# ===============================================


# ===============================================
# 1. IMPORTS
# ===============================================
# Matplotlib for plotting training progress graphs and OS utilities for file path management.
#
# Changes made to original:
# - Switched from TkAgg to Agg backend to prevent macOS pygame/matplotlib conflict
# - Removed IPython display dependency (was for Jupyter notebooks)
# - Added os import for file path management

import matplotlib
# Use non-GUI backend (Agg) instead of default TkAgg.
# This is required because pygame and matplotlib's default Tkinter backend
# both attempt to control the display, causing crashes on macOS.
# Agg renders plots to image files without needing a display window.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


# ===============================================
# 2. CONFIGURATION
# ===============================================
# Define output directory for saving training plots and ensure it exists.
#
# Changes made to original:
# - NEW SECTION: Added PLOT_DIR constant and directory creation for saving plots

PLOT_DIR = './plots'
if not os.path.exists(PLOT_DIR):  # Create directory if it doesn't exist
    os.makedirs(PLOT_DIR)


# ===============================================
# 3. PLOTTING FUNCTION
# ===============================================
# Main plotting function to visualize score progression and mean scores during training.
#
# Changes made to original:
# - Plots now saved to file (./plots/training_progress.png) instead of displayed
# - Added plt.close() to prevent memory leaks during long training runs
# - Added legend to distinguish Score vs Mean Score lines

def plot(scores, mean_scores):
    """Plot the training progress and save to file"""
    plt.clf()  # Clear the current figure to prepare for new plot

    # Set up plot labels and title
    plt.title('Training Progress')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')

    # Plot both score lines
    plt.plot(scores, label='Score')
    plt.plot(mean_scores, label='Mean Score')

    # Configure plot display options
    plt.ylim(ymin=0)  # Set y-axis minimum to 0
    plt.legend(loc='upper left')  # Add legend in top-left corner

    # Add text annotations for latest values
    if len(scores) > 0:
        plt.text(len(scores) - 1, scores[-1], str(scores[-1]))  # Annotate final score
    if len(mean_scores) > 0:
        plt.text(len(mean_scores) - 1, mean_scores[-1], str(round(mean_scores[-1], 2)))  # Annotate final mean

    # Save plot to file and close to free memory
    plt.savefig(os.path.join(PLOT_DIR, 'training_progress.png'))
    plt.close()  # Close figure to prevent memory leaks
