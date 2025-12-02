import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

def ensure_data_folder_exists():
    """
    Creates 'data' directory in the project root safely.
    Path logic: src/utils/plot.py -> src/utils -> src -> [ROOT] -> data
    """
    # Get the directory where this script (plot.py) is located
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Go up two levels to find the project root
    project_root = os.path.dirname(os.path.dirname(current_script_dir))
    
    data_path = os.path.join(project_root, 'data')
    
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    return data_path

def plot_solution(z: np.ndarray, n_circles: int, filename: str = None):
    L = z[0]
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Draw Container
    rect = patches.Rectangle((0, 0), L, L, linewidth=3, edgecolor='black', facecolor='none')
    ax.add_patch(rect)
    
    # Draw Circles
    cmap = plt.get_cmap("viridis")
    for i in range(n_circles):
        x = z[1 + 2*i]
        y = z[2 + 2*i]
        color = cmap(i / n_circles)
        
        circle = patches.Circle((x, y), radius=1, edgecolor='black', facecolor=color, alpha=0.6)
        ax.add_patch(circle)
        ax.text(x, y, str(i+1), ha='center', va='center', color='white', weight='bold')

    ax.set_xlim(-0.5, L + 0.5)
    ax.set_ylim(-0.5, L + 0.5)
    ax.set_aspect('equal')
    ax.set_title(f"Optimal Packing N={n_circles}, Size L={L:.4f}")
    plt.grid(True, linestyle='--', alpha=0.5)

    if filename:
        # Get the correct data path regardless of where the script is run from
        data_dir = ensure_data_folder_exists()
        save_path = os.path.join(data_dir, filename)
        
        plt.savefig(save_path)
        plt.close()
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()