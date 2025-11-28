import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_solution(z: np.ndarray, n_circles: int):
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
    plt.show()