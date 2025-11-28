import numpy as np
from typing import List, Callable

class CirclePackingProblem:
    """
    Defines the math for packing N circles into a square of size L.
    State Vector z structure: [L, x1, y1, x2, y2, ..., xN, yN]
    """

    def __init__(self, n_circles: int):
        self.n_circles = n_circles

    def get_objective(self) -> Callable:
        """Returns the function to minimize: f(z) = L"""
        return lambda z: z[0]

    def get_constraints(self) -> List[Callable]:
        """Generates the list of inequality constraints g(z) <= 0"""
        constraints = []

        # --- A. Boundary Constraints ---
        # Ensure circles stay inside [0, L]
        for i in range(self.n_circles):
            # Capture index i using a factory function to avoid closure bugs
            constraints.extend(self._create_boundary_funcs(i))

        # --- B. Non-Overlap Constraints ---
        # Ensure distance between any pair >= 2 (so distance^2 >= 4)
        for i in range(self.n_circles):
            for j in range(i + 1, self.n_circles):
                constraints.append(self._create_overlap_func(i, j))

        return constraints

    def get_initial_guess(self) -> np.ndarray:
        """Generates a smart starting point (grid layout)"""
        # Start with a loose box size
        L_guess = float(self.n_circles * 2)
        z0 = [L_guess]
        
        # Place circles in a grid to avoid initial overlaps
        grid_dim = int(np.ceil(np.sqrt(self.n_circles)))
        for i in range(self.n_circles):
            row = i // grid_dim
            col = i % grid_dim
            z0.append(2.0 * col + 1.0) # x
            z0.append(2.0 * row + 1.0) # y
        
        return np.array(z0)

    # --- Internal Helper Methods to create closures ---
    def _create_boundary_funcs(self, i: int) -> List[Callable]:
        ix, iy = 1 + 2*i, 2 + 2*i
        
        # 1. Left Wall: 1 - x <= 0  (x >= 1)
        # 2. Bottom Wall: 1 - y <= 0 (y >= 1)
        # 3. Right Wall: x - L + 1 <= 0 (x <= L - 1)
        # 4. Top Wall: y - L + 1 <= 0 (y <= L - 1)
        return [
            lambda z: 1 - z[ix],
            lambda z: 1 - z[iy],
            lambda z: z[ix] - z[0] + 1,
            lambda z: z[iy] - z[0] + 1
        ]

    def _create_overlap_func(self, i: int, j: int) -> Callable:
        ix, iy = 1 + 2*i, 2 + 2*i
        jx, jy = 1 + 2*j, 2 + 2*j
        # 4 - dist^2 <= 0
        return lambda z: 4.0 - ((z[ix] - z[jx])**2 + (z[iy] - z[jy])**2)