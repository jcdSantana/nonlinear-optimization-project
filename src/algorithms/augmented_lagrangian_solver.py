import numpy as np
from scipy.optimize import minimize
from typing import Callable, List

class AugmentedLagrangianSolver:
    """
    Generic Solver for inequality constrained problems:
    min f(x) subject to g_i(x) <= 0
    """

    def __init__(self, rho: float = 1.0, rho_multiplier: float = 1.2, 
                 max_iter: int = 50, tol: float = 1e-4):
        self.rho = rho
        self.rho_multiplier = rho_multiplier
        self.max_iter = max_iter
        self.tol = tol

    def optimize(self, objective: Callable, constraints: List[Callable], x0: np.ndarray) -> np.ndarray:
        x_k = np.array(x0, dtype=float)
        lmbda = np.zeros(len(constraints))  # Lagrange Multipliers

        print(f"--- Starting Optimization (Max Iter: {self.max_iter}) ---")

        for k in range(self.max_iter):
            
            # 1. Define the Subproblem (PHR Method)
            def subproblem(x):
                lagrangian_sum = 0
                for i, g in enumerate(constraints):
                    g_val = g(x)
                    # PHR Term: (rho/2) * [max(0, g(x) + lambda/rho)]^2
                    term = max(0, g_val + (lmbda[i] / self.rho))
                    lagrangian_sum += (self.rho / 2.0) * (term ** 2)
                
                # We minimize f(x) + Penalty
                return objective(x) + lagrangian_sum

            # 2. Solve Unconstrained Subproblem (Inner Loop)
            res = minimize(subproblem, x_k, method='BFGS', tol=1e-4)
            x_k = res.x

            # 3. Check Convergence (Constraint Violation)
            g_values = np.array([g(x_k) for g in constraints])
            max_violation = np.max(np.maximum(0, g_values))
            
            # Print status every few iterations
            if k % 5 == 0 or k == 0:
                print(f"Iter {k:02d} | Objective: {objective(x_k):.4f} | Max Violation: {max_violation:.6e}")

            if max_violation < self.tol:
                print(f"Converged at iteration {k}")
                return x_k

            # 4. Update Multipliers and Rho
            lmbda = np.maximum(0, lmbda + self.rho * g_values)
            self.rho *= self.rho_multiplier

        print("Maximum iterations reached.")
        return x_k