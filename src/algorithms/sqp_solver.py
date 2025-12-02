import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint
from typing import Callable, List

class SQPSolver:
    """
    Sequential Quadratic Programming (SQP) Solver.
    Approximates the problem as a Quadratic Program (QP) at each iteration:
      min  0.5 * d^T * B * d + grad_f^T * d
      s.t. grad_g * d + g <= 0
    """

    def __init__(self, max_iter: int = 50, tol: float = 1e-4, eta: float = 0.2):
        self.max_iter = max_iter
        self.tol = tol
        self.eta = eta  # Parameter for line search (merit function)

    def _numerical_gradient(self, f: Callable, x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Calculates gradient of f at x using finite differences."""
        n = len(x)
        grad = np.zeros(n)
        perturb = np.eye(n) * eps
        for i in range(n):
            grad[i] = (f(x + perturb[i]) - f(x - perturb[i])) / (2 * eps)
        return grad

    def _numerical_jacobian(self, constraints: List[Callable], x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Calculates Jacobian matrix of constraints at x."""
        m = len(constraints)
        n = len(x)
        jac = np.zeros((m, n))
        perturb = np.eye(n) * eps
        for i, g in enumerate(constraints):
            for j in range(n):
                jac[i, j] = (g(x + perturb[j]) - g(x - perturb[j])) / (2 * eps)
        return jac

    def optimize(self, objective: Callable, constraints: List[Callable], x0: np.ndarray) -> np.ndarray:
        n = len(x0)
        x_k = np.array(x0, dtype=float)
        
        # Initialize Hessian approximation (B) as Identity matrix
        B_k = np.eye(n)

        print(f"--- Starting SQP Optimization (Max Iter: {self.max_iter}) ---")

        for k in range(self.max_iter):
            # 1. Evaluate functions and derivatives at current point
            f_k = objective(x_k)
            grad_f_k = self._numerical_gradient(objective, x_k)
            
            # Evaluate constraints (g(x) <= 0)
            g_k = np.array([c(x_k) for c in constraints])
            jac_g_k = self._numerical_jacobian(constraints, x_k)

            # Check Convergence
            max_violation = np.max(np.maximum(0, g_k)) if len(g_k) > 0 else 0.0
            lagrangian_norm = np.linalg.norm(grad_f_k) # Simplified check
            
            if k % 5 == 0 or k == 0:
                print(f"Iter {k:02d} | Obj: {f_k:.4f} | Max Violation: {max_violation:.6e}")

            if max_violation < self.tol and lagrangian_norm < self.tol:
                print(f"Converged at iteration {k}")
                return x_k

            # 2. Define the QP Subproblem to find direction d
            # min 0.5 * d.T * B * d + grad_f.T * d
            # s.t. jac_g * d + g <= 0  ->  jac_g * d <= -g
            
            def qp_obj(d):
                return 0.5 * d @ B_k @ d + grad_f_k @ d
            
            def qp_obj_jac(d):
                return B_k @ d + grad_f_k

            # Linearize constraints: J*d <= -g
            # Scipy LinearConstraint takes format: lb <= A.dot(x) <= ub
            # We want: -inf <= jac_g.dot(d) <= -g_k
            qp_constraints = []
            if len(constraints) > 0:
                qp_constraints.append(LinearConstraint(
                    jac_g_k, 
                    -np.inf * np.ones_like(g_k), 
                    -g_k
                ))

            # Solve QP for direction d_k
            # We constrain the step size slightly to prevent explosion in early iterations
            bounds = Bounds(-10.0, 10.0) 
            res = minimize(qp_obj, np.zeros(n), jac=qp_obj_jac, 
                           constraints=qp_constraints, method='SLSQP', bounds=bounds)
            
            d_k = res.x

            # 3. Step Update (Simple Line Search / Full Step)
            # For this implementation, we take the full step (alpha=1.0)
            # A production solver would use a Merit Function here to determine alpha.
            x_new = x_k + d_k

            # 4. BFGS Update for Hessian B_k
            # Calculate y_k = difference in gradient of Lagrangian
            # Note: For strict SQP we need Lagrange multipliers, here we approximate 
            # using just the objective gradient difference for stability in this simple example.
            grad_f_new = self._numerical_gradient(objective, x_new)
            
            s_k = d_k
            y_k = grad_f_new - grad_f_k # Simplified (Objective-only approximation)

            # Damped BFGS check: s^T * y must be positive
            if s_k @ y_k > 1e-10:
                rho_inv = s_k @ y_k
                # BFGS formula
                term1 = np.outer(B_k @ s_k, s_k @ B_k) / (s_k @ B_k @ s_k)
                term2 = np.outer(y_k, y_k) / rho_inv
                B_k = B_k - term1 + term2

            x_k = x_new

        print("Maximum iterations reached.")
        return x_k