import numpy as np

class Newton_Method:
    def __init__(self, max_iter=100, tol=1e-6):
        """
        initializes the Newton Method optimizer.
            param max_iter: Maximum number of iterations
            param tol: Tolerance for convergence
        """
        self.max_iter = max_iter
        self.tol = tol

    def optimize(self, func, gradient, hessian, x0):
        x_k = np.array(x0, dtype=float)

        for i in range(self.max_iter):
            grad_k = np.array(gradient(x_k))
            hess_k = np.array(hessian(x_k))

            # Check for convergence (if gradient is close to zero)
            if np.linalg.norm(grad_k) < self.tol:
                print(f"Converged in {i} iterations.")
                return x_k

            # Newton's update step
            try:
                # Solves H * delta = -grad
                delta = np.linalg.solve(hess_k, -grad_k) 
            except np.linalg.LinAlgError:
                print("Hessian is singular (not invertible).")
                print("Method failed.")
                return x_k

            x_k = x_k + delta

        print("Reached maximum iterations without full convergence.")
        return x_k