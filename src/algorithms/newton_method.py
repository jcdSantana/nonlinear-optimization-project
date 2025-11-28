import math
import numpy as np

class Newton_Method:
    """
    Implements the Newton-Raphson method for finding the root of a function f(x)=0.
    """

    def optimize(self, f, df, initial_point, threshold=1e-6, max_iter=1000):
        x = float(initial_point)
        
        for i in range(1, max_iter + 1):
            f_x = f(x)
            df_x = df(x)

            # Check for near-zero derivative (critical issue for Newton's method)
            if np.isclose(df_x, 0):
                print(f"Warning: Derivative is near zero at iteration {i}. Stopping.")
                return x, i - 1

            x_old = x
            x = x_old - (f_x / df_x)  # Newton-Raphson Step
            
            # Convergence Check
            if math.fabs(x - x_old) < threshold:
                return x, i

        # If loop completes without meeting the threshold
        print(f"Warning: Did not converge after {max_iter} iterations.")
        return x, max_iter