import sys
import os

# Setup paths to include 'src' directory
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')

if current_dir not in sys.path:
    sys.path.append(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

# Imports from src modules
from algorithms.augmented_lagrangian_solver import AugmentedLagrangianSolver
from algorithms.sqp_solver import SQPSolver
from Functions.circle_packing_problem import CirclePackingProblem
from utils.plot import plot_solution

def run_augmented_lagrangian(objective, constraints, x0, n_circles):
    print(f"\n--- [1/2] Running Augmented Lagrangian (N={n_circles}) ---")
    
    solver = AugmentedLagrangianSolver(
        rho=1.0, 
        rho_multiplier=1.1, 
        max_iter=100, 
        tol=1e-5
    )
    
    result = solver.optimize(objective, constraints, x0)
    print(f"Final Result (L): {result[0]:.5f}")
    
    # Plot and save using the updated plot_solution logic
    # Note: plot.py handles the 'data' folder creation and path joining internally
    fname = f"result_aug_lagrangian_N{n_circles}.png"
    plot_solution(result, n_circles, filename=fname)

def run_sqp(objective, constraints, x0, n_circles):
    print(f"\n--- [2/2] Running SQP (N={n_circles}) ---")
    
    solver = SQPSolver(
        max_iter=50, 
        tol=1e-5
    )
    
    result = solver.optimize(objective, constraints, x0)
    print(f"Final Result (L): {result[0]:.5f}")
    
    # Plot and save
    fname = f"result_sqp_N{n_circles}.png"
    plot_solution(result, n_circles, filename=fname)

def main():
    # 1. Configuration
    N_CIRCLES = [1,2,3, 5, 5, 7]
    for circle in N_CIRCLES:
        print(f"=== Circle Packing Optimization (N={circle}) ===")

        # 2. Instantiate Problem
        problem = CirclePackingProblem(circle)
        objective = problem.get_objective()
        constraints = problem.get_constraints()
        initial_guess = problem.get_initial_guess()

        # 3. Execute Solvers
        # We no longer need to pass 'save_path' as plot.py handles it
        run_augmented_lagrangian(objective, constraints, initial_guess, circle)
        run_sqp(objective, constraints, initial_guess, circle)

        print("\n=== Execution Finished. Check 'data' folder. ===")

if __name__ == "__main__":
    main()