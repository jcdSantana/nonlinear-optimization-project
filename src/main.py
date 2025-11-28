import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
from algorithms.augmented_lagrangian_solver import AugmentedLagrangianSolver 
from Functions.circle_packing_problem import CirclePackingProblem
from utils.plot import plot_solution

def main():
    # 1. Configuration
    N_CIRCLES = 3  # Change this number to test different packings!
    
    print(f"=== Optimization of Circle Packing (N={N_CIRCLES}) ===")

    # 2. Instantiate Problem
    problem = CirclePackingProblem(N_CIRCLES)
    objective = problem.get_objective()
    constraints = problem.get_constraints()
    initial_guess = problem.get_initial_guess()

    # 3. Instantiate Solver
    solver = AugmentedLagrangianSolver(
        rho=1.0, 
        rho_multiplier=1.1, 
        max_iter=100, 
        tol=1e-5
    )

    # 4. Run Optimization
    result_z = solver.optimize(objective, constraints, initial_guess)

    # 5. Output Results
    final_L = result_z[0]
    print("\n=== Final Results ===")
    print(f"Minimum Container Size L: {final_L:.5f}")
    
    # 6. Visualize
    plot_solution(result_z, N_CIRCLES)

if __name__ == "__main__":
    main()