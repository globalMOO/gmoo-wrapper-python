"""
Multi-Pipe Example of GMOO SDK Usage

This example demonstrates how to use multiple optimization pipes (parallel searches)
to find inputs that produce target outputs. Multiple pipes can help avoid local
minima and increase the chances of finding a good solution.

The example uses a multi-modal function with 3 inputs and 3 outputs, where
multiple different input combinations can produce similar outputs.
"""

import numpy as np
import os
from dotenv import load_dotenv
import time

# Add parent directory to path to access src
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'src'))

# Import GMOO SDK components
from gmoo_sdk.load_dll import load_dll
from gmoo_sdk.dll_interface import GMOOAPI
from gmoo_sdk.satisfaction import check_satisfaction


def multi_modal_model(inputs):
    """
    A mostly linear function with slight nonlinearity.
    This demonstrates multi-pipe optimization on a simpler problem
    where different starting points may still converge differently.
    
    3 inputs -> 3 outputs
    """
    x1, x2, x3 = inputs[0], inputs[1], inputs[2]
    
    # Output 1: Linear combination with small quadratic term
    o1 = 0.3*x1 + 0.2*x2 + 0.1*x3 + 0.05*(x1**2)
    
    # Output 2: Mostly linear with interaction term
    o2 = 0.2*x1 + 0.4*x2 + 0.3*x3 + 0.1*x1*x2
    
    # Output 3: Linear with slight nonlinearity from x3
    o3 = 0.1*x1 + 0.3*x2 + 0.2*x3 + 0.05*np.sqrt(x3)
    
    return np.array([o1, o2, o3])


def generate_random_starting_points(num_pipes, var_mins, var_maxs):
    """Generate random starting points for multiple pipes."""
    starting_points = []
    num_vars = len(var_mins)
    
    for pipe in range(num_pipes):
        # Generate random starting point within bounds
        random_inputs = []
        for i in range(num_vars):
            random_val = np.random.uniform(var_mins[i], var_maxs[i])
            random_inputs.append(random_val)
        starting_points.append(random_inputs)
    
    return starting_points


def run_multi_pipe_example():
    """Run the multi-pipe GMOO example."""
    
    # Load environment variables from .env file
    load_dotenv()
    
    print("=" * 60)
    print("GMOO SDK Multi-Pipe Example")
    print("=" * 60)
    
    # Step 1: Load the DLL
    print("\n1. Loading GMOO DLL...")
    try:
        dll_path = os.environ.get('MOOLIB')
        if not dll_path:
            raise ValueError("MOOLIB environment variable not set. Please check your .env file.")
        
        vsme_dll = load_dll(dll_path)
        print(f"   ✓ DLL loaded successfully from: {dll_path}")
    except Exception as e:
        print(f"   ✗ Failed to load DLL: {e}")
        return
    
    # Step 2: Define problem parameters
    print("\n2. Setting up problem parameters...")
    
    # Variable bounds (3 inputs)
    var_mins = [0.0, 0.0, 0.0]
    var_maxs = [10.0, 10.0, 2.0]
    num_outputs = 3
    
    # Number of parallel optimization pipes
    num_pipes = 4
    print(f"   - Using {num_pipes} parallel optimization pipes")
    
    # Create GMOO API instance
    gmoo_model = GMOOAPI(
        vsme_windll=vsme_dll,
        vsme_input_filename="multi_pipe_example",
        var_mins=var_mins,
        var_maxs=var_maxs,
        num_output_vars=num_outputs,
        model_function=multi_modal_model,
        save_file_dir="."
    )
    print("   ✓ GMOO model created")
    
    # Step 3: Train the inverse model (Development phase)
    print("\n3. Training inverse model...")
    
    try:
        # Initialize development
        gmoo_model.development.load_vsme_name()
        gmoo_model.development.initialize_variables()
        gmoo_model.development.load_variable_types()
        gmoo_model.development.load_variable_limits()
        
        # Design cases
        gmoo_model.development.design_agents()
        gmoo_model.development.design_cases()
        case_count = gmoo_model.development.get_case_count()
        print(f"   - Designed {case_count} training cases")
        
        # Generate and evaluate training cases
        input_cases = []
        output_cases = []
        
        for i in range(1, case_count + 1):
            case_vars = gmoo_model.development.poke_case_variables(i)
            input_cases.append(case_vars)
            output = multi_modal_model(case_vars)
            output_cases.append(output)
        
        # Load results
        gmoo_model.development.initialize_outcomes()
        for i in range(1, case_count + 1):
            gmoo_model.development.load_case_results(i, output_cases[i-1])
        
        # Develop and export model
        gmoo_model.development.develop_vsme()
        gmoo_file = gmoo_model.development.export_vsme()
        print(f"   ✓ Inverse model trained and saved to: {gmoo_file}")
        
        # Unload development
        gmoo_model.development.unload_vsme()
        
    except Exception as e:
        print(f"   ✗ Failed to train model: {e}")
        return
    
    # Step 4: Perform inverse optimization with multiple pipes
    print("\n4. Performing inverse optimization with multiple pipes...")
    
    # Define target outputs
    # Let's calculate reasonable outputs by evaluating at a mid-range point
    # For example, at x1=5, x2=5, x3=1:
    # o1 = 0.3*5 + 0.2*5 + 0.1*1 + 0.05*(5**2) = 1.5 + 1.0 + 0.1 + 1.25 = 3.85
    # o2 = 0.2*5 + 0.4*5 + 0.3*1 + 0.1*5*5 = 1.0 + 2.0 + 0.3 + 2.5 = 5.8
    # o3 = 0.1*5 + 0.3*5 + 0.2*1 + 0.05*sqrt(1) = 0.5 + 1.5 + 0.2 + 0.05 = 2.25
    target_outputs = [3.85, 5.8, 2.25]
    print(f"   - Target outputs: {target_outputs}")
    
    # Define objective types (1 = percentage error)
    objective_types = [1, 1, 1]
    
    # Define acceptable percentage error (5%)
    # Note: For objective type 1 (percentage error), only uncertainty_plus is used
    # uncertainty_minus is ignored in the satisfaction check
    uncertainty_plus = [5.0, 5.0, 5.0]
    uncertainty_minus = [-5.0, -5.0, -5.0]  # Not used for percentage error, but keeping consistent format
    
    try:
        # Load model for application
        gmoo_model.application.load_model()
        
        # Initialize variables for multiple pipes
        gmoo_model.application.init_variables(nPipes=num_pipes)
        
        # Set objectives (same for all pipes)
        gmoo_model.application.assign_objectives_target(target_outputs, objective_types)
        gmoo_model.application.load_objective_uncertainty(uncertainty_plus, uncertainty_minus)
        
        # Generate random starting points for each pipe
        starting_points = generate_random_starting_points(num_pipes, var_mins, var_maxs)
        
        # Initialize storage for each pipe
        pipe_results = []
        pipe_converged = [False] * num_pipes
        current_inputs = []
        current_outputs = []
        
        for pipe in range(num_pipes):
            initial_outputs = multi_modal_model(starting_points[pipe])
            pipe_results.append({
                'best_inputs': starting_points[pipe].copy(),
                'best_outputs': initial_outputs.copy(),
                'best_error': float('inf'),
                'iterations': 0,
                'converged': False
            })
            current_inputs.append(starting_points[pipe].copy())
            current_outputs.append(initial_outputs.copy())
            
            print(f"\n   Pipe {pipe + 1} starting point: {[f'{x:.3f}' for x in starting_points[pipe]]}")
            print(f"   Initial outputs: {[f'{x:.3f}' for x in initial_outputs]}")
        
        # Run optimization iterations
        max_iterations = 30
        print(f"\n   Running up to {max_iterations} iterations...")
        
        for iteration in range(1, max_iterations + 1):
            # Perform inverse iteration for all pipes at once
            # The DLL handles parallel optimization internally
            next_inputs_all, l1_norms, l2_norms = gmoo_model.application.perform_inverse_iteration(
                target_outputs=target_outputs,
                current_inputs=current_inputs,
                current_outputs=current_outputs,
                objective_types=objective_types,
                objective_uncertainty_minus=uncertainty_minus,
                objective_uncertainty_plus=uncertainty_plus
            )
            
            # Update inputs and evaluate outputs for all pipes
            any_pipe_active = False
            for pipe in range(num_pipes):
                if pipe_converged[pipe]:
                    continue
                
                any_pipe_active = True
                result = pipe_results[pipe]
                
                # Update current values
                current_inputs[pipe] = next_inputs_all[pipe].tolist()
                current_outputs[pipe] = multi_modal_model(current_inputs[pipe]).tolist()
                result['iterations'] = iteration
                
                # Calculate error
                error = l2_norms[pipe]
                
                # Update best if improved
                if error < result['best_error']:
                    result['best_error'] = error
                    result['best_inputs'] = current_inputs[pipe].copy()
                    result['best_outputs'] = current_outputs[pipe].copy()
                
                # Check satisfaction
                satisfied, _ = check_satisfaction(
                    current_outputs[pipe],
                    target_outputs,
                    objective_types,
                    uncertainty_minus,
                    uncertainty_plus
                )
                
                if satisfied:
                    pipe_converged[pipe] = True
                    result['converged'] = True
                    print(f"\n   🎯 Pipe {pipe + 1} converged at iteration {iteration}!")
                    print(f"      Final inputs: {[f'{x:.3f}' for x in result['best_inputs']]}")
                    print(f"      Final outputs: {[f'{x:.3f}' for x in result['best_outputs']]}")
                    print(f"      Error: {result['best_error']:.6f}")
            
            # Progress update every 5 iterations
            if iteration % 5 == 0 and any_pipe_active:
                print(f"\n   Iteration {iteration}:")
                for pipe in range(num_pipes):
                    if not pipe_converged[pipe]:
                        print(f"     Pipe {pipe + 1}: error = {pipe_results[pipe]['best_error']:.6f}")
            
            # Stop if all pipes converged
            if not any_pipe_active:
                print(f"\n   All pipes converged!")
                break
        
        # Final results summary
        print("\n" + "=" * 60)
        print("FINAL RESULTS SUMMARY")
        print("=" * 60)
        
        converged_count = sum(pipe_converged)
        print(f"\nConverged: {converged_count}/{num_pipes} pipes")
        
        # Find best overall result
        best_pipe = min(range(num_pipes), key=lambda p: pipe_results[p]['best_error'])
        best_result = pipe_results[best_pipe]
        
        print(f"\nBest solution found by Pipe {best_pipe + 1}:")
        print(f"  Inputs:  {[f'{x:.3f}' for x in best_result['best_inputs']]}")
        print(f"  Outputs: {[f'{x:.3f}' for x in best_result['best_outputs']]}")
        print(f"  Target:  {[f'{x:.3f}' for x in target_outputs]}")
        print(f"  Error:   {best_result['best_error']:.6f}")
        print(f"  Status:  {'✓ Converged' if best_result['converged'] else '✗ Not converged'}")
        
        # Show all pipe results
        print("\nAll pipe results:")
        for pipe in range(num_pipes):
            result = pipe_results[pipe]
            print(f"\nPipe {pipe + 1}:")
            print(f"  Starting point: {[f'{x:.3f}' for x in starting_points[pipe]]}")
            print(f"  Final point:    {[f'{x:.3f}' for x in result['best_inputs']]}")
            print(f"  Final error:    {result['best_error']:.6f}")
            print(f"  Iterations:     {result['iterations']}")
            print(f"  Converged:      {result['converged']}")
        
        # Unload the model
        gmoo_model.application.unload_vsme()
        
    except Exception as e:
        print(f"   ✗ Failed during optimization: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n✓ Multi-pipe example completed successfully!")
    
    # Clean up the generated .gmoo file
    if os.path.exists(gmoo_file):
        try:
            os.remove(gmoo_file)
        except:
            pass


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run the example
    run_multi_pipe_example()