"""
Simple Example of GMOO SDK Usage

This example demonstrates how to:
1. Define a simple model function
2. Train an inverse model using development cases
3. Perform inverse optimization to find inputs that produce target outputs

The example uses a simple quadratic function with 2 inputs and 2 outputs.
"""

import numpy as np
import os
from dotenv import load_dotenv

# Add parent directory to path to access src
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'src'))

# Import GMOO SDK components
from gmoo_sdk.load_dll import load_dll
from gmoo_sdk.dll_interface import GMOOAPI
from gmoo_sdk.satisfaction import check_satisfaction


def simple_quadratic_model(inputs):
    """
    A simple quadratic model for demonstration.
    
    Inputs: [x1, x2]
    Outputs: [y1, y2] where:
        y1 = x1^2 + x2^2
        y2 = (x1 - 3)^2 + (x2 - 3)^2
    
    This creates a model where:
    - y1 is minimized at (0, 0)
    - y2 is minimized at (3, 3)
    """
    x1, x2 = inputs
    
    y1 = x1**2 + x2**2
    y2 = (x1 - 3)**2 + (x2 - 3)**2
    
    return np.array([y1, y2])


def main():
    """Run the simple GMOO example."""
    
    # Load environment variables from .env file
    load_dotenv()
    
    print("=" * 60)
    print("GMOO SDK Simple Example")
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
    
    # Variable bounds (2 inputs, both ranging from 0 to 5)
    var_mins = [0.0, 0.0]
    var_maxs = [5.0, 5.0]
    num_outputs = 2
    
    # Create GMOO API instance
    gmoo_model = GMOOAPI(
        vsme_windll=vsme_dll,
        vsme_input_filename="simple_example",
        var_mins=var_mins,
        var_maxs=var_maxs,
        num_output_vars=num_outputs,
        model_function=simple_quadratic_model,
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
        
        # Design cases for training
        gmoo_model.development.design_agents()
        gmoo_model.development.design_cases()
        case_count = gmoo_model.development.get_case_count()
        print(f"   - Generated {case_count} training cases")
        
        # Evaluate all cases
        gmoo_model.development.initialize_outcomes()
        
        for i in range(1, case_count + 1):
            case_inputs = gmoo_model.development.poke_case_variables(i)
            case_outputs = simple_quadratic_model(case_inputs)
            gmoo_model.development.load_case_results(i, case_outputs)
        
        # Train the inverse model
        gmoo_model.development.develop_vsme()
        gmoo_file = gmoo_model.development.export_vsme()
        print(f"   ✓ Inverse model trained and saved to: {gmoo_file}")
        
        # Clean up development
        gmoo_model.development.unload_vsme()
        
    except Exception as e:
        print(f"   ✗ Failed during development: {e}")
        return
    
    # Step 4: Perform inverse optimization (Application phase)
    print("\n4. Performing inverse optimization...")
    
    # Define target outputs we want to achieve
    # Let's aim for y1=5.0 and y2=5.0
    target_outputs = [5.0, 5.0]
    print(f"   - Target outputs: {target_outputs}")
    
    # Define objective types (1 = percentage error)
    objective_types = [1, 1]
    
    # Define acceptable percentage error (3%)
    uncertainty_plus = [3.0, 3.0]
    uncertainty_minus = [-3.0, -3.0]
    
    try:
        # Load the trained model
        gmoo_model.application.load_model()
        
        # Set objectives
        gmoo_model.application.assign_objectives_target(target_outputs, objective_types)
        gmoo_model.application.load_objective_uncertainty(uncertainty_plus, uncertainty_minus)
        
        # Initial guess (center of the domain)
        current_inputs = np.array([2.5, 2.5])
        current_outputs = simple_quadratic_model(current_inputs)
        
        print(f"   - Starting from: inputs={current_inputs}, outputs={current_outputs}")
        
        # Optimization loop
        max_iterations = 50
        converged = False
        
        for iteration in range(1, max_iterations + 1):
            # Get next suggestion from GMOO
            next_inputs, l1_norm, l2_norm = gmoo_model.application.perform_inverse_iteration(
                target_outputs=target_outputs,
                current_inputs=current_inputs,
                current_outputs=current_outputs,
                objective_types=objective_types
            )
            
            # Evaluate the suggested inputs
            current_inputs = next_inputs
            current_outputs = simple_quadratic_model(current_inputs)
            
            # Check satisfaction
            satisfied, error_values = check_satisfaction(
                current_outputs,
                target_outputs,
                objective_types,
                uncertainty_minus,
                uncertainty_plus
            )
            
            if iteration % 10 == 0 or satisfied:
                print(f"   - Iteration {iteration}: inputs={current_inputs}, "
                      f"outputs={current_outputs}, L1 error={l1_norm:.4f}")
            
            if satisfied:
                converged = True
                print(f"   ✓ Converged in {iteration} iterations!")
                break
        
        # Clean up
        gmoo_model.application.unload_vsme()
        
    except Exception as e:
        print(f"   ✗ Failed during optimization: {e}")
        return
    
    # Step 5: Display results
    print("\n5. Results:")
    print("=" * 60)
    
    if converged:
        print(f"Target outputs: {target_outputs}")
        print(f"Achieved outputs: {current_outputs}")
        print(f"Final inputs: {current_inputs}")
        
        # Calculate percentage errors for display
        percentage_errors = []
        for i in range(len(target_outputs)):
            if target_outputs[i] != 0:
                error = abs((current_outputs[i] - target_outputs[i]) / target_outputs[i]) * 100
                percentage_errors.append(f"{error:.2f}%")
            else:
                percentage_errors.append("N/A")
        
        print(f"Percentage errors: {percentage_errors}")
        print("\n✓ Successfully found inputs that produce the target outputs!")
    else:
        print("✗ Failed to converge within the maximum iterations")
        print(f"Best outputs achieved: {current_outputs}")
        print(f"Best inputs found: {current_inputs}")
    
    # Clean up the generated .gmoo file
    if os.path.exists(gmoo_file):
        os.remove(gmoo_file)
        print(f"\n✓ Cleaned up temporary file: {gmoo_file}")


if __name__ == "__main__":
    main()