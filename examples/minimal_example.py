"""
Minimal Example of GMOO SDK Usage

This is the simplest possible example showing how to:
1. Define a model function
2. Train an inverse model
3. Find inputs that produce desired outputs

Model: A simple linear function with 2 inputs and 2 outputs
Goal: Find inputs that produce specific target outputs
"""

import numpy as np
import os
from dotenv import load_dotenv

# Add parent directory to path to access src
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'src'))

from gmoo_sdk.load_dll import load_dll
from gmoo_sdk.dll_interface import GMOOAPI


def linear_model(inputs):
    """
    Simple linear model: 
    y1 = 2*x1 + x2
    y2 = x1 + 3*x2
    
    For example:
    - inputs [1, 2] → outputs [4, 7]
    - inputs [2, 1] → outputs [5, 5]
    """
    x1, x2 = inputs
    y1 = 2*x1 + x2
    y2 = x1 + 3*x2
    return np.array([y1, y2])


# Load environment variables
load_dotenv()

# Load the DLL
print("Loading GMOO DLL...")
vsme_dll = load_dll()

# Create GMOO model
print("Creating model...")
model = GMOOAPI(
    vsme_windll=vsme_dll,
    vsme_input_filename="minimal_example",
    var_mins=[0.0, 0.0],
    var_maxs=[10.0, 10.0],
    num_output_vars=2,
    model_function=linear_model,
    save_file_dir="."
)

# Train inverse model
print("Training inverse model...")
model.development.load_vsme_name()
model.development.initialize_variables()
model.development.load_variable_types()
model.development.load_variable_limits()
model.development.design_agents()
model.development.design_cases()

# Get and evaluate training cases
case_count = model.development.get_case_count()
model.development.initialize_outcomes()

for i in range(1, case_count + 1):
    inputs = model.development.poke_case_variables(i)
    outputs = linear_model(inputs)
    model.development.load_case_results(i, outputs)

# Build and save model
model.development.develop_vsme()
gmoo_file = model.development.export_vsme()
model.development.unload_vsme()

print(f"Model saved to: {gmoo_file}")

# Now use the model to find inputs for target outputs
print("\nFinding inputs for target outputs [8.0, 11.0]...")

# Load model for optimization
model.application.load_model()

# Set target: we want outputs [8.0, 11.0]
target = [8.0, 11.0]
model.application.assign_objectives_target(target, [0, 0])  # 0 = exact match

# Start optimization from center of domain
inputs = np.array([5.0, 5.0])
outputs = linear_model(inputs)

print(f"Starting: inputs={inputs}, outputs={outputs}")

# Run optimization
for i in range(20):
    # Get next suggestion
    inputs, l1, l2 = model.application.perform_inverse_iteration(
        target_outputs=target,
        current_inputs=inputs,
        current_outputs=outputs,
        objective_types=[0, 0]
    )
    
    # Evaluate new inputs
    outputs = linear_model(inputs)
    
    # Check if we're close enough
    error = np.abs(outputs - target).max()
    if error < 0.01:
        print(f"\n✓ Found solution in {i+1} iterations!")
        break
    
    if i % 5 == 0:
        print(f"Iteration {i+1}: inputs={inputs}, outputs={outputs}, error={error:.4f}")

# Show final result
print(f"\nFinal result:")
print(f"Target outputs: {target}")
print(f"Achieved outputs: {outputs}")
print(f"Found inputs: {inputs}")

# Verify the solution
print(f"\nVerification: {inputs} → {linear_model(inputs)}")

# Clean up
model.application.unload_vsme()
if os.path.exists(gmoo_file):
    os.remove(gmoo_file)