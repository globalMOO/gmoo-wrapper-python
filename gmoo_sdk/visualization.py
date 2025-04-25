# -*- coding: utf-8 -*-
"""
GMOO Visualization Tools Module

This module contains the VisualizationTools class, which provides methods for
visualizing optimization results, model behavior, and data analysis for the GMOO SDK.

Classes:
    VisualizationTools: Visualization and plotting functionality for the GMOO SDK

Authors: Matt Freeman, Jason Hopkins
Version: 2.0.0 (Refactored)
"""

import logging
from typing import List, Optional, Tuple, Union

import numpy as np

# Configure logging
logger = logging.getLogger(__name__)


class VisualizationTools:
    """
    Visualization and plotting functionality for the GMOO SDK.
    
    This class contains methods for visualizing optimization results,
    including 2D and 3D plots, error analysis, and convergence tracking.
    """
    
    def __init__(self, api):
        """
        Initialize the VisualizationTools object.
        
        Args:
            api: The parent GMOOAPI instance
        """
        self.api = api
    
    def plot_behavior(self, highlight_point: np.ndarray, initial_points: List[np.ndarray],
                     improved_points: List[np.ndarray]) -> None:
        """
        Plot the behavior of the function and optimization process in 2D.
        
        Args:
            highlight_point: Point to highlight (typically the final solution).
            initial_points: Initial points used in the optimization.
            improved_points: Points that represent improvements during optimization.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("Matplotlib is not installed. Cannot create plot.")
            logger.warning("Install matplotlib using: pip install matplotlib")
            return
        
        # Define the grid limits based on variable ranges
        x_min, x_max = (self.api.dVarLimMin[0], self.api.dVarLimMax[0])
        y_min, y_max = (self.api.dVarLimMin[1], self.api.dVarLimMax[1])

        # Create meshgrid for 2D visualization
        x_values = np.linspace(x_min, x_max, 100)
        y_values = np.linspace(y_min, y_max, 100)
        xx, yy = np.meshgrid(x_values, y_values)

        # Evaluate model function at each grid point for the first objective
        zz = np.zeros_like(xx)
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                zz[i, j] = self.api.modelFunction(np.array([xx[i, j], yy[i, j]]))[0]

        # Create the figure and plot the contour
        plt.figure(figsize=(10, 8))
        contour = plt.contourf(xx, yy, zz, levels=20, cmap="RdYlBu_r")
        plt.colorbar(contour, label='Objective Value')

        # Convert point lists to numpy arrays for easier plotting
        initial_points = np.array(initial_points)
        improved_points = np.array(improved_points)
        
        # Plot the points from the optimization process
        plt.scatter(initial_points[:, 0], initial_points[:, 1], c='blue', label='Sample Points')
        plt.scatter(improved_points[:, 0], improved_points[:, 1], c='green', label='Improvement Points')
        plt.scatter(highlight_point[0], highlight_point[1], c='red', s=100, marker='*', label='Convergence Point')
        
        # Plot arrows showing the optimization path
        smaller_head_width = 0.005 * (x_max - x_min)
        smaller_head_length = 0.01 * (x_max - x_min)
        for sp, ip in zip(initial_points, improved_points):
            plt.arrow(
                sp[0], sp[1], 
                ip[0] - sp[0], ip[1] - sp[1], 
                head_width=smaller_head_width, 
                head_length=smaller_head_length, 
                fc='black', ec='black'
            )

        # Add labels and legend
        plt.xlabel('Variable 1')
        plt.ylabel('Variable 2')
        plt.title('Optimization Path with Objective Contours')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        # Show the plot
        plt.tight_layout()
        plt.show()
        
    def plot_convergence(self, iterations: List[int], errors: List[float], 
                        target_threshold: Optional[float] = None) -> None:
        """
        Plot the convergence history of the optimization process.
        
        Args:
            iterations: List of iteration numbers
            errors: List of error values at each iteration
            target_threshold: Optional convergence threshold to highlight
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("Matplotlib is not installed. Cannot create plot.")
            logger.warning("Install matplotlib using: pip install matplotlib")
            return
            
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, errors, 'b-', marker='o', markersize=4)
        
        if target_threshold is not None:
            plt.axhline(y=target_threshold, color='r', linestyle='--', 
                      label=f'Convergence Threshold ({target_threshold})')
            
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.title('Optimization Convergence History')
        plt.grid(True, alpha=0.3)
        
        if target_threshold is not None:
            plt.legend()
            
        # Use logarithmic scale for y-axis if the error values span multiple orders of magnitude
        if max(errors) / (min(errors) + 1e-10) > 100:
            plt.yscale('log')
            
        plt.tight_layout()
        plt.show()
        
    def plot_variable_convergence(self, iterations: List[int], 
                                variable_values: List[List[float]],
                                variable_names: Optional[List[str]] = None) -> None:
        """
        Plot the convergence of individual variables over iterations.
        
        Args:
            iterations: List of iteration numbers
            variable_values: List of lists containing variable values at each iteration
            variable_names: Optional list of variable names for the legend
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("Matplotlib is not installed. Cannot create plot.")
            logger.warning("Install matplotlib using: pip install matplotlib")
            return
            
        # Convert to numpy array for easier handling
        var_values = np.array(variable_values)
        num_vars = var_values.shape[1]
        
        # Generate variable names if not provided
        if variable_names is None:
            variable_names = [f'Variable {i+1}' for i in range(num_vars)]
            
        # Create the plot
        plt.figure(figsize=(10, 6))
        
        for i in range(num_vars):
            plt.plot(iterations, var_values[:, i], marker='o', markersize=4, 
                   label=variable_names[i])
            
        plt.xlabel('Iteration')
        plt.ylabel('Variable Value')
        plt.title('Variable Convergence History')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    def plot_outcomes_vs_targets(self, outcomes: List[float], targets: List[float], 
                                outcome_names: Optional[List[str]] = None) -> None:
        """
        Create a bar chart comparing current outcomes with target values.
        
        Args:
            outcomes: List of current outcome values
            targets: List of target outcome values
            outcome_names: Optional list of outcome names
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("Matplotlib is not installed. Cannot create plot.")
            logger.warning("Install matplotlib using: pip install matplotlib")
            return
            
        num_outcomes = len(outcomes)
        
        # Generate outcome names if not provided
        if outcome_names is None:
            outcome_names = [f'Outcome {i+1}' for i in range(num_outcomes)]
            
        # Set up the bar positions
        index = np.arange(num_outcomes)
        bar_width = 0.35
        
        # Create the plot
        plt.figure(figsize=(12, 6))
        
        plt.bar(index, outcomes, bar_width, label='Current', color='blue', alpha=0.7)
        plt.bar(index + bar_width, targets, bar_width, label='Target', color='green', alpha=0.7)
        
        # Add labels and formatting
        plt.xlabel('Outcomes')
        plt.ylabel('Value')
        plt.title('Comparison of Current Outcomes vs. Targets')
        plt.xticks(index + bar_width/2, outcome_names)
        plt.legend()
        
        # Add value labels on the bars
        for i, v in enumerate(outcomes):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', color='blue', fontweight='bold')
            
        for i, v in enumerate(targets):
            plt.text(i + bar_width, v + 0.01, f'{v:.3f}', ha='center', va='bottom', color='green', fontweight='bold')
            
        plt.tight_layout()
        plt.show()
        
    def plot_error_distribution(self, outcomes: List[float], targets: List[float],
                               objective_types: Optional[List[int]] = None) -> None:
        """
        Create a visualization showing the error distribution across objectives.
        
        Args:
            outcomes: List of current outcome values
            targets: List of target outcome values
            objective_types: Optional list of objective types for context
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("Matplotlib is not installed. Cannot create plot.")
            logger.warning("Install matplotlib using: pip install matplotlib")
            return
            
        num_outcomes = len(outcomes)
        outcomes_array = np.array(outcomes)
        targets_array = np.array(targets)
        
        # Calculate errors
        absolute_errors = np.abs(outcomes_array - targets_array)
        
        if targets_array.any():  # Avoid division by zero
            percentage_errors = np.abs((outcomes_array - targets_array) / np.where(targets_array == 0, 1e-10, targets_array)) * 100
        else:
            percentage_errors = np.zeros_like(absolute_errors)
            
        # Set up the figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Absolute errors
        y_pos = np.arange(num_outcomes)
        ax1.barh(y_pos, absolute_errors, align='center')
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([f'Obj {i+1}' for i in range(num_outcomes)])
        ax1.set_xlabel('Absolute Error')
        ax1.set_title('Absolute Error by Objective')
        
        # Percentage errors
        ax2.barh(y_pos, percentage_errors, align='center')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([f'Obj {i+1}' for i in range(num_outcomes)])
        ax2.set_xlabel('Percentage Error (%)')
        ax2.set_title('Percentage Error by Objective')
        
        # Add type labels if provided
        if objective_types is not None:
            type_names = {
                0: "Exact Match",
                1: "Percentage Error",
                2: "Absolute Error",
                11: "Less Than",
                12: "Less Than/Equal",
                13: "Greater Than",
                14: "Greater Than/Equal",
                21: "Minimize",
                22: "Maximize"
            }
            
            for i, obj_type in enumerate(objective_types):
                type_text = type_names.get(obj_type, f"Type {obj_type}")
                ax1.text(absolute_errors[i] * 1.05, i, type_text, va='center')
                
        plt.tight_layout()
        plt.show()
        
    def plot_3d_response(self, var1_range: np.ndarray, var2_range: np.ndarray, 
                      objective_index: int = 0, fixed_vars: Optional[List[float]] = None) -> None:
        """
        Create a 3D surface plot showing the response of an objective to two variables.
        
        Args:
            var1_range: Array of values for the first variable
            var2_range: Array of values for the second variable
            objective_index: Index of the objective to plot (default 0)
            fixed_vars: Values of other variables to use for evaluation (if None, zeros are used)
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            logger.warning("Matplotlib is not installed. Cannot create plot.")
            logger.warning("Install matplotlib using: pip install matplotlib")
            return
            
        # Create meshgrid for 3D visualization
        xx, yy = np.meshgrid(var1_range, var2_range)
        
        # Prepare fixed variable values if needed
        if fixed_vars is None:
            fixed_vars = [0.0] * (self.api.nVars.value - 2)
            
        # Evaluate model function at each grid point
        zz = np.zeros_like(xx)
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                # Combine the two varied variables with fixed variables
                input_vars = np.array([xx[i, j], yy[i, j]] + fixed_vars)
                zz[i, j] = self.api.modelFunction(input_vars)[objective_index]
                
        # Create the 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        surf = ax.plot_surface(xx, yy, zz, cmap='viridis', edgecolor='none', alpha=0.8)
        
        # Add labels and colorbar
        ax.set_xlabel('Variable 1')
        ax.set_ylabel('Variable 2')
        ax.set_zlabel(f'Objective {objective_index+1}')
        ax.set_title(f'Response Surface for Objective {objective_index+1}')
        
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        plt.tight_layout()
        plt.show()
        
    def print_an_array(self, array_in: np.ndarray, label: Optional[str] = None) -> None:
        """
        Print the elements of an array along with their indices.
        
        Args:
            array_in: The array to print
            label: Optional label to print before the array
        """
        if label:
            print(f"{label}:")
            
        for ii, jj in enumerate(array_in):
            print(f"  {ii}: {jj}")