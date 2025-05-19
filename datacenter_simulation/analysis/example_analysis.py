"""
Example Analysis Script for Data Center Optimization and Security
"""

import pandas as pd
import numpy as np
from models.workload.workload_optimization_model import WorkloadOptimizationModel, WorkloadParameters
from models.security.security_model import SecurityModel, SecurityParameters
from analysis.workload_visualization import WorkloadVisualizer
from analysis.security_visualization import SecurityVisualizer

def main():
    # Define regions and simulation parameters
    regions = ['North America', 'Europe', 'Asia Pacific']
    years = 5
    initial_workload = 1000  # Initial workload in units

    # Initialize workload optimization parameters
    workload_params = WorkloadParameters(
        base_utilization=0.7,
        workload_variability=0.2,
        optimization_threshold=0.8,
        scaling_factor=1.2,
        efficiency_target=0.9,
        cost_performance_ratio=0.85
    )

    # Initialize security parameters
    security_params = SecurityParameters(
        base_threat_level=0.3,
        attack_frequency=0.1,
        detection_rate=0.9,
        response_time=4.0,
        vulnerability_rate=0.05,
        security_budget_ratio=0.1
    )

    # Create model instances
    workload_model = WorkloadOptimizationModel(workload_params)
    security_model = SecurityModel(security_params)

    # Run workload optimization simulation
    print("Running workload optimization simulation...")
    workload_patterns = workload_model.simulate_workload_patterns(years, regions, initial_workload)
    workload_requirements = workload_model.calculate_optimization_requirements(workload_patterns, regions)
    workload_metrics = workload_model.evaluate_optimization_metrics(workload_patterns, workload_requirements, regions)
    workload_strategies = workload_model.recommend_optimization_strategies(workload_metrics, workload_requirements, budget_constraint=1000000)
    workload_costs = workload_model.calculate_optimization_costs(workload_requirements, workload_metrics)

    # Run security simulation
    print("Running security simulation...")
    security_threats = security_model.simulate_security_threats(years, regions)
    security_requirements = security_model.calculate_security_requirements(security_threats, regions)
    security_metrics = security_model.evaluate_security_metrics(security_threats, security_requirements, regions)
    security_strategies = security_model.recommend_security_strategies(security_metrics, security_requirements, budget_constraint=1000000)
    security_costs = security_model.calculate_security_costs(security_requirements, security_metrics)

    # Initialize visualizers
    workload_visualizer = WorkloadVisualizer()
    security_visualizer = SecurityVisualizer()

    # Create and save visualizations
    print("Creating visualizations...")
    
    # Save workload analysis
    workload_visualizer.save_workload_analysis(
        workload_patterns,
        workload_metrics,
        workload_requirements,
        workload_costs,
        workload_strategies,
        regions,
        'workload_analysis.html'
    )

    # Save security analysis
    security_visualizer.save_security_analysis(
        security_threats,
        security_metrics,
        security_costs,
        security_strategies,
        regions,
        'security_analysis.html'
    )

    # Print summary statistics
    print("\nWorkload Optimization Summary:")
    for region in regions:
        print(f"\n{region}:")
        print(f"Average Utilization: {workload_metrics[region]['utilization']:.2%}")
        print(f"Optimization Score: {workload_metrics[region]['optimization_score']:.2%}")
        print(f"Total Optimization Cost: ${workload_costs[region]['total_cost']:,.2f}")
        print("\nRecommended Strategies:")
        for strategy in workload_strategies[region]:
            print(f"- {strategy}")

    print("\nSecurity Analysis Summary:")
    for region in regions:
        print(f"\n{region}:")
        print(f"Security Score: {security_metrics[region]['security_score']:.2%}")
        print(f"Risk Level: {security_metrics[region]['risk_level']:.2%}")
        print(f"Total Security Cost: ${security_costs[region]['total_cost']:,.2f}")
        print("\nRecommended Strategies:")
        for strategy in security_strategies[region]:
            print(f"- {strategy}")

if __name__ == "__main__":
    main() 