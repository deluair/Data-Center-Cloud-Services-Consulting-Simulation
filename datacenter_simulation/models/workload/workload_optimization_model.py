"""
Workload Optimization Model for Data Center Operations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel

class WorkloadParameters(BaseModel):
    """Parameters for workload optimization simulation"""
    base_utilization: float  # Base resource utilization
    workload_variability: float  # Workload variation factor
    optimization_threshold: float  # Target optimization threshold
    scaling_factor: float  # Resource scaling factor
    efficiency_target: float  # Target efficiency level
    cost_performance_ratio: float  # Cost-performance optimization factor

class WorkloadOptimizationModel:
    """Simulates workload optimization for data centers"""
    
    def __init__(self, parameters: WorkloadParameters):
        self.parameters = parameters
        
    def simulate_workload_patterns(self,
                                 years: int,
                                 regions: List[str],
                                 initial_workload: float) -> pd.DataFrame:
        """
        Simulate workload patterns and variations
        
        Args:
            years: Number of years to simulate
            regions: List of regions to simulate
            initial_workload: Initial workload in units
            
        Returns:
            DataFrame with workload pattern data
        """
        dates = pd.date_range(start='2025-01-01', periods=years*365, freq='D')
        results = pd.DataFrame(index=dates)
        
        for region in regions:
            # Generate base workload
            base_workload = initial_workload * (1 + np.random.normal(0, 0.1, len(dates)))
            
            # Add daily patterns
            daily_pattern = 1 + 0.2 * np.sin(2 * np.pi * np.arange(len(dates)) / 24)
            
            # Add weekly patterns
            weekly_pattern = 1 + 0.3 * np.sin(2 * np.pi * np.arange(len(dates)) / (24 * 7))
            
            # Add seasonal patterns
            seasonal_pattern = 1 + 0.4 * np.sin(2 * np.pi * np.arange(len(dates)) / (24 * 365))
            
            # Combine patterns
            workload = base_workload * daily_pattern * weekly_pattern * seasonal_pattern
            
            # Add random variations
            variation = np.random.normal(1, self.parameters.workload_variability, len(dates))
            workload = workload * variation
            
            results[f'{region}_workload'] = workload
            results[f'{region}_utilization'] = workload / initial_workload
            results[f'{region}_variation'] = variation
            
        return results
    
    def calculate_optimization_requirements(self,
                                         patterns: pd.DataFrame,
                                         regions: List[str]) -> Dict[str, Dict]:
        """
        Calculate workload optimization requirements
        
        Args:
            patterns: DataFrame with workload pattern data
            regions: List of regions to analyze
            
        Returns:
            Dictionary with optimization requirements by region
        """
        requirements = {}
        
        for region in regions:
            # Calculate peak workload
            peak_workload = patterns[f'{region}_workload'].max()
            
            # Calculate required capacity
            required_capacity = peak_workload * self.parameters.scaling_factor
            
            # Calculate optimization targets
            optimization_target = patterns[f'{region}_utilization'].mean() * self.parameters.optimization_threshold
            
            # Calculate efficiency requirements
            efficiency_requirement = self.parameters.efficiency_target
            
            requirements[region] = {
                'peak_workload': peak_workload,
                'required_capacity': required_capacity,
                'optimization_target': optimization_target,
                'efficiency_requirement': efficiency_requirement,
                'scaling_factor': self.parameters.scaling_factor
            }
            
        return requirements
    
    def evaluate_optimization_metrics(self,
                                    patterns: pd.DataFrame,
                                    requirements: Dict[str, Dict],
                                    regions: List[str]) -> Dict[str, Dict]:
        """
        Evaluate workload optimization metrics
        
        Args:
            patterns: DataFrame with workload pattern data
            requirements: Dictionary with optimization requirements
            regions: List of regions to analyze
            
        Returns:
            Dictionary with optimization metrics by region
        """
        metrics = {}
        
        for region in regions:
            # Calculate resource utilization
            utilization = patterns[f'{region}_utilization'].mean()
            
            # Calculate optimization effectiveness
            optimization_effectiveness = 1 - (utilization / requirements[region]['optimization_target'])
            
            # Calculate efficiency
            efficiency = 1 - (patterns[f'{region}_variation'].std() * 
                            (1 - self.parameters.efficiency_target))
            
            # Calculate cost-performance ratio
            cost_performance = efficiency / utilization
            
            metrics[region] = {
                'utilization': utilization,
                'optimization_effectiveness': optimization_effectiveness,
                'efficiency': efficiency,
                'cost_performance': cost_performance,
                'optimization_score': (optimization_effectiveness * 0.4 + 
                                     efficiency * 0.3 + 
                                     cost_performance * 0.3)
            }
            
        return metrics
    
    def recommend_optimization_strategies(self,
                                       metrics: Dict[str, Dict],
                                       requirements: Dict[str, Dict],
                                       budget_constraint: float) -> Dict[str, List[str]]:
        """
        Recommend workload optimization strategies
        
        Args:
            metrics: Dictionary with optimization metrics
            requirements: Dictionary with optimization requirements
            budget_constraint: Maximum budget for optimization
            
        Returns:
            Dictionary of recommended strategies by region
        """
        strategies = {}
        
        for region, metric in metrics.items():
            region_strategies = []
            
            if metric['utilization'] < 0.7:
                region_strategies.append('Implement workload consolidation')
                region_strategies.append('Optimize resource allocation')
                
            if metric['optimization_effectiveness'] < 0.8:
                region_strategies.append('Implement auto-scaling')
                region_strategies.append('Optimize scheduling algorithms')
                
            if metric['efficiency'] < 0.9:
                region_strategies.append('Implement load balancing')
                region_strategies.append('Optimize workload distribution')
                
            if metric['cost_performance'] < 0.8:
                region_strategies.append('Implement cost-aware scheduling')
                region_strategies.append('Optimize resource pricing')
                
            if metric['optimization_score'] < 0.85:
                region_strategies.append('Implement comprehensive optimization program')
                region_strategies.append('Enhance monitoring and analytics')
                
            strategies[region] = region_strategies
            
        return strategies
    
    def calculate_optimization_costs(self,
                                   requirements: Dict[str, Dict],
                                   metrics: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Calculate workload optimization costs
        
        Args:
            requirements: Dictionary with optimization requirements
            metrics: Dictionary with optimization metrics
            
        Returns:
            Dictionary with cost breakdown by region
        """
        costs = {}
        
        for region in requirements.keys():
            # Calculate infrastructure costs
            infrastructure_cost = 100000 * (1 - metrics[region]['utilization'])
            
            # Calculate optimization costs
            optimization_cost = 75000 * (1 - metrics[region]['optimization_effectiveness'])
            
            # Calculate efficiency costs
            efficiency_cost = 50000 * (1 - metrics[region]['efficiency'])
            
            # Calculate performance costs
            performance_cost = 25000 * (1 - metrics[region]['cost_performance'])
            
            # Calculate maintenance costs
            maintenance_cost = (infrastructure_cost + optimization_cost + 
                              efficiency_cost + performance_cost) * 0.2
            
            costs[region] = {
                'infrastructure_cost': infrastructure_cost,
                'optimization_cost': optimization_cost,
                'efficiency_cost': efficiency_cost,
                'performance_cost': performance_cost,
                'maintenance_cost': maintenance_cost,
                'total_cost': (infrastructure_cost + optimization_cost + 
                             efficiency_cost + performance_cost + maintenance_cost)
            }
            
        return costs 