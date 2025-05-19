"""
AI Workload and Compute Density Simulation Model
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel

class ComputeDensityParameters(BaseModel):
    """Parameters for compute density simulation"""
    base_power_density: float  # Base power density in kW/rack
    ai_growth_rate: float  # Annual growth rate of AI workloads
    efficiency_improvement: float  # Annual efficiency improvement rate
    training_ratio: float  # Ratio of training to inference workloads
    edge_workload_ratio: float  # Ratio of edge to cloud workloads

class ComputeDensityModel:
    """Simulates AI workload evolution and compute density trends"""
    
    def __init__(self, parameters: ComputeDensityParameters):
        self.parameters = parameters
        
    def simulate_workload_evolution(self,
                                  years: int,
                                  initial_workload: float) -> pd.DataFrame:
        """
        Simulate the evolution of AI workloads
        
        Args:
            years: Number of years to simulate
            initial_workload: Initial workload in TFLOPS
            
        Returns:
            DataFrame with workload projections
        """
        dates = pd.date_range(start='2025-01-01', periods=years*365, freq='D')
        results = pd.DataFrame(index=dates)
        
        # Calculate daily growth factor
        daily_growth = (1 + self.parameters.ai_growth_rate) ** (1/365)
        
        # Simulate workload growth
        workload = initial_workload * (daily_growth ** np.arange(len(dates)))
        
        # Split into training and inference
        training_workload = workload * self.parameters.training_ratio
        inference_workload = workload * (1 - self.parameters.training_ratio)
        
        # Split into edge and cloud
        edge_workload = workload * self.parameters.edge_workload_ratio
        cloud_workload = workload * (1 - self.parameters.edge_workload_ratio)
        
        results['total_workload'] = workload
        results['training_workload'] = training_workload
        results['inference_workload'] = inference_workload
        results['edge_workload'] = edge_workload
        results['cloud_workload'] = cloud_workload
        
        return results
    
    def calculate_power_density(self,
                              workload: pd.DataFrame,
                              facility_type: str) -> pd.DataFrame:
        """
        Calculate power density requirements based on workload
        
        Args:
            workload: DataFrame with workload projections
            facility_type: Type of facility (hyperscale, enterprise, edge)
            
        Returns:
            DataFrame with power density projections
        """
        # Base efficiency factors by facility type
        efficiency_factors = {
            'hyperscale': 1.0,
            'enterprise': 0.8,
            'edge': 0.6
        }
        
        efficiency = efficiency_factors.get(facility_type, 1.0)
        
        # Calculate daily efficiency improvement
        daily_improvement = (1 + self.parameters.efficiency_improvement) ** (1/365)
        efficiency_trend = efficiency * (daily_improvement ** np.arange(len(workload)))
        
        # Calculate power density
        power_density = self.parameters.base_power_density * (workload['total_workload'] / workload['total_workload'].iloc[0])
        power_density = power_density / efficiency_trend
        
        return pd.DataFrame({
            'power_density': power_density,
            'efficiency_factor': efficiency_trend
        })
    
    def estimate_infrastructure_requirements(self,
                                          power_density: pd.DataFrame,
                                          facility_size: float) -> Dict[str, float]:
        """
        Estimate infrastructure requirements based on power density
        
        Args:
            power_density: DataFrame with power density projections
            facility_size: Facility size in square feet
            
        Returns:
            Dictionary with infrastructure requirements
        """
        max_density = power_density['power_density'].max()
        avg_density = power_density['power_density'].mean()
        
        # Estimate rack count
        racks_per_sqft = 1/30  # Assuming 30 sqft per rack
        total_racks = facility_size * racks_per_sqft
        
        # Calculate power requirements
        total_power = total_racks * avg_density
        peak_power = total_racks * max_density
        
        # Estimate cooling requirements (assuming 1.2x power for cooling)
        cooling_capacity = peak_power * 1.2
        
        return {
            'total_racks': total_racks,
            'average_power_density': avg_density,
            'peak_power_density': max_density,
            'total_power_requirement': total_power,
            'peak_power_requirement': peak_power,
            'cooling_capacity': cooling_capacity
        }
    
    def recommend_optimization_strategies(self,
                                       requirements: Dict[str, float],
                                       budget_constraint: float) -> List[str]:
        """
        Recommend optimization strategies based on infrastructure requirements
        
        Args:
            requirements: Dictionary with infrastructure requirements
            budget_constraint: Maximum budget for optimization
            
        Returns:
            List of recommended optimization strategies
        """
        strategies = []
        
        if requirements['peak_power_density'] > 30:  # kW/rack
            strategies.append('Implement liquid cooling system')
            strategies.append('Consider immersion cooling for high-density racks')
            
        if requirements['cooling_capacity'] > 1000:  # kW
            strategies.append('Implement advanced cooling optimization')
            strategies.append('Consider waste heat recovery system')
            
        if requirements['total_power_requirement'] > 5000:  # kW
            strategies.append('Implement power distribution optimization')
            strategies.append('Consider on-site power generation')
            
        return strategies 