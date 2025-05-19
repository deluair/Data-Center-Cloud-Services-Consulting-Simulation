"""
Cooling Technology Model for Data Center Thermal Management
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel

class CoolingParameters(BaseModel):
    """Parameters for cooling technology simulation"""
    base_pue: float  # Base Power Usage Effectiveness
    ambient_temp: float  # Average ambient temperature in Celsius
    humidity_level: float  # Average humidity level
    water_usage_efficiency: float  # Water usage effectiveness
    cooling_capacity: float  # Cooling capacity in kW
    redundancy_level: float  # Cooling system redundancy factor

class CoolingTechnologyModel:
    """Simulates cooling technology and thermal management for data centers"""
    
    def __init__(self, parameters: CoolingParameters):
        self.parameters = parameters
        
    def simulate_cooling_performance(self,
                                   years: int,
                                   regions: List[str],
                                   power_density: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate cooling system performance
        
        Args:
            years: Number of years to simulate
            regions: List of regions to simulate
            power_density: DataFrame with power density data
            
        Returns:
            DataFrame with cooling performance data
        """
        dates = pd.date_range(start='2025-01-01', periods=years*365, freq='D')
        results = pd.DataFrame(index=dates)
        
        for region in regions:
            # Calculate cooling load
            cooling_load = power_density[region] * (self.parameters.base_pue - 1)
            
            # Add seasonal variations
            seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)
            cooling_load = cooling_load * seasonal_factor
            
            # Calculate water usage
            water_usage = cooling_load * self.parameters.water_usage_efficiency
            
            # Calculate cooling efficiency
            efficiency = 1 / (1 + (cooling_load / self.parameters.cooling_capacity) ** 2)
            
            results[f'{region}_cooling_load'] = cooling_load
            results[f'{region}_water_usage'] = water_usage
            results[f'{region}_efficiency'] = efficiency
            
        return results
    
    def calculate_cooling_requirements(self,
                                    performance: pd.DataFrame,
                                    regions: List[str]) -> Dict[str, Dict]:
        """
        Calculate cooling infrastructure requirements
        
        Args:
            performance: DataFrame with cooling performance data
            regions: List of regions to analyze
            
        Returns:
            Dictionary with cooling requirements by region
        """
        requirements = {}
        
        for region in regions:
            # Calculate peak cooling load
            peak_load = performance[f'{region}_cooling_load'].max()
            
            # Calculate required cooling capacity with redundancy
            required_capacity = peak_load * self.parameters.redundancy_level
            
            # Calculate number of cooling units
            units_per_rack = 1  # Assuming 1 cooling unit per rack
            total_units = int(np.ceil(required_capacity / self.parameters.cooling_capacity))
            
            # Calculate water requirements
            water_requirement = performance[f'{region}_water_usage'].mean()
            
            requirements[region] = {
                'peak_cooling_load': peak_load,
                'required_capacity': required_capacity,
                'total_units': total_units,
                'water_requirement': water_requirement,
                'redundancy_level': self.parameters.redundancy_level
            }
            
        return requirements
    
    def evaluate_cooling_efficiency(self,
                                  performance: pd.DataFrame,
                                  requirements: Dict[str, Dict],
                                  regions: List[str]) -> Dict[str, Dict]:
        """
        Evaluate cooling system efficiency
        
        Args:
            performance: DataFrame with cooling performance data
            requirements: Dictionary with cooling requirements
            regions: List of regions to analyze
            
        Returns:
            Dictionary with efficiency metrics by region
        """
        efficiency = {}
        
        for region in regions:
            # Calculate average efficiency
            avg_efficiency = performance[f'{region}_efficiency'].mean()
            
            # Calculate capacity utilization
            avg_load = performance[f'{region}_cooling_load'].mean()
            utilization = avg_load / requirements[region]['required_capacity']
            
            # Calculate water efficiency
            water_efficiency = 1 / (performance[f'{region}_water_usage'].mean() / 
                                  performance[f'{region}_cooling_load'].mean())
            
            # Calculate energy efficiency
            energy_efficiency = 1 / (performance[f'{region}_cooling_load'].mean() / 
                                   requirements[region]['required_capacity'])
            
            efficiency[region] = {
                'average_efficiency': avg_efficiency,
                'capacity_utilization': utilization,
                'water_efficiency': water_efficiency,
                'energy_efficiency': energy_efficiency,
                'pue_impact': self.parameters.base_pue * (1 - avg_efficiency)
            }
            
        return efficiency
    
    def recommend_cooling_optimizations(self,
                                     efficiency: Dict[str, Dict],
                                     requirements: Dict[str, Dict],
                                     budget_constraint: float) -> Dict[str, List[str]]:
        """
        Recommend cooling optimization strategies
        
        Args:
            efficiency: Dictionary with efficiency metrics
            requirements: Dictionary with cooling requirements
            budget_constraint: Maximum budget for optimization
            
        Returns:
            Dictionary of recommended strategies by region
        """
        strategies = {}
        
        for region, metrics in efficiency.items():
            region_strategies = []
            
            if metrics['average_efficiency'] < 0.8:
                region_strategies.append('Upgrade cooling systems')
                region_strategies.append('Implement hot aisle containment')
                
            if metrics['water_efficiency'] < 0.7:
                region_strategies.append('Implement water recycling')
                region_strategies.append('Optimize water usage')
                
            if metrics['energy_efficiency'] < 0.8:
                region_strategies.append('Implement free cooling')
                region_strategies.append('Optimize airflow management')
                
            if metrics['pue_impact'] > 0.2:
                region_strategies.append('Implement liquid cooling')
                region_strategies.append('Optimize temperature setpoints')
                
            if metrics['capacity_utilization'] > 0.9:
                region_strategies.append('Add cooling capacity')
                region_strategies.append('Implement load balancing')
                
            strategies[region] = region_strategies
            
        return strategies
    
    def calculate_cooling_costs(self,
                              requirements: Dict[str, Dict],
                              efficiency: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Calculate cooling infrastructure costs
        
        Args:
            requirements: Dictionary with cooling requirements
            efficiency: Dictionary with efficiency metrics
            
        Returns:
            Dictionary with cost breakdown by region
        """
        costs = {}
        
        for region in requirements.keys():
            # Calculate hardware costs
            unit_cost = 50000  # Cost per cooling unit
            hardware_cost = requirements[region]['total_units'] * unit_cost
            
            # Calculate water costs
            water_cost = requirements[region]['water_requirement'] * 0.1  # $0.1 per liter
            
            # Calculate energy costs
            energy_cost = requirements[region]['required_capacity'] * 0.1  # $0.1 per kW
            
            # Calculate maintenance costs
            maintenance_cost = hardware_cost * 0.15  # 15% of hardware cost per year
            
            costs[region] = {
                'hardware_cost': hardware_cost,
                'water_cost': water_cost,
                'energy_cost': energy_cost,
                'maintenance_cost': maintenance_cost,
                'total_cost': hardware_cost + water_cost + energy_cost + maintenance_cost
            }
            
        return costs 