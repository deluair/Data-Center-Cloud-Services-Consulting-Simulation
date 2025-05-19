"""
Carbon Emissions and Sustainability Model for Data Center Operations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from pydantic import BaseModel

class CarbonEmissionsParameters(BaseModel):
    """Parameters for carbon emissions simulation"""
    grid_carbon_intensity: float  # Base grid carbon intensity in kg CO2/kWh
    renewable_energy_ratio: float  # Ratio of renewable energy in the mix
    carbon_tax_rate: float  # Carbon tax rate in $/ton CO2
    efficiency_improvement_rate: float  # Annual efficiency improvement rate
    scope_2_emissions_factor: float  # Factor for scope 2 emissions
    scope_3_emissions_factor: float  # Factor for scope 3 emissions

class CarbonEmissionsModel:
    """Simulates carbon emissions and sustainability metrics for data centers"""
    
    def __init__(self, parameters: CarbonEmissionsParameters):
        self.parameters = parameters
        
    def simulate_emissions(self,
                          years: int,
                          power_consumption: pd.DataFrame,
                          regions: List[str]) -> pd.DataFrame:
        """
        Simulate carbon emissions over time
        
        Args:
            years: Number of years to simulate
            power_consumption: DataFrame with power consumption data
            regions: List of regions to simulate
            
        Returns:
            DataFrame with emissions data
        """
        dates = pd.date_range(start='2025-01-01', periods=years*365, freq='D')
        results = pd.DataFrame(index=dates)
        
        for region in regions:
            # Calculate daily efficiency improvement
            daily_improvement = (1 + self.parameters.efficiency_improvement_rate) ** (1/365)
            efficiency_trend = daily_improvement ** np.arange(len(dates))
            
            # Calculate grid carbon intensity with regional variation
            base_intensity = self.parameters.grid_carbon_intensity * (1 + np.random.normal(0, 0.1))
            
            # Add renewable energy impact
            renewable_impact = self.parameters.renewable_energy_ratio * (1 - base_intensity/1000)
            
            # Calculate scope 1 emissions (direct)
            scope_1 = power_consumption[region] * base_intensity * (1 - renewable_impact)
            
            # Calculate scope 2 emissions (indirect)
            scope_2 = scope_1 * self.parameters.scope_2_emissions_factor
            
            # Calculate scope 3 emissions (value chain)
            scope_3 = (scope_1 + scope_2) * self.parameters.scope_3_emissions_factor
            
            # Calculate total emissions
            total_emissions = scope_1 + scope_2 + scope_3
            
            # Apply efficiency improvements
            total_emissions = total_emissions / efficiency_trend
            
            # Store results
            results[f'{region}_scope1'] = scope_1
            results[f'{region}_scope2'] = scope_2
            results[f'{region}_scope3'] = scope_3
            results[f'{region}_total'] = total_emissions
            
        return results
    
    def calculate_carbon_cost(self,
                            emissions: pd.DataFrame,
                            regions: List[str]) -> Dict[str, float]:
        """
        Calculate carbon tax costs based on emissions
        
        Args:
            emissions: DataFrame with emissions data
            regions: List of regions to analyze
            
        Returns:
            Dictionary with carbon costs by region
        """
        costs = {}
        
        for region in regions:
            total_emissions = emissions[f'{region}_total'].sum()
            carbon_cost = total_emissions * self.parameters.carbon_tax_rate / 1000  # Convert to tons
            costs[region] = carbon_cost
            
        return costs
    
    def recommend_sustainability_strategies(self,
                                         emissions: pd.DataFrame,
                                         regions: List[str],
                                         budget_constraint: float) -> Dict[str, List[str]]:
        """
        Recommend sustainability strategies based on emissions data
        
        Args:
            emissions: DataFrame with emissions data
            regions: List of regions to analyze
            budget_constraint: Maximum budget for sustainability initiatives
            
        Returns:
            Dictionary of recommended strategies by region
        """
        strategies = {}
        
        for region in regions:
            region_strategies = []
            total_emissions = emissions[f'{region}_total'].mean()
            
            if total_emissions > 1000:  # High emissions
                region_strategies.append('Implement on-site renewable energy generation')
                region_strategies.append('Upgrade to more efficient cooling systems')
                
            if emissions[f'{region}_scope2'].mean() > 500:  # High scope 2
                region_strategies.append('Purchase renewable energy certificates')
                region_strategies.append('Implement energy storage solutions')
                
            if emissions[f'{region}_scope3'].mean() > 500:  # High scope 3
                region_strategies.append('Optimize supply chain logistics')
                region_strategies.append('Implement circular economy practices')
                
            if total_emissions > 2000:  # Very high emissions
                region_strategies.append('Consider carbon capture and storage')
                region_strategies.append('Implement comprehensive sustainability program')
                
            strategies[region] = region_strategies
            
        return strategies
    
    def calculate_sustainability_metrics(self,
                                      emissions: pd.DataFrame,
                                      power_consumption: pd.DataFrame,
                                      regions: List[str]) -> Dict[str, Dict]:
        """
        Calculate sustainability metrics
        
        Args:
            emissions: DataFrame with emissions data
            power_consumption: DataFrame with power consumption data
            regions: List of regions to analyze
            
        Returns:
            Dictionary with sustainability metrics by region
        """
        metrics = {}
        
        for region in regions:
            # Calculate PUE (Power Usage Effectiveness)
            total_power = power_consumption[region].sum()
            it_power = total_power * 0.7  # Assuming 70% IT load
            pue = total_power / it_power
            
            # Calculate carbon intensity
            total_emissions = emissions[f'{region}_total'].sum()
            carbon_intensity = total_emissions / total_power
            
            # Calculate renewable energy percentage
            renewable_percentage = self.parameters.renewable_energy_ratio * 100
            
            metrics[region] = {
                'pue': pue,
                'carbon_intensity': carbon_intensity,
                'renewable_percentage': renewable_percentage,
                'total_emissions': total_emissions,
                'emissions_per_kwh': carbon_intensity
            }
            
        return metrics 