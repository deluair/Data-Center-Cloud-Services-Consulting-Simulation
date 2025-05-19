"""
Electricity Price Simulation Model for Data Center TCO Analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from pydantic import BaseModel

class ElectricityPriceParameters(BaseModel):
    """Parameters for electricity price simulation"""
    base_price: float  # Base electricity price in $/MWh
    carbon_tax_rate: float  # Carbon tax rate in $/ton CO2
    transmission_congestion_factor: float  # Congestion premium factor
    seasonal_variation: float  # Seasonal price variation factor
    peak_hour_multiplier: float  # Peak vs off-peak price multiplier
    renewable_integration_factor: float  # Impact of renewable integration

class ElectricityPriceModel:
    """Simulates nodal electricity prices and their impact on TCO"""
    
    def __init__(self, parameters: ElectricityPriceParameters):
        self.parameters = parameters
        
    def simulate_nodal_prices(self, 
                            years: int,
                            regions: List[str],
                            renewable_penetration: Dict[str, float]) -> pd.DataFrame:
        """
        Simulate nodal electricity prices for different regions
        
        Args:
            years: Number of years to simulate
            regions: List of regions to simulate
            renewable_penetration: Dictionary of renewable penetration by region
            
        Returns:
            DataFrame with simulated prices
        """
        # Initialize results DataFrame
        dates = pd.date_range(start='2025-01-01', periods=years*365, freq='D')
        results = pd.DataFrame(index=dates)
        
        for region in regions:
            # Base price with regional variation
            base = self.parameters.base_price * (1 + np.random.normal(0, 0.1))
            
            # Add seasonal variation
            seasonal = np.sin(np.arange(len(dates)) * 2 * np.pi / 365) * self.parameters.seasonal_variation
            
            # Add peak/off-peak variation
            peak_hours = np.random.choice([0, 1], size=len(dates), p=[0.7, 0.3])
            peak_variation = peak_hours * self.parameters.peak_hour_multiplier
            
            # Add renewable integration impact
            renewable_impact = renewable_penetration[region] * self.parameters.renewable_integration_factor
            
            # Add carbon tax impact
            carbon_impact = self.parameters.carbon_tax_rate * (1 - renewable_penetration[region])
            
            # Add transmission congestion
            congestion = np.random.normal(0, self.parameters.transmission_congestion_factor, len(dates))
            
            # Combine all factors
            prices = base * (1 + seasonal + peak_variation - renewable_impact + carbon_impact + congestion)
            
            results[f'{region}_price'] = prices
            
        return results
    
    def calculate_tco_impact(self, 
                           prices: pd.DataFrame,
                           power_consumption: float,
                           years: int) -> Dict[str, float]:
        """
        Calculate TCO impact of electricity prices
        
        Args:
            prices: DataFrame with simulated prices
            power_consumption: Annual power consumption in MWh
            years: Number of years to analyze
            
        Returns:
            Dictionary with TCO impact metrics
        """
        annual_costs = {}
        for column in prices.columns:
            region = column.replace('_price', '')
            annual_cost = prices[column].mean() * power_consumption
            annual_costs[region] = annual_cost
            
        return {
            'annual_costs': annual_costs,
            'total_tco': sum(annual_costs.values()) * years,
            'min_cost_region': min(annual_costs.items(), key=lambda x: x[1])[0],
            'max_cost_region': max(annual_costs.items(), key=lambda x: x[1])[0]
        } 