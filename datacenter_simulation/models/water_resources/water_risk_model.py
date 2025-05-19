"""
Water Risk Assessment Model for Data Center Operations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from pydantic import BaseModel

class WaterRiskParameters(BaseModel):
    """Parameters for water risk assessment"""
    base_water_stress: float  # Base water stress index (0-1)
    drought_frequency: float  # Annual drought probability
    regulatory_restriction_threshold: float  # Water stress level triggering restrictions
    water_price_escalation: float  # Annual water price increase rate
    community_impact_factor: float  # Impact of community relations on water access

class WaterRiskModel:
    """Simulates water resource risks and their impact on data center operations"""
    
    def __init__(self, parameters: WaterRiskParameters):
        self.parameters = parameters
        
    def simulate_water_stress(self,
                            years: int,
                            regions: List[str],
                            cooling_technology: str) -> pd.DataFrame:
        """
        Simulate water stress levels for different regions
        
        Args:
            years: Number of years to simulate
            regions: List of regions to simulate
            cooling_technology: Type of cooling technology used
            
        Returns:
            DataFrame with simulated water stress levels
        """
        dates = pd.date_range(start='2025-01-01', periods=years*365, freq='D')
        results = pd.DataFrame(index=dates)
        
        for region in regions:
            # Base water stress with regional variation
            base_stress = self.parameters.base_water_stress * (1 + np.random.normal(0, 0.1))
            
            # Add seasonal variation
            seasonal = np.sin(np.arange(len(dates)) * 2 * np.pi / 365) * 0.2
            
            # Simulate drought events
            drought_events = np.random.binomial(1, self.parameters.drought_frequency/365, len(dates))
            drought_impact = drought_events * 0.3
            
            # Add long-term trend (increasing water stress)
            trend = np.linspace(0, 0.2, len(dates))
            
            # Combine factors
            stress = base_stress + seasonal + drought_impact + trend
            stress = np.clip(stress, 0, 1)  # Ensure stress stays between 0 and 1
            
            results[f'{region}_stress'] = stress
            
        return results
    
    def calculate_operational_risk(self,
                                 stress_levels: pd.DataFrame,
                                 water_consumption: Dict[str, float],
                                 cooling_technology: str) -> Dict[str, Dict]:
        """
        Calculate operational risks based on water stress levels
        
        Args:
            stress_levels: DataFrame with simulated stress levels
            water_consumption: Dictionary of water consumption by region
            cooling_technology: Type of cooling technology used
            
        Returns:
            Dictionary with risk metrics by region
        """
        risk_metrics = {}
        
        for column in stress_levels.columns:
            region = column.replace('_stress', '')
            stress = stress_levels[column]
            
            # Calculate probability of operational restrictions
            restriction_prob = np.mean(stress > self.parameters.regulatory_restriction_threshold)
            
            # Estimate water cost escalation
            cost_escalation = (1 + self.parameters.water_price_escalation) ** len(stress_levels)
            
            # Calculate community impact risk
            community_risk = np.mean(stress) * self.parameters.community_impact_factor
            
            # Calculate cooling technology specific risks
            if cooling_technology == 'water_cooled':
                tech_risk = restriction_prob * 0.8
            elif cooling_technology == 'air_cooled':
                tech_risk = restriction_prob * 0.3
            else:  # hybrid or other
                tech_risk = restriction_prob * 0.5
                
            risk_metrics[region] = {
                'restriction_probability': restriction_prob,
                'cost_escalation_factor': cost_escalation,
                'community_risk': community_risk,
                'technology_specific_risk': tech_risk,
                'total_risk_score': (restriction_prob + community_risk + tech_risk) / 3
            }
            
        return risk_metrics
    
    def recommend_mitigation_strategies(self,
                                     risk_metrics: Dict[str, Dict],
                                     budget_constraint: float) -> Dict[str, List[str]]:
        """
        Recommend water risk mitigation strategies based on risk metrics
        
        Args:
            risk_metrics: Dictionary of risk metrics by region
            budget_constraint: Maximum budget for mitigation strategies
            
        Returns:
            Dictionary of recommended strategies by region
        """
        strategies = {}
        
        for region, metrics in risk_metrics.items():
            region_strategies = []
            
            if metrics['restriction_probability'] > 0.7:
                region_strategies.append('Implement water recycling system')
                region_strategies.append('Develop alternative water sources')
                
            if metrics['community_risk'] > 0.6:
                region_strategies.append('Establish community water stewardship program')
                region_strategies.append('Develop water conservation partnerships')
                
            if metrics['technology_specific_risk'] > 0.5:
                region_strategies.append('Upgrade to hybrid cooling system')
                region_strategies.append('Implement advanced water treatment')
                
            if metrics['total_risk_score'] > 0.8:
                region_strategies.append('Consider site relocation')
                
            strategies[region] = region_strategies
            
        return strategies 