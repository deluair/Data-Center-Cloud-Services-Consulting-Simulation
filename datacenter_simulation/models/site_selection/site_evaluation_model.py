"""
Site Selection and Evaluation Model for Data Center Location Analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel

class SiteParameters(BaseModel):
    """Parameters for site evaluation"""
    power_weight: float  # Weight for power availability/cost
    water_weight: float  # Weight for water availability
    network_weight: float  # Weight for network connectivity
    disaster_weight: float  # Weight for natural disaster risk
    tax_weight: float  # Weight for tax incentives
    climate_weight: float  # Weight for climate suitability

class SiteEvaluationModel:
    """Evaluates potential data center sites based on multiple criteria"""
    
    def __init__(self, parameters: SiteParameters):
        self.parameters = parameters
        
    def evaluate_sites(self,
                      sites: List[Dict],
                      electricity_prices: Dict[str, float],
                      water_risks: Dict[str, float],
                      network_latency: Dict[str, float]) -> pd.DataFrame:
        """
        Evaluate potential sites based on multiple criteria
        
        Args:
            sites: List of site dictionaries with location data
            electricity_prices: Dictionary of electricity prices by region
            water_risks: Dictionary of water risk scores by region
            network_latency: Dictionary of network latency by region
            
        Returns:
            DataFrame with site evaluations
        """
        results = []
        
        for site in sites:
            # Calculate power score
            power_score = self._calculate_power_score(
                site['region'],
                electricity_prices[site['region']],
                site['power_availability']
            )
            
            # Calculate water score
            water_score = self._calculate_water_score(
                site['region'],
                water_risks[site['region']],
                site['water_availability']
            )
            
            # Calculate network score
            network_score = self._calculate_network_score(
                site['region'],
                network_latency[site['region']],
                site['fiber_availability']
            )
            
            # Calculate disaster risk score
            disaster_score = self._calculate_disaster_score(
                site['disaster_risk'],
                site['flood_risk'],
                site['earthquake_risk']
            )
            
            # Calculate tax incentive score
            tax_score = self._calculate_tax_score(
                site['tax_incentives'],
                site['property_tax_rate'],
                site['sales_tax_rate']
            )
            
            # Calculate climate score
            climate_score = self._calculate_climate_score(
                site['average_temperature'],
                site['humidity'],
                site['free_cooling_hours']
            )
            
            # Calculate weighted total score
            total_score = (
                power_score * self.parameters.power_weight +
                water_score * self.parameters.water_weight +
                network_score * self.parameters.network_weight +
                disaster_score * self.parameters.disaster_weight +
                tax_score * self.parameters.tax_weight +
                climate_score * self.parameters.climate_weight
            )
            
            results.append({
                'site_id': site['id'],
                'region': site['region'],
                'power_score': power_score,
                'water_score': water_score,
                'network_score': network_score,
                'disaster_score': disaster_score,
                'tax_score': tax_score,
                'climate_score': climate_score,
                'total_score': total_score
            })
            
        return pd.DataFrame(results)
    
    def _calculate_power_score(self,
                             region: str,
                             electricity_price: float,
                             power_availability: float) -> float:
        """Calculate power availability and cost score"""
        # Normalize electricity price (lower is better)
        price_score = 1 - (electricity_price / max(electricity_price, 100))
        
        # Combine with power availability
        return (price_score + power_availability) / 2
    
    def _calculate_water_score(self,
                             region: str,
                             water_risk: float,
                             water_availability: float) -> float:
        """Calculate water availability and risk score"""
        # Normalize water risk (lower is better)
        risk_score = 1 - water_risk
        
        # Combine with water availability
        return (risk_score + water_availability) / 2
    
    def _calculate_network_score(self,
                               region: str,
                               latency: float,
                               fiber_availability: float) -> float:
        """Calculate network connectivity score"""
        # Normalize latency (lower is better)
        latency_score = 1 - (latency / max(latency, 100))
        
        # Combine with fiber availability
        return (latency_score + fiber_availability) / 2
    
    def _calculate_disaster_score(self,
                                disaster_risk: float,
                                flood_risk: float,
                                earthquake_risk: float) -> float:
        """Calculate natural disaster risk score"""
        # Combine different risk factors
        total_risk = (disaster_risk + flood_risk + earthquake_risk) / 3
        
        # Convert to score (lower risk is better)
        return 1 - total_risk
    
    def _calculate_tax_score(self,
                           tax_incentives: float,
                           property_tax_rate: float,
                           sales_tax_rate: float) -> float:
        """Calculate tax incentive score"""
        # Normalize tax rates (lower is better)
        property_tax_score = 1 - (property_tax_rate / max(property_tax_rate, 0.02))
        sales_tax_score = 1 - (sales_tax_rate / max(sales_tax_rate, 0.08))
        
        # Combine with tax incentives
        return (property_tax_score + sales_tax_score + tax_incentives) / 3
    
    def _calculate_climate_score(self,
                               avg_temperature: float,
                               humidity: float,
                               free_cooling_hours: float) -> float:
        """Calculate climate suitability score"""
        # Normalize temperature (optimal range is 15-25°C)
        temp_score = 1 - abs(avg_temperature - 20) / 20
        
        # Normalize humidity (lower is better)
        humidity_score = 1 - (humidity / 100)
        
        # Normalize free cooling hours
        cooling_score = free_cooling_hours / 8760  # Hours in a year
        
        return (temp_score + humidity_score + cooling_score) / 3
    
    def recommend_sites(self,
                       evaluations: pd.DataFrame,
                       min_score: float = 0.7,
                       max_sites: int = 5) -> List[Dict]:
        """
        Recommend best sites based on evaluation scores
        
        Args:
            evaluations: DataFrame with site evaluations
            min_score: Minimum total score for recommendation
            max_sites: Maximum number of sites to recommend
            
        Returns:
            List of recommended sites with their scores
        """
        # Filter by minimum score
        qualified_sites = evaluations[evaluations['total_score'] >= min_score]
        
        # Sort by total score
        sorted_sites = qualified_sites.sort_values('total_score', ascending=False)
        
        # Select top sites
        top_sites = sorted_sites.head(max_sites)
        
        # Convert to list of dictionaries
        recommendations = []
        for _, site in top_sites.iterrows():
            recommendations.append({
                'site_id': site['site_id'],
                'region': site['region'],
                'total_score': site['total_score'],
                'strengths': self._identify_strengths(site),
                'weaknesses': self._identify_weaknesses(site)
            })
            
        return recommendations
    
    def _identify_strengths(self, site: pd.Series) -> List[str]:
        """Identify site strengths based on scores"""
        strengths = []
        
        if site['power_score'] > 0.8:
            strengths.append('Excellent power availability and cost')
        if site['water_score'] > 0.8:
            strengths.append('Strong water availability and low risk')
        if site['network_score'] > 0.8:
            strengths.append('Superior network connectivity')
        if site['disaster_score'] > 0.8:
            strengths.append('Very low natural disaster risk')
        if site['tax_score'] > 0.8:
            strengths.append('Favorable tax environment')
        if site['climate_score'] > 0.8:
            strengths.append('Ideal climate for data center operations')
            
        return strengths
    
    def _identify_weaknesses(self, site: pd.Series) -> List[str]:
        """Identify site weaknesses based on scores"""
        weaknesses = []
        
        if site['power_score'] < 0.6:
            weaknesses.append('Power availability or cost concerns')
        if site['water_score'] < 0.6:
            weaknesses.append('Water availability or risk concerns')
        if site['network_score'] < 0.6:
            weaknesses.append('Network connectivity limitations')
        if site['disaster_score'] < 0.6:
            weaknesses.append('Elevated natural disaster risk')
        if site['tax_score'] < 0.6:
            weaknesses.append('Unfavorable tax environment')
        if site['climate_score'] < 0.6:
            weaknesses.append('Climate challenges for operations')
            
        return weaknesses 