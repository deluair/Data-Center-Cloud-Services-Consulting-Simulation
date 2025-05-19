"""
Security Model for Data Center Operations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel

class SecurityParameters(BaseModel):
    """Parameters for security simulation"""
    base_threat_level: float  # Base threat level (0-1)
    attack_frequency: float  # Annual attack frequency
    detection_rate: float  # Base detection rate
    response_time: float  # Average response time in hours
    vulnerability_rate: float  # Rate of new vulnerabilities
    security_budget_ratio: float  # Security budget as ratio of total budget

class SecurityModel:
    """Simulates security threats and defenses for data centers"""
    
    def __init__(self, parameters: SecurityParameters):
        self.parameters = parameters
        
    def simulate_security_threats(self,
                                years: int,
                                regions: List[str]) -> pd.DataFrame:
        """
        Simulate security threats and attacks
        
        Args:
            years: Number of years to simulate
            regions: List of regions to simulate
            
        Returns:
            DataFrame with security threat data
        """
        dates = pd.date_range(start='2025-01-01', periods=years*365, freq='D')
        results = pd.DataFrame(index=dates)
        
        # Calculate daily attack probability
        daily_probability = 1 - (1 - self.parameters.attack_frequency) ** (1/365)
        
        for region in regions:
            # Generate attack events
            attacks = np.random.binomial(1, daily_probability, len(dates))
            
            # Calculate attack severity (0-1)
            severity = np.random.beta(2, 5, len(dates)) * attacks
            
            # Calculate detection probability
            detection = np.random.binomial(1, self.parameters.detection_rate, len(dates)) * attacks
            
            # Calculate response effectiveness
            response = detection * (1 - np.random.exponential(self.parameters.response_time/24, len(dates)))
            
            results[f'{region}_attacks'] = attacks
            results[f'{region}_severity'] = severity
            results[f'{region}_detection'] = detection
            results[f'{region}_response'] = response
            
        return results
    
    def calculate_security_requirements(self,
                                     threats: pd.DataFrame,
                                     regions: List[str]) -> Dict[str, Dict]:
        """
        Calculate security infrastructure requirements
        
        Args:
            threats: DataFrame with security threat data
            regions: List of regions to analyze
            
        Returns:
            Dictionary with security requirements by region
        """
        requirements = {}
        
        for region in regions:
            # Calculate maximum threat level
            max_threat = threats[f'{region}_severity'].max()
            
            # Calculate required detection capability
            required_detection = max_threat * self.parameters.detection_rate
            
            # Calculate response requirements
            response_requirement = threats[f'{region}_response'].mean()
            
            # Calculate monitoring requirements
            monitoring_requirement = max_threat * 24  # 24/7 monitoring
            
            requirements[region] = {
                'max_threat': max_threat,
                'required_detection': required_detection,
                'response_requirement': response_requirement,
                'monitoring_requirement': monitoring_requirement,
                'vulnerability_scan_frequency': self.parameters.vulnerability_rate * 365
            }
            
        return requirements
    
    def evaluate_security_metrics(self,
                                threats: pd.DataFrame,
                                requirements: Dict[str, Dict],
                                regions: List[str]) -> Dict[str, Dict]:
        """
        Evaluate security metrics
        
        Args:
            threats: DataFrame with security threat data
            requirements: Dictionary with security requirements
            regions: List of regions to analyze
            
        Returns:
            Dictionary with security metrics by region
        """
        metrics = {}
        
        for region in regions:
            # Calculate detection effectiveness
            detection_rate = threats[f'{region}_detection'].sum() / threats[f'{region}_attacks'].sum()
            
            # Calculate response effectiveness
            response_rate = threats[f'{region}_response'].mean()
            
            # Calculate threat mitigation
            threat_mitigation = 1 - (threats[f'{region}_severity'].mean() * 
                                   (1 - detection_rate))
            
            # Calculate security score
            security_score = (detection_rate * 0.4 + 
                            response_rate * 0.3 + 
                            threat_mitigation * 0.3)
            
            metrics[region] = {
                'detection_rate': detection_rate,
                'response_rate': response_rate,
                'threat_mitigation': threat_mitigation,
                'security_score': security_score,
                'risk_level': 1 - security_score
            }
            
        return metrics
    
    def recommend_security_strategies(self,
                                   metrics: Dict[str, Dict],
                                   requirements: Dict[str, Dict],
                                   budget_constraint: float) -> Dict[str, List[str]]:
        """
        Recommend security improvement strategies
        
        Args:
            metrics: Dictionary with security metrics
            requirements: Dictionary with security requirements
            budget_constraint: Maximum budget for improvement
            
        Returns:
            Dictionary of recommended strategies by region
        """
        strategies = {}
        
        for region, metric in metrics.items():
            region_strategies = []
            
            if metric['detection_rate'] < 0.95:
                region_strategies.append('Implement advanced threat detection')
                region_strategies.append('Enhance monitoring systems')
                
            if metric['response_rate'] < 0.9:
                region_strategies.append('Improve incident response procedures')
                region_strategies.append('Implement automated response systems')
                
            if metric['threat_mitigation'] < 0.9:
                region_strategies.append('Enhance security controls')
                region_strategies.append('Implement zero-trust architecture')
                
            if metric['security_score'] < 0.95:
                region_strategies.append('Develop comprehensive security program')
                region_strategies.append('Implement security training')
                
            if metric['risk_level'] > 0.1:
                region_strategies.append('Implement risk mitigation measures')
                region_strategies.append('Enhance security monitoring')
                
            strategies[region] = region_strategies
            
        return strategies
    
    def calculate_security_costs(self,
                               requirements: Dict[str, Dict],
                               metrics: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Calculate security infrastructure costs
        
        Args:
            requirements: Dictionary with security requirements
            metrics: Dictionary with security metrics
            
        Returns:
            Dictionary with cost breakdown by region
        """
        costs = {}
        
        for region in requirements.keys():
            # Calculate detection costs
            detection_cost = 100000 * (1 - metrics[region]['detection_rate'])
            
            # Calculate response costs
            response_cost = 75000 * (1 - metrics[region]['response_rate'])
            
            # Calculate monitoring costs
            monitoring_cost = 50000 * requirements[region]['monitoring_requirement']
            
            # Calculate training costs
            training_cost = 25000 * (1 - metrics[region]['security_score'])
            
            # Calculate incident costs
            incident_cost = 1000000 * metrics[region]['risk_level']
            
            costs[region] = {
                'detection_cost': detection_cost,
                'response_cost': response_cost,
                'monitoring_cost': monitoring_cost,
                'training_cost': training_cost,
                'incident_cost': incident_cost,
                'total_cost': (detection_cost + response_cost + 
                             monitoring_cost + training_cost + incident_cost)
            }
            
        return costs 