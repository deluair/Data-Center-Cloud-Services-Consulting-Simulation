"""
Network Architecture Model for Data Center Operations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel

class NetworkParameters(BaseModel):
    """Parameters for network architecture simulation"""
    base_bandwidth: float  # Base bandwidth in Gbps
    latency_threshold: float  # Maximum acceptable latency in ms
    redundancy_factor: float  # Network redundancy factor
    traffic_growth_rate: float  # Annual traffic growth rate
    peering_ratio: float  # Ratio of peered traffic
    security_overhead: float  # Security processing overhead

class NetworkArchitectureModel:
    """Simulates network architecture and performance for data centers"""
    
    def __init__(self, parameters: NetworkParameters):
        self.parameters = parameters
        
    def simulate_network_traffic(self,
                               years: int,
                               regions: List[str],
                               initial_traffic: float) -> pd.DataFrame:
        """
        Simulate network traffic evolution
        
        Args:
            years: Number of years to simulate
            regions: List of regions to simulate
            initial_traffic: Initial traffic in Gbps
            
        Returns:
            DataFrame with traffic data
        """
        dates = pd.date_range(start='2025-01-01', periods=years*365, freq='D')
        results = pd.DataFrame(index=dates)
        
        # Calculate daily growth factor
        daily_growth = (1 + self.parameters.traffic_growth_rate) ** (1/365)
        
        for region in regions:
            # Simulate base traffic growth
            base_traffic = initial_traffic * (daily_growth ** np.arange(len(dates)))
            
            # Add daily variation
            daily_variation = np.random.normal(1, 0.1, len(dates))
            traffic = base_traffic * daily_variation
            
            # Add peering traffic
            peered_traffic = traffic * self.parameters.peering_ratio
            
            # Add security overhead
            security_traffic = traffic * self.parameters.security_overhead
            
            results[f'{region}_total'] = traffic
            results[f'{region}_peered'] = peered_traffic
            results[f'{region}_security'] = security_traffic
            
        return results
    
    def calculate_network_requirements(self,
                                    traffic: pd.DataFrame,
                                    regions: List[str]) -> Dict[str, Dict]:
        """
        Calculate network infrastructure requirements
        
        Args:
            traffic: DataFrame with traffic data
            regions: List of regions to analyze
            
        Returns:
            Dictionary with network requirements by region
        """
        requirements = {}
        
        for region in regions:
            # Calculate peak traffic
            peak_traffic = traffic[f'{region}_total'].max()
            
            # Calculate required bandwidth with redundancy
            required_bandwidth = peak_traffic * self.parameters.redundancy_factor
            
            # Calculate number of network devices
            devices_per_rack = 2  # Assuming 2 network devices per rack
            total_devices = int(np.ceil(required_bandwidth / self.parameters.base_bandwidth))
            
            # Calculate network latency
            base_latency = 5  # Base latency in ms
            latency = base_latency * (1 + np.random.normal(0, 0.1))
            
            requirements[region] = {
                'peak_traffic': peak_traffic,
                'required_bandwidth': required_bandwidth,
                'total_devices': total_devices,
                'average_latency': latency,
                'redundancy_level': self.parameters.redundancy_factor
            }
            
        return requirements
    
    def evaluate_network_performance(self,
                                   traffic: pd.DataFrame,
                                   requirements: Dict[str, Dict],
                                   regions: List[str]) -> Dict[str, Dict]:
        """
        Evaluate network performance metrics
        
        Args:
            traffic: DataFrame with traffic data
            requirements: Dictionary with network requirements
            regions: List of regions to analyze
            
        Returns:
            Dictionary with performance metrics by region
        """
        performance = {}
        
        for region in regions:
            # Calculate bandwidth utilization
            avg_traffic = traffic[f'{region}_total'].mean()
            utilization = avg_traffic / requirements[region]['required_bandwidth']
            
            # Calculate packet loss probability
            packet_loss = 0.001 * (utilization ** 2)  # Simple model
            
            # Calculate network availability
            device_reliability = 0.999  # 99.9% reliability per device
            total_devices = requirements[region]['total_devices']
            availability = device_reliability ** total_devices
            
            # Calculate security effectiveness
            security_traffic = traffic[f'{region}_security'].mean()
            security_effectiveness = 1 - (security_traffic / avg_traffic)
            
            performance[region] = {
                'bandwidth_utilization': utilization,
                'packet_loss_rate': packet_loss,
                'network_availability': availability,
                'security_effectiveness': security_effectiveness,
                'average_latency': requirements[region]['average_latency']
            }
            
        return performance
    
    def recommend_network_optimizations(self,
                                     performance: Dict[str, Dict],
                                     requirements: Dict[str, Dict],
                                     budget_constraint: float) -> Dict[str, List[str]]:
        """
        Recommend network optimization strategies
        
        Args:
            performance: Dictionary with performance metrics
            requirements: Dictionary with network requirements
            budget_constraint: Maximum budget for optimization
            
        Returns:
            Dictionary of recommended strategies by region
        """
        strategies = {}
        
        for region, metrics in performance.items():
            region_strategies = []
            
            if metrics['bandwidth_utilization'] > 0.8:
                region_strategies.append('Upgrade network bandwidth')
                region_strategies.append('Implement traffic shaping')
                
            if metrics['packet_loss_rate'] > 0.001:
                region_strategies.append('Optimize network routing')
                region_strategies.append('Implement QoS policies')
                
            if metrics['network_availability'] < 0.999:
                region_strategies.append('Add network redundancy')
                region_strategies.append('Implement failover systems')
                
            if metrics['security_effectiveness'] < 0.95:
                region_strategies.append('Upgrade security infrastructure')
                region_strategies.append('Implement advanced threat protection')
                
            if metrics['average_latency'] > self.parameters.latency_threshold:
                region_strategies.append('Optimize network topology')
                region_strategies.append('Implement edge caching')
                
            strategies[region] = region_strategies
            
        return strategies
    
    def calculate_network_costs(self,
                              requirements: Dict[str, Dict],
                              performance: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Calculate network infrastructure costs
        
        Args:
            requirements: Dictionary with network requirements
            performance: Dictionary with performance metrics
            
        Returns:
            Dictionary with cost breakdown by region
        """
        costs = {}
        
        for region in requirements.keys():
            # Calculate hardware costs
            device_cost = 10000  # Cost per network device
            hardware_cost = requirements[region]['total_devices'] * device_cost
            
            # Calculate bandwidth costs
            bandwidth_cost = requirements[region]['required_bandwidth'] * 100  # $100 per Gbps
            
            # Calculate maintenance costs
            maintenance_cost = hardware_cost * 0.1  # 10% of hardware cost per year
            
            # Calculate security costs
            security_cost = hardware_cost * 0.2  # 20% of hardware cost for security
            
            costs[region] = {
                'hardware_cost': hardware_cost,
                'bandwidth_cost': bandwidth_cost,
                'maintenance_cost': maintenance_cost,
                'security_cost': security_cost,
                'total_cost': hardware_cost + bandwidth_cost + maintenance_cost + security_cost
            }
            
        return costs 