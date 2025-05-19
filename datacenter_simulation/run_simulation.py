"""
Main Simulation Runner for Data Center & Cloud Services Consulting
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime

from models.power_grid.electricity_price_model import ElectricityPriceModel, ElectricityPriceParameters
from models.water_resources.water_risk_model import WaterRiskModel, WaterRiskParameters
from models.ai_workload.compute_density_model import ComputeDensityModel, ComputeDensityParameters
from models.site_selection.site_evaluation_model import SiteEvaluationModel, SiteParameters

class DataCenterSimulation:
    """Main simulation class that integrates all models"""
    
    def __init__(self):
        # Initialize model parameters
        self.electricity_params = ElectricityPriceParameters(
            base_price=50.0,  # $/MWh
            carbon_tax_rate=30.0,  # $/ton CO2
            transmission_congestion_factor=0.1,
            seasonal_variation=0.2,
            peak_hour_multiplier=1.5,
            renewable_integration_factor=0.3
        )
        
        self.water_params = WaterRiskParameters(
            base_water_stress=0.3,
            drought_frequency=0.1,
            regulatory_restriction_threshold=0.7,
            water_price_escalation=0.05,
            community_impact_factor=0.2
        )
        
        self.compute_params = ComputeDensityParameters(
            base_power_density=10.0,  # kW/rack
            ai_growth_rate=0.3,
            efficiency_improvement=0.1,
            training_ratio=0.3,
            edge_workload_ratio=0.2
        )
        
        self.site_params = SiteParameters(
            power_weight=0.3,
            water_weight=0.2,
            network_weight=0.2,
            disaster_weight=0.1,
            tax_weight=0.1,
            climate_weight=0.1
        )
        
        # Initialize models
        self.electricity_model = ElectricityPriceModel(self.electricity_params)
        self.water_model = WaterRiskModel(self.water_params)
        self.compute_model = ComputeDensityModel(self.compute_params)
        self.site_model = SiteEvaluationModel(self.site_params)
        
    def run_simulation(self,
                      years: int,
                      regions: List[str],
                      initial_workload: float,
                      facility_size: float) -> Dict:
        """
        Run the complete simulation
        
        Args:
            years: Number of years to simulate
            regions: List of regions to analyze
            initial_workload: Initial AI workload in TFLOPS
            facility_size: Facility size in square feet
            
        Returns:
            Dictionary with simulation results
        """
        # Simulate electricity prices
        renewable_penetration = {region: np.random.uniform(0.2, 0.8) for region in regions}
        electricity_prices = self.electricity_model.simulate_nodal_prices(
            years=years,
            regions=regions,
            renewable_penetration=renewable_penetration
        )
        
        # Simulate water risks
        water_stress = self.water_model.simulate_water_stress(
            years=years,
            regions=regions,
            cooling_technology='hybrid'
        )
        
        # Simulate AI workload evolution
        workload = self.compute_model.simulate_workload_evolution(
            years=years,
            initial_workload=initial_workload
        )
        
        # Calculate power density
        power_density = self.compute_model.calculate_power_density(
            workload=workload,
            facility_type='hyperscale'
        )
        
        # Estimate infrastructure requirements
        infrastructure = self.compute_model.estimate_infrastructure_requirements(
            power_density=power_density,
            facility_size=facility_size
        )
        
        # Prepare site data
        sites = self._generate_site_data(regions)
        
        # Evaluate sites
        site_evaluations = self.site_model.evaluate_sites(
            sites=sites,
            electricity_prices={region: prices.mean() for region, prices in electricity_prices.items()},
            water_risks={region: stress.mean() for region, stress in water_stress.items()},
            network_latency={region: np.random.uniform(1, 50) for region in regions}
        )
        
        # Get site recommendations
        recommendations = self.site_model.recommend_sites(site_evaluations)
        
        # Get optimization strategies
        strategies = self.compute_model.recommend_optimization_strategies(
            requirements=infrastructure,
            budget_constraint=1000000
        )
        
        return {
            'electricity_prices': electricity_prices,
            'water_stress': water_stress,
            'workload': workload,
            'power_density': power_density,
            'infrastructure': infrastructure,
            'site_evaluations': site_evaluations,
            'recommendations': recommendations,
            'strategies': strategies
        }
    
    def _generate_site_data(self, regions: List[str]) -> List[Dict]:
        """Generate sample site data for evaluation"""
        sites = []
        
        for i, region in enumerate(regions):
            sites.append({
                'id': f'site_{i+1}',
                'region': region,
                'power_availability': np.random.uniform(0.7, 1.0),
                'water_availability': np.random.uniform(0.7, 1.0),
                'fiber_availability': np.random.uniform(0.7, 1.0),
                'disaster_risk': np.random.uniform(0.1, 0.5),
                'flood_risk': np.random.uniform(0.1, 0.5),
                'earthquake_risk': np.random.uniform(0.1, 0.5),
                'tax_incentives': np.random.uniform(0.1, 0.5),
                'property_tax_rate': np.random.uniform(0.01, 0.03),
                'sales_tax_rate': np.random.uniform(0.05, 0.09),
                'average_temperature': np.random.uniform(10, 30),
                'humidity': np.random.uniform(30, 80),
                'free_cooling_hours': np.random.uniform(2000, 6000)
            })
            
        return sites

def main():
    """Run the simulation with sample parameters"""
    # Initialize simulation
    simulation = DataCenterSimulation()
    
    # Define simulation parameters
    years = 15  # 2025-2040
    regions = ['North America', 'Europe', 'Asia Pacific', 'Latin America', 'Middle East']
    initial_workload = 1000  # TFLOPS
    facility_size = 100000  # square feet
    
    # Run simulation
    results = simulation.run_simulation(
        years=years,
        regions=regions,
        initial_workload=initial_workload,
        facility_size=facility_size
    )
    
    # Print key findings
    print("\nData Center & Cloud Services Consulting Simulation Results (2025-2040)")
    print("=" * 80)
    
    print("\nRecommended Sites:")
    for site in results['recommendations']:
        print(f"\nRegion: {site['region']}")
        print(f"Total Score: {site['total_score']:.2f}")
        print("Strengths:")
        for strength in site['strengths']:
            print(f"  - {strength}")
        print("Weaknesses:")
        for weakness in site['weaknesses']:
            print(f"  - {weakness}")
    
    print("\nInfrastructure Requirements:")
    for key, value in results['infrastructure'].items():
        print(f"{key}: {value:.2f}")
    
    print("\nRecommended Optimization Strategies:")
    for strategy in results['strategies']:
        print(f"  - {strategy}")

if __name__ == "__main__":
    main() 