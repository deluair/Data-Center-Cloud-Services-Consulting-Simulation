"""
Visualization tools for Data Center & Cloud Services Consulting Simulation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class SimulationVisualizer:
    """Visualization tools for simulation results"""
    
    def __init__(self):
        # Set style
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
    def plot_electricity_prices(self, prices: pd.DataFrame, title: str = "Electricity Price Evolution"):
        """
        Plot electricity price evolution over time
        
        Args:
            prices: DataFrame with electricity prices
            title: Plot title
        """
        plt.figure(figsize=(12, 6))
        
        for column in prices.columns:
            plt.plot(prices.index, prices[column], label=column.replace('_price', ''))
            
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Price ($/MWh)')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_water_stress(self, stress: pd.DataFrame, title: str = "Water Stress Evolution"):
        """
        Plot water stress evolution over time
        
        Args:
            stress: DataFrame with water stress levels
            title: Plot title
        """
        plt.figure(figsize=(12, 6))
        
        for column in stress.columns:
            plt.plot(stress.index, stress[column], label=column.replace('_stress', ''))
            
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Water Stress Index')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_workload_evolution(self, workload: pd.DataFrame, title: str = "AI Workload Evolution"):
        """
        Plot AI workload evolution over time
        
        Args:
            workload: DataFrame with workload data
            title: Plot title
        """
        plt.figure(figsize=(12, 6))
        
        for column in workload.columns:
            plt.plot(workload.index, workload[column], label=column.replace('_workload', ''))
            
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Workload (TFLOPS)')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_power_density(self, density: pd.DataFrame, title: str = "Power Density Evolution"):
        """
        Plot power density evolution over time
        
        Args:
            density: DataFrame with power density data
            title: Plot title
        """
        plt.figure(figsize=(12, 6))
        
        plt.plot(density.index, density['power_density'], label='Power Density')
        plt.plot(density.index, density['efficiency_factor'], label='Efficiency Factor')
        
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return plt.gcf()
    
    def create_site_comparison(self, evaluations: pd.DataFrame, title: str = "Site Comparison"):
        """
        Create a radar chart comparing site evaluations
        
        Args:
            evaluations: DataFrame with site evaluations
            title: Plot title
        """
        # Prepare data for radar chart
        categories = ['power_score', 'water_score', 'network_score', 
                     'disaster_score', 'tax_score', 'climate_score']
        
        fig = go.Figure()
        
        for _, site in evaluations.iterrows():
            values = [site[cat] for cat in categories]
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=site['region']
            ))
            
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title=title
        )
        
        return fig
    
    def create_infrastructure_dashboard(self, 
                                      infrastructure: Dict[str, float],
                                      title: str = "Infrastructure Requirements"):
        """
        Create a dashboard of infrastructure requirements
        
        Args:
            infrastructure: Dictionary with infrastructure requirements
            title: Dashboard title
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Power Requirements",
                "Cooling Requirements",
                "Rack Count",
                "Efficiency Metrics"
            )
        )
        
        # Power requirements
        fig.add_trace(
            go.Bar(
                x=['Total Power', 'Peak Power'],
                y=[infrastructure['total_power_requirement'], 
                   infrastructure['peak_power_requirement']],
                name='Power (kW)'
            ),
            row=1, col=1
        )
        
        # Cooling requirements
        fig.add_trace(
            go.Bar(
                x=['Cooling Capacity'],
                y=[infrastructure['cooling_capacity']],
                name='Cooling (kW)'
            ),
            row=1, col=2
        )
        
        # Rack count
        fig.add_trace(
            go.Bar(
                x=['Total Racks'],
                y=[infrastructure['total_racks']],
                name='Racks'
            ),
            row=2, col=1
        )
        
        # Efficiency metrics
        fig.add_trace(
            go.Bar(
                x=['Avg Density', 'Peak Density'],
                y=[infrastructure['average_power_density'],
                   infrastructure['peak_power_density']],
                name='Density (kW/rack)'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text=title,
            showlegend=True
        )
        
        return fig
    
    def save_plots(self, figures: List[plt.Figure], filename: str):
        """
        Save multiple plots to a single PDF file
        
        Args:
            figures: List of matplotlib figures
            filename: Output filename
        """
        from matplotlib.backends.backend_pdf import PdfPages
        
        with PdfPages(filename) as pdf:
            for fig in figures:
                pdf.savefig(fig)
                plt.close(fig)
                
    def save_plotly_figures(self, figures: List[go.Figure], filenames: List[str]):
        """
        Save multiple Plotly figures to HTML files
        
        Args:
            figures: List of Plotly figures
            filenames: List of output filenames
        """
        for fig, filename in zip(figures, filenames):
            fig.write_html(filename) 