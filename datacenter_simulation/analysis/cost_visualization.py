"""
Cost Analysis Visualization Tools for Data Center & Cloud Services Consulting Simulation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class CostVisualizer:
    """Visualization tools for cost analysis"""
    
    def __init__(self):
        # Set style
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
    def plot_tco_breakdown(self, 
                          costs: Dict[str, float],
                          title: str = "Total Cost of Ownership Breakdown") -> go.Figure:
        """
        Create a pie chart showing TCO breakdown
        
        Args:
            costs: Dictionary with cost components
            title: Plot title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure(data=[go.Pie(
            labels=list(costs.keys()),
            values=list(costs.values()),
            hole=.3
        )])
        
        fig.update_layout(
            title=title,
            annotations=[dict(text='TCO', x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
        
        return fig
    
    def plot_cost_evolution(self,
                           costs: pd.DataFrame,
                           title: str = "Cost Evolution Over Time") -> go.Figure:
        """
        Plot cost evolution over time
        
        Args:
            costs: DataFrame with cost data
            title: Plot title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        for column in costs.columns:
            fig.add_trace(go.Scatter(
                x=costs.index,
                y=costs[column],
                name=column.replace('_cost', ''),
                mode='lines'
            ))
            
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Cost ($)',
            hovermode='x unified'
        )
        
        return fig
    
    def create_cost_comparison(self,
                             costs_by_region: Dict[str, Dict[str, float]],
                             title: str = "Cost Comparison by Region") -> go.Figure:
        """
        Create a bar chart comparing costs across regions
        
        Args:
            costs_by_region: Dictionary of costs by region
            title: Plot title
            
        Returns:
            Plotly figure
        """
        # Prepare data
        regions = list(costs_by_region.keys())
        cost_types = list(costs_by_region[regions[0]].keys())
        
        fig = go.Figure()
        
        for cost_type in cost_types:
            values = [costs_by_region[region][cost_type] for region in regions]
            fig.add_trace(go.Bar(
                name=cost_type,
                x=regions,
                y=values
            ))
            
        fig.update_layout(
            title=title,
            xaxis_title='Region',
            yaxis_title='Cost ($)',
            barmode='group'
        )
        
        return fig
    
    def plot_cost_sensitivity(self,
                            base_costs: Dict[str, float],
                            sensitivity_results: Dict[str, Dict[str, float]],
                            title: str = "Cost Sensitivity Analysis") -> go.Figure:
        """
        Plot cost sensitivity analysis results
        
        Args:
            base_costs: Dictionary with base case costs
            sensitivity_results: Dictionary with sensitivity analysis results
            title: Plot title
            
        Returns:
            Plotly figure
        """
        # Prepare data
        parameters = list(sensitivity_results.keys())
        cost_types = list(base_costs.keys())
        
        fig = go.Figure()
        
        for cost_type in cost_types:
            base_value = base_costs[cost_type]
            values = [sensitivity_results[param][cost_type] / base_value - 1 
                     for param in parameters]
            
            fig.add_trace(go.Bar(
                name=cost_type,
                x=parameters,
                y=values,
                text=[f'{v:.1%}' for v in values],
                textposition='auto'
            ))
            
        fig.update_layout(
            title=title,
            xaxis_title='Parameter',
            yaxis_title='Cost Impact',
            yaxis_tickformat='.1%'
        )
        
        return fig
    
    def create_cost_dashboard(self,
                            costs: Dict[str, Dict[str, float]],
                            metrics: Dict[str, float],
                            title: str = "Cost Analysis Dashboard") -> go.Figure:
        """
        Create a comprehensive cost analysis dashboard
        
        Args:
            costs: Dictionary with cost data
            metrics: Dictionary with cost metrics
            title: Dashboard title
            
        Returns:
            Plotly figure
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Cost Breakdown",
                "Cost Trends",
                "Key Metrics",
                "Cost Efficiency"
            )
        )
        
        # Cost breakdown pie chart
        fig.add_trace(
            go.Pie(
                labels=list(costs['breakdown'].keys()),
                values=list(costs['breakdown'].values()),
                hole=.3
            ),
            row=1, col=1
        )
        
        # Cost trends line chart
        for cost_type, values in costs['trends'].items():
            fig.add_trace(
                go.Scatter(
                    x=list(values.keys()),
                    y=list(values.values()),
                    name=cost_type
                ),
                row=1, col=2
            )
        
        # Key metrics table
        metrics_table = go.Table(
            header=dict(
                values=['Metric', 'Value'],
                fill_color='paleturquoise',
                align='left'
            ),
            cells=dict(
                values=[list(metrics.keys()), list(metrics.values())],
                fill_color='lavender',
                align='left'
            )
        )
        fig.add_trace(metrics_table, row=2, col=1)
        
        # Cost efficiency bar chart
        fig.add_trace(
            go.Bar(
                x=list(costs['efficiency'].keys()),
                y=list(costs['efficiency'].values())
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text=title,
            showlegend=True
        )
        
        return fig
    
    def plot_cost_optimization(self,
                             optimization_results: Dict[str, List[float]],
                             title: str = "Cost Optimization Results") -> go.Figure:
        """
        Plot cost optimization results
        
        Args:
            optimization_results: Dictionary with optimization results
            title: Plot title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        for strategy, costs in optimization_results.items():
            fig.add_trace(go.Scatter(
                x=list(range(len(costs))),
                y=costs,
                name=strategy,
                mode='lines+markers'
            ))
            
        fig.update_layout(
            title=title,
            xaxis_title='Iteration',
            yaxis_title='Total Cost ($)',
            hovermode='x unified'
        )
        
        return fig
    
    def save_cost_analysis(self,
                          figures: List[go.Figure],
                          filenames: List[str]):
        """
        Save cost analysis visualizations
        
        Args:
            figures: List of Plotly figures
            filenames: List of output filenames
        """
        for fig, filename in zip(figures, filenames):
            fig.write_html(filename) 