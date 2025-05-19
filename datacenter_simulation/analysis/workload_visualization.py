"""
Workload Optimization Visualization Module
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional
import plotly.express as px

class WorkloadVisualizer:
    """Visualization tools for workload optimization analysis"""
    
    def __init__(self):
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
    def plot_workload_patterns(self,
                             patterns: pd.DataFrame,
                             regions: List[str],
                             title: str = "Workload Patterns Over Time") -> go.Figure:
        """
        Create interactive plot of workload patterns
        
        Args:
            patterns: DataFrame with workload pattern data
            regions: List of regions to plot
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        for region in regions:
            fig.add_trace(go.Scatter(
                x=patterns.index,
                y=patterns[f'{region}_workload'],
                name=f'{region} Workload',
                mode='lines'
            ))
            
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Workload',
            hovermode='x unified',
            showlegend=True
        )
        
        return fig
    
    def plot_utilization_metrics(self,
                               patterns: pd.DataFrame,
                               regions: List[str],
                               title: str = "Resource Utilization Metrics") -> go.Figure:
        """
        Create interactive plot of utilization metrics
        
        Args:
            patterns: DataFrame with workload pattern data
            regions: List of regions to plot
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        fig = make_subplots(rows=2, cols=1,
                          subplot_titles=('Utilization Over Time', 'Utilization Distribution'))
        
        for region in regions:
            # Time series plot
            fig.add_trace(
                go.Scatter(
                    x=patterns.index,
                    y=patterns[f'{region}_utilization'],
                    name=f'{region} Utilization',
                    mode='lines'
                ),
                row=1, col=1
            )
            
            # Distribution plot
            fig.add_trace(
                go.Histogram(
                    x=patterns[f'{region}_utilization'],
                    name=f'{region} Distribution',
                    opacity=0.7
                ),
                row=2, col=1
            )
            
        fig.update_layout(
            title=title,
            height=800,
            showlegend=True
        )
        
        return fig
    
    def create_optimization_dashboard(self,
                                    metrics: Dict[str, Dict],
                                    requirements: Dict[str, Dict],
                                    title: str = "Workload Optimization Dashboard") -> go.Figure:
        """
        Create comprehensive optimization dashboard
        
        Args:
            metrics: Dictionary with optimization metrics
            requirements: Dictionary with optimization requirements
            title: Dashboard title
            
        Returns:
            Plotly figure object
        """
        regions = list(metrics.keys())
        
        # Prepare data for visualization
        optimization_data = []
        for region in regions:
            optimization_data.append({
                'Region': region,
                'Utilization': metrics[region]['utilization'],
                'Optimization Effectiveness': metrics[region]['optimization_effectiveness'],
                'Efficiency': metrics[region]['efficiency'],
                'Cost Performance': metrics[region]['cost_performance'],
                'Optimization Score': metrics[region]['optimization_score']
            })
        
        df = pd.DataFrame(optimization_data)
        
        # Create radar chart
        fig = go.Figure()
        
        for region in regions:
            fig.add_trace(go.Scatterpolar(
                r=[df.loc[df['Region'] == region, col].values[0] for col in 
                   ['Utilization', 'Optimization Effectiveness', 'Efficiency', 
                    'Cost Performance', 'Optimization Score']],
                theta=['Utilization', 'Optimization Effectiveness', 'Efficiency', 
                      'Cost Performance', 'Optimization Score'],
                name=region,
                fill='toself'
            ))
            
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title=title,
            showlegend=True
        )
        
        return fig
    
    def plot_cost_breakdown(self,
                          costs: Dict[str, Dict],
                          title: str = "Optimization Cost Breakdown") -> go.Figure:
        """
        Create cost breakdown visualization
        
        Args:
            costs: Dictionary with cost data
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        regions = list(costs.keys())
        cost_types = ['infrastructure_cost', 'optimization_cost', 'efficiency_cost',
                     'performance_cost', 'maintenance_cost']
        
        # Prepare data for visualization
        cost_data = []
        for region in regions:
            for cost_type in cost_types:
                cost_data.append({
                    'Region': region,
                    'Cost Type': cost_type.replace('_', ' ').title(),
                    'Cost': costs[region][cost_type]
                })
        
        df = pd.DataFrame(cost_data)
        
        fig = px.bar(df, x='Region', y='Cost', color='Cost Type',
                    title=title, barmode='stack')
        
        fig.update_layout(
            xaxis_title='Region',
            yaxis_title='Cost ($)',
            showlegend=True
        )
        
        return fig
    
    def plot_optimization_strategies(self,
                                   strategies: Dict[str, List[str]],
                                   title: str = "Recommended Optimization Strategies") -> go.Figure:
        """
        Create visualization of recommended strategies
        
        Args:
            strategies: Dictionary of recommended strategies
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        # Prepare data for visualization
        strategy_data = []
        for region, region_strategies in strategies.items():
            for strategy in region_strategies:
                strategy_data.append({
                    'Region': region,
                    'Strategy': strategy
                })
        
        df = pd.DataFrame(strategy_data)
        
        # Create heatmap of strategy presence
        strategy_matrix = pd.crosstab(df['Region'], df['Strategy'])
        
        fig = px.imshow(strategy_matrix,
                       title=title,
                       labels=dict(x="Strategy", y="Region", color="Present"),
                       aspect="auto")
        
        fig.update_layout(
            xaxis_title='Strategy',
            yaxis_title='Region'
        )
        
        return fig
    
    def save_workload_analysis(self,
                             patterns: pd.DataFrame,
                             metrics: Dict[str, Dict],
                             requirements: Dict[str, Dict],
                             costs: Dict[str, Dict],
                             strategies: Dict[str, List[str]],
                             regions: List[str],
                             output_file: str = "workload_analysis.html"):
        """
        Save complete workload analysis visualization
        
        Args:
            patterns: DataFrame with workload pattern data
            metrics: Dictionary with optimization metrics
            requirements: Dictionary with optimization requirements
            costs: Dictionary with cost data
            strategies: Dictionary of recommended strategies
            regions: List of regions analyzed
            output_file: Output HTML file path
        """
        # Create all visualizations
        workload_fig = self.plot_workload_patterns(patterns, regions)
        utilization_fig = self.plot_utilization_metrics(patterns, regions)
        dashboard_fig = self.create_optimization_dashboard(metrics, requirements)
        cost_fig = self.plot_cost_breakdown(costs)
        strategy_fig = self.plot_optimization_strategies(strategies)
        
        # Combine all figures into a single HTML file
        with open(output_file, 'w') as f:
            f.write('<html><head><title>Workload Optimization Analysis</title></head><body>')
            
            # Add each figure
            for fig, title in [(workload_fig, "Workload Patterns"),
                             (utilization_fig, "Utilization Metrics"),
                             (dashboard_fig, "Optimization Dashboard"),
                             (cost_fig, "Cost Breakdown"),
                             (strategy_fig, "Optimization Strategies")]:
                f.write(f'<h2>{title}</h2>')
                f.write(fig.to_html(full_html=False))
                f.write('<hr>')
            
            f.write('</body></html>') 