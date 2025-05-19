"""
Security Analysis Visualization Module
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional
import plotly.express as px

class SecurityVisualizer:
    """Visualization tools for security analysis"""
    
    def __init__(self):
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
    def plot_threat_evolution(self,
                            threats: pd.DataFrame,
                            regions: List[str],
                            title: str = "Security Threat Evolution") -> go.Figure:
        """
        Create interactive plot of threat evolution
        
        Args:
            threats: DataFrame with security threat data
            regions: List of regions to plot
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        fig = make_subplots(rows=2, cols=1,
                          subplot_titles=('Attack Frequency', 'Threat Severity'))
        
        for region in regions:
            # Attack frequency plot
            fig.add_trace(
                go.Scatter(
                    x=threats.index,
                    y=threats[f'{region}_attacks'],
                    name=f'{region} Attacks',
                    mode='lines'
                ),
                row=1, col=1
            )
            
            # Threat severity plot
            fig.add_trace(
                go.Scatter(
                    x=threats.index,
                    y=threats[f'{region}_severity'],
                    name=f'{region} Severity',
                    mode='lines'
                ),
                row=2, col=1
            )
            
        fig.update_layout(
            title=title,
            height=800,
            showlegend=True
        )
        
        return fig
    
    def plot_security_metrics(self,
                            metrics: Dict[str, Dict],
                            title: str = "Security Performance Metrics") -> go.Figure:
        """
        Create interactive plot of security metrics
        
        Args:
            metrics: Dictionary with security metrics
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        regions = list(metrics.keys())
        
        # Prepare data for visualization
        metric_data = []
        for region in regions:
            metric_data.append({
                'Region': region,
                'Detection Rate': metrics[region]['detection_rate'],
                'Response Rate': metrics[region]['response_rate'],
                'Threat Mitigation': metrics[region]['threat_mitigation'],
                'Security Score': metrics[region]['security_score'],
                'Risk Level': metrics[region]['risk_level']
            })
        
        df = pd.DataFrame(metric_data)
        
        # Create radar chart
        fig = go.Figure()
        
        for region in regions:
            fig.add_trace(go.Scatterpolar(
                r=[df.loc[df['Region'] == region, col].values[0] for col in 
                   ['Detection Rate', 'Response Rate', 'Threat Mitigation', 
                    'Security Score', 'Risk Level']],
                theta=['Detection Rate', 'Response Rate', 'Threat Mitigation', 
                      'Security Score', 'Risk Level'],
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
    
    def plot_security_costs(self,
                          costs: Dict[str, Dict],
                          title: str = "Security Cost Breakdown") -> go.Figure:
        """
        Create cost breakdown visualization
        
        Args:
            costs: Dictionary with cost data
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        regions = list(costs.keys())
        cost_types = ['detection_cost', 'response_cost', 'monitoring_cost',
                     'training_cost', 'incident_cost']
        
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
    
    def plot_security_strategies(self,
                               strategies: Dict[str, List[str]],
                               title: str = "Recommended Security Strategies") -> go.Figure:
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
    
    def create_security_dashboard(self,
                                threats: pd.DataFrame,
                                metrics: Dict[str, Dict],
                                costs: Dict[str, Dict],
                                strategies: Dict[str, List[str]],
                                regions: List[str],
                                title: str = "Security Analysis Dashboard") -> go.Figure:
        """
        Create comprehensive security dashboard
        
        Args:
            threats: DataFrame with security threat data
            metrics: Dictionary with security metrics
            costs: Dictionary with cost data
            strategies: Dictionary of recommended strategies
            regions: List of regions analyzed
            title: Dashboard title
            
        Returns:
            Plotly figure object
        """
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                "Threat Evolution", "Security Metrics",
                "Cost Breakdown", "Risk Analysis",
                "Strategy Distribution", "Security Score"
            )
        )
        
        # Add threat evolution plot
        for region in regions:
            fig.add_trace(
                go.Scatter(
                    x=threats.index,
                    y=threats[f'{region}_severity'],
                    name=f'{region} Severity',
                    mode='lines'
                ),
                row=1, col=1
            )
        
        # Add security metrics radar chart
        for region in regions:
            fig.add_trace(
                go.Scatterpolar(
                    r=[metrics[region]['detection_rate'],
                       metrics[region]['response_rate'],
                       metrics[region]['threat_mitigation'],
                       metrics[region]['security_score']],
                    theta=['Detection', 'Response', 'Mitigation', 'Score'],
                    name=region,
                    fill='toself'
                ),
                row=1, col=2
            )
        
        # Add cost breakdown
        cost_types = ['detection_cost', 'response_cost', 'monitoring_cost',
                     'training_cost', 'incident_cost']
        for i, cost_type in enumerate(cost_types):
            fig.add_trace(
                go.Bar(
                    x=regions,
                    y=[costs[region][cost_type] for region in regions],
                    name=cost_type.replace('_', ' ').title()
                ),
                row=2, col=1
            )
        
        # Add risk analysis
        for region in regions:
            fig.add_trace(
                go.Bar(
                    x=[region],
                    y=[metrics[region]['risk_level']],
                    name=region
                ),
                row=2, col=2
            )
        
        # Add strategy distribution
        strategy_counts = {region: len(strategies[region]) for region in regions}
        fig.add_trace(
            go.Bar(
                x=list(strategy_counts.keys()),
                y=list(strategy_counts.values()),
                name='Strategy Count'
            ),
            row=3, col=1
        )
        
        # Add security score
        for region in regions:
            fig.add_trace(
                go.Bar(
                    x=[region],
                    y=[metrics[region]['security_score']],
                    name=region
                ),
                row=3, col=2
            )
        
        fig.update_layout(
            title=title,
            height=1200,
            showlegend=True
        )
        
        return fig
    
    def save_security_analysis(self,
                             threats: pd.DataFrame,
                             metrics: Dict[str, Dict],
                             costs: Dict[str, Dict],
                             strategies: Dict[str, List[str]],
                             regions: List[str],
                             output_file: str = "security_analysis.html"):
        """
        Save complete security analysis visualization
        
        Args:
            threats: DataFrame with security threat data
            metrics: Dictionary with security metrics
            costs: Dictionary with cost data
            strategies: Dictionary of recommended strategies
            regions: List of regions analyzed
            output_file: Output HTML file path
        """
        # Create all visualizations
        threat_fig = self.plot_threat_evolution(threats, regions)
        metrics_fig = self.plot_security_metrics(metrics)
        cost_fig = self.plot_security_costs(costs)
        strategy_fig = self.plot_security_strategies(strategies)
        dashboard_fig = self.create_security_dashboard(
            threats, metrics, costs, strategies, regions
        )
        
        # Combine all figures into a single HTML file
        with open(output_file, 'w') as f:
            f.write('<html><head><title>Security Analysis</title></head><body>')
            
            # Add each figure
            for fig, title in [(threat_fig, "Threat Evolution"),
                             (metrics_fig, "Security Metrics"),
                             (cost_fig, "Cost Breakdown"),
                             (strategy_fig, "Security Strategies"),
                             (dashboard_fig, "Security Dashboard")]:
                f.write(f'<h2>{title}</h2>')
                f.write(fig.to_html(full_html=False))
                f.write('<hr>')
            
            f.write('</body></html>') 