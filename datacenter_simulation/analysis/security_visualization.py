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
        plt.style.use('default')
        sns.set_palette("husl")
        
    def plot_threat_evolution(self,
                            threats: pd.DataFrame,
                            regions: List[str],
                            title: str = "Security Threat Evolution") -> go.Figure:
        """
        Create interactive plot of threat evolution
        
        Args:
            threats: DataFrame with security threat data (long format)
            regions: List of regions to plot
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        fig = make_subplots(rows=2, cols=1,
                          subplot_titles=('Attack Frequency', 'Threat Severity'))
        
        for region in regions:
            region_data = threats[threats['region'] == region]
            # Attack frequency (rolling mean for smoothing)
            attack_freq = region_data['attack_occurred'].rolling(window=7, min_periods=1).mean()
            fig.add_trace(
                go.Scatter(
                    x=region_data['timestamp'],
                    y=attack_freq,
                    name=f'{region} Attacks',
                    mode='lines'
                ),
                row=1, col=1
            )
            
            # Threat severity (rolling mean for smoothing)
            severity = region_data['attack_severity'].rolling(window=7, min_periods=1).mean()
            fig.add_trace(
                go.Scatter(
                    x=region_data['timestamp'],
                    y=severity,
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
                'Detection Effectiveness': metrics[region]['detection_effectiveness'],
                'Response Effectiveness': metrics[region]['response_effectiveness'],
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
                   ['Detection Effectiveness', 'Response Effectiveness', 'Threat Mitigation', 
                    'Security Score', 'Risk Level']],
                theta=['Detection Effectiveness', 'Response Effectiveness', 'Threat Mitigation', 
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
    
    def plot_security_strategies(self, strategies: Dict[str, List[str]], title: str = "Recommended Security Strategies") -> go.Figure:
        """
        Create visualization of recommended security strategies
        Args:
            strategies: Dictionary of recommended strategies by region
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
        if df.empty:
            # No strategies to show
            fig = go.Figure()
            fig.update_layout(title=title)
            return fig
        # Create heatmap of strategy presence
        strategy_matrix = pd.crosstab(df['Region'], df['Strategy'])
        fig = px.imshow(strategy_matrix,
                        title=title,
                        labels=dict(x="Strategy", y="Region", color="Present"),
                        aspect="auto")
        fig.update_layout(
            xaxis_title='Strategy',
            yaxis_title='Region',
            height=400,
            margin=dict(l=50, r=50, t=50, b=50)
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

        # Add threat evolution plot (severity)
        for region in regions:
            region_data = threats[threats['region'] == region]
            severity = region_data['attack_severity'].rolling(window=7, min_periods=1).mean()
            fig.add_trace(
                go.Scatter(
                    x=region_data['timestamp'],
                    y=severity,
                    name=f'{region} Severity',
                    mode='lines'
                ),
                row=1, col=1
            )

        # Add security metrics as grouped bar chart
        metric_names = ['detection_effectiveness', 'response_effectiveness', 'threat_mitigation', 'security_score']
        for metric_name in metric_names:
            fig.add_trace(
                go.Bar(
                    x=regions,
                    y=[metrics[region][metric_name] for region in regions],
                    name=metric_name.replace('_', ' ').title()
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
            showlegend=True,
            barmode='group'
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
        with open(output_file, 'w', encoding='utf-8') as f:
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