"""
Interactive Dashboard Module for Data Center Analysis
"""

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from models.workload.workload_optimization_model import WorkloadOptimizationModel, WorkloadParameters
from models.security.security_model import SecurityModel, SecurityParameters
from analysis.workload_visualization import WorkloadVisualizer
from analysis.security_visualization import SecurityVisualizer

class DataCenterDashboard:
    """Interactive dashboard for data center analysis"""
    
    def __init__(self):
        self.app = dash.Dash(__name__)
        self.workload_visualizer = WorkloadVisualizer()
        self.security_visualizer = SecurityVisualizer()
        self.setup_layout()
        self.setup_callbacks()
        
    def setup_layout(self):
        """Setup dashboard layout"""
        self.app.layout = html.Div([
            html.H1("Data Center Analysis Dashboard"),
            
            # Region selection
            html.Div([
                html.H3("Region Selection"),
                dcc.Dropdown(
                    id='region-selector',
                    options=[
                        {'label': 'North America', 'value': 'North America'},
                        {'label': 'Europe', 'value': 'Europe'},
                        {'label': 'Asia Pacific', 'value': 'Asia Pacific'}
                    ],
                    value=['North America'],
                    multi=True
                )
            ]),
            
            # Time range selection
            html.Div([
                html.H3("Time Range"),
                dcc.DatePickerRange(
                    id='date-range',
                    start_date='2025-01-01',
                    end_date='2030-01-01'
                )
            ]),
            
            # Analysis type selection
            html.Div([
                html.H3("Analysis Type"),
                dcc.RadioItems(
                    id='analysis-type',
                    options=[
                        {'label': 'Current State', 'value': 'current'},
                        {'label': 'Trend Analysis', 'value': 'trend'},
                        {'label': 'Predictive Analysis', 'value': 'predictive'},
                        {'label': 'Comparative Analysis', 'value': 'comparative'}
                    ],
                    value='current'
                )
            ]),
            
            # Tabs for different analyses
            dcc.Tabs([
                # Workload Optimization Tab
                dcc.Tab(label='Workload Optimization', children=[
                    html.Div([
                        html.H3("Workload Patterns"),
                        dcc.Graph(id='workload-patterns'),
                        
                        html.H3("Resource Utilization"),
                        dcc.Graph(id='utilization-metrics'),
                        
                        html.H3("Optimization Dashboard"),
                        dcc.Graph(id='optimization-dashboard'),
                        
                        html.H3("Cost Breakdown"),
                        dcc.Graph(id='workload-costs'),
                        
                        html.H3("Optimization Strategies"),
                        dcc.Graph(id='workload-strategies'),
                        
                        html.H3("Trend Analysis"),
                        dcc.Graph(id='workload-trends'),
                        
                        html.H3("Predictive Metrics"),
                        dcc.Graph(id='workload-predictions')
                    ])
                ]),
                
                # Security Analysis Tab
                dcc.Tab(label='Security Analysis', children=[
                    html.Div([
                        html.H3("Threat Evolution"),
                        dcc.Graph(id='threat-evolution'),
                        
                        html.H3("Security Metrics"),
                        dcc.Graph(id='security-metrics'),
                        
                        html.H3("Security Dashboard"),
                        dcc.Graph(id='security-dashboard'),
                        
                        html.H3("Security Costs"),
                        dcc.Graph(id='security-costs'),
                        
                        html.H3("Security Strategies"),
                        dcc.Graph(id='security-strategies'),
                        
                        html.H3("Threat Trends"),
                        dcc.Graph(id='threat-trends'),
                        
                        html.H3("Risk Predictions"),
                        dcc.Graph(id='risk-predictions')
                    ])
                ]),
                
                # Combined Analysis Tab
                dcc.Tab(label='Combined Analysis', children=[
                    html.Div([
                        html.H3("Workload vs Security Correlation"),
                        dcc.Graph(id='workload-security-correlation'),
                        
                        html.H3("Cost Comparison"),
                        dcc.Graph(id='cost-comparison'),
                        
                        html.H3("Strategy Overlap"),
                        dcc.Graph(id='strategy-overlap'),
                        
                        html.H3("Risk vs Performance"),
                        dcc.Graph(id='risk-performance'),
                        
                        html.H3("Trend Correlation"),
                        dcc.Graph(id='trend-correlation'),
                        
                        html.H3("Predictive Insights"),
                        dcc.Graph(id='predictive-insights')
                    ])
                ])
            ])
        ])
        
    def setup_callbacks(self):
        """Setup dashboard callbacks"""
        
        @self.app.callback(
            [Output('workload-patterns', 'figure'),
             Output('utilization-metrics', 'figure'),
             Output('optimization-dashboard', 'figure'),
             Output('workload-costs', 'figure'),
             Output('workload-strategies', 'figure'),
             Output('workload-trends', 'figure'),
             Output('workload-predictions', 'figure')],
            [Input('region-selector', 'value'),
             Input('date-range', 'start_date'),
             Input('date-range', 'end_date'),
             Input('analysis-type', 'value')]
        )
        def update_workload_visualizations(regions, start_date, end_date, analysis_type):
            # Initialize models and run simulation
            workload_params = WorkloadParameters(
                base_utilization=0.7,
                workload_variability=0.2,
                optimization_threshold=0.8,
                scaling_factor=1.2,
                efficiency_target=0.9,
                cost_performance_ratio=0.85
            )
            workload_model = WorkloadOptimizationModel(workload_params)
            
            # Run simulation
            workload_patterns = workload_model.simulate_workload_patterns(5, regions, 1000)
            workload_requirements = workload_model.calculate_optimization_requirements(workload_patterns, regions)
            workload_metrics = workload_model.evaluate_optimization_metrics(workload_patterns, workload_requirements, regions)
            workload_strategies = workload_model.recommend_optimization_strategies(workload_metrics, workload_requirements, 1000000)
            workload_costs = workload_model.calculate_optimization_costs(workload_requirements, workload_metrics)
            
            # Create basic visualizations
            patterns_fig = self.workload_visualizer.plot_workload_patterns(workload_patterns, regions)
            utilization_fig = self.workload_visualizer.plot_utilization_metrics(workload_patterns, regions)
            dashboard_fig = self.workload_visualizer.create_optimization_dashboard(workload_metrics, workload_requirements)
            costs_fig = self.workload_visualizer.plot_cost_breakdown(workload_costs)
            strategies_fig = self.workload_visualizer.plot_optimization_strategies(workload_strategies)
            
            # Create trend analysis
            trends_fig = go.Figure()
            for region in regions:
                # Calculate moving averages
                workload_ma = workload_patterns[f'{region}_workload'].rolling(window=30).mean()
                utilization_ma = workload_patterns[f'{region}_utilization'].rolling(window=30).mean()
                
                trends_fig.add_trace(go.Scatter(
                    x=workload_patterns.index,
                    y=workload_ma,
                    name=f'{region} Workload Trend',
                    mode='lines'
                ))
                trends_fig.add_trace(go.Scatter(
                    x=workload_patterns.index,
                    y=utilization_ma,
                    name=f'{region} Utilization Trend',
                    mode='lines'
                ))
            trends_fig.update_layout(
                title='Workload and Utilization Trends',
                xaxis_title='Time',
                yaxis_title='Value'
            )
            
            # Create predictive analysis
            predictions_fig = go.Figure()
            for region in regions:
                # Simple linear regression for prediction
                x = np.arange(len(workload_patterns))
                y = workload_patterns[f'{region}_workload'].values
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                
                # Extend prediction for next 6 months
                future_x = np.arange(len(workload_patterns), len(workload_patterns) + 180)
                future_y = p(future_x)
                
                predictions_fig.add_trace(go.Scatter(
                    x=workload_patterns.index,
                    y=workload_patterns[f'{region}_workload'],
                    name=f'{region} Actual',
                    mode='lines'
                ))
                predictions_fig.add_trace(go.Scatter(
                    x=pd.date_range(start=workload_patterns.index[-1], periods=181, freq='D')[1:],
                    y=future_y,
                    name=f'{region} Predicted',
                    mode='lines',
                    line=dict(dash='dash')
                ))
            predictions_fig.update_layout(
                title='Workload Predictions',
                xaxis_title='Time',
                yaxis_title='Workload'
            )
            
            return patterns_fig, utilization_fig, dashboard_fig, costs_fig, strategies_fig, trends_fig, predictions_fig
        
        @self.app.callback(
            [Output('threat-evolution', 'figure'),
             Output('security-metrics', 'figure'),
             Output('security-dashboard', 'figure'),
             Output('security-costs', 'figure'),
             Output('security-strategies', 'figure'),
             Output('threat-trends', 'figure'),
             Output('risk-predictions', 'figure')],
            [Input('region-selector', 'value'),
             Input('date-range', 'start_date'),
             Input('date-range', 'end_date'),
             Input('analysis-type', 'value')]
        )
        def update_security_visualizations(regions, start_date, end_date, analysis_type):
            # Initialize models and run simulation
            security_params = SecurityParameters(
                base_threat_level=0.3,
                attack_frequency=0.1,
                detection_rate=0.9,
                response_time=4.0,
                vulnerability_rate=0.05,
                security_budget_ratio=0.1
            )
            security_model = SecurityModel(security_params)
            
            # Run simulation
            security_threats = security_model.simulate_security_threats(5, regions)
            security_requirements = security_model.calculate_security_requirements(security_threats, regions)
            security_metrics = security_model.evaluate_security_metrics(security_threats, security_requirements, regions)
            security_strategies = security_model.recommend_security_strategies(security_metrics, security_requirements, 1000000)
            security_costs = security_model.calculate_security_costs(security_requirements, security_metrics)
            
            # Create basic visualizations
            threats_fig = self.security_visualizer.plot_threat_evolution(security_threats, regions)
            metrics_fig = self.security_visualizer.plot_security_metrics(security_metrics)
            dashboard_fig = self.security_visualizer.create_security_dashboard(
                security_threats, security_metrics, security_costs, security_strategies, regions
            )
            costs_fig = self.security_visualizer.plot_security_costs(security_costs)
            strategies_fig = self.security_visualizer.plot_security_strategies(security_strategies)
            
            # Create trend analysis
            trends_fig = go.Figure()
            for region in regions:
                # Calculate moving averages
                attacks_ma = security_threats[f'{region}_attacks'].rolling(window=30).mean()
                severity_ma = security_threats[f'{region}_severity'].rolling(window=30).mean()
                
                trends_fig.add_trace(go.Scatter(
                    x=security_threats.index,
                    y=attacks_ma,
                    name=f'{region} Attack Trend',
                    mode='lines'
                ))
                trends_fig.add_trace(go.Scatter(
                    x=security_threats.index,
                    y=severity_ma,
                    name=f'{region} Severity Trend',
                    mode='lines'
                ))
            trends_fig.update_layout(
                title='Security Threat Trends',
                xaxis_title='Time',
                yaxis_title='Value'
            )
            
            # Create predictive analysis
            predictions_fig = go.Figure()
            for region in regions:
                # Simple linear regression for prediction
                x = np.arange(len(security_threats))
                y = security_threats[f'{region}_severity'].values
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                
                # Extend prediction for next 6 months
                future_x = np.arange(len(security_threats), len(security_threats) + 180)
                future_y = p(future_x)
                
                predictions_fig.add_trace(go.Scatter(
                    x=security_threats.index,
                    y=security_threats[f'{region}_severity'],
                    name=f'{region} Actual',
                    mode='lines'
                ))
                predictions_fig.add_trace(go.Scatter(
                    x=pd.date_range(start=security_threats.index[-1], periods=181, freq='D')[1:],
                    y=future_y,
                    name=f'{region} Predicted',
                    mode='lines',
                    line=dict(dash='dash')
                ))
            predictions_fig.update_layout(
                title='Risk Level Predictions',
                xaxis_title='Time',
                yaxis_title='Risk Level'
            )
            
            return threats_fig, metrics_fig, dashboard_fig, costs_fig, strategies_fig, trends_fig, predictions_fig
        
        @self.app.callback(
            [Output('workload-security-correlation', 'figure'),
             Output('cost-comparison', 'figure'),
             Output('strategy-overlap', 'figure'),
             Output('risk-performance', 'figure'),
             Output('trend-correlation', 'figure'),
             Output('predictive-insights', 'figure')],
            [Input('region-selector', 'value'),
             Input('date-range', 'start_date'),
             Input('date-range', 'end_date'),
             Input('analysis-type', 'value')]
        )
        def update_combined_visualizations(regions, start_date, end_date, analysis_type):
            # Initialize models and run simulations
            workload_params = WorkloadParameters(
                base_utilization=0.7,
                workload_variability=0.2,
                optimization_threshold=0.8,
                scaling_factor=1.2,
                efficiency_target=0.9,
                cost_performance_ratio=0.85
            )
            security_params = SecurityParameters(
                base_threat_level=0.3,
                attack_frequency=0.1,
                detection_rate=0.9,
                response_time=4.0,
                vulnerability_rate=0.05,
                security_budget_ratio=0.1
            )
            
            workload_model = WorkloadOptimizationModel(workload_params)
            security_model = SecurityModel(security_params)
            
            # Run simulations
            workload_patterns = workload_model.simulate_workload_patterns(5, regions, 1000)
            workload_requirements = workload_model.calculate_optimization_requirements(workload_patterns, regions)
            workload_metrics = workload_model.evaluate_optimization_metrics(workload_patterns, workload_requirements, regions)
            workload_strategies = workload_model.recommend_optimization_strategies(workload_metrics, workload_requirements, 1000000)
            workload_costs = workload_model.calculate_optimization_costs(workload_requirements, workload_metrics)
            
            security_threats = security_model.simulate_security_threats(5, regions)
            security_requirements = security_model.calculate_security_requirements(security_threats, regions)
            security_metrics = security_model.evaluate_security_metrics(security_threats, security_requirements, regions)
            security_strategies = security_model.recommend_security_strategies(security_metrics, security_requirements, 1000000)
            security_costs = security_model.calculate_security_costs(security_requirements, security_metrics)
            
            # Create correlation plot
            correlation_fig = go.Figure()
            for region in regions:
                correlation_fig.add_trace(go.Scatter(
                    x=workload_patterns[f'{region}_workload'],
                    y=security_threats[f'{region}_severity'],
                    mode='markers',
                    name=region
                ))
            correlation_fig.update_layout(
                title='Workload vs Security Threat Correlation',
                xaxis_title='Workload',
                yaxis_title='Threat Severity'
            )
            
            # Create cost comparison plot
            cost_comparison_fig = go.Figure()
            for region in regions:
                cost_comparison_fig.add_trace(go.Bar(
                    name=f'{region} - Workload',
                    x=['Workload Cost'],
                    y=[workload_costs[region]['total_cost']],
                    text=[f'${workload_costs[region]["total_cost"]:,.2f}'],
                    textposition='auto'
                ))
                cost_comparison_fig.add_trace(go.Bar(
                    name=f'{region} - Security',
                    x=['Security Cost'],
                    y=[security_costs[region]['total_cost']],
                    text=[f'${security_costs[region]["total_cost"]:,.2f}'],
                    textposition='auto'
                ))
            cost_comparison_fig.update_layout(
                title='Cost Comparison',
                barmode='group'
            )
            
            # Create strategy overlap plot
            strategy_overlap_fig = go.Figure()
            for region in regions:
                workload_set = set(workload_strategies[region])
                security_set = set(security_strategies[region])
                overlap = len(workload_set.intersection(security_set))
                strategy_overlap_fig.add_trace(go.Bar(
                    name=region,
                    x=['Strategy Overlap'],
                    y=[overlap],
                    text=[f'{overlap} strategies'],
                    textposition='auto'
                ))
            strategy_overlap_fig.update_layout(
                title='Strategy Overlap Analysis'
            )
            
            # Create risk-performance plot
            risk_performance_fig = go.Figure()
            for region in regions:
                risk_performance_fig.add_trace(go.Scatter(
                    x=[security_metrics[region]['risk_level']],
                    y=[workload_metrics[region]['optimization_score']],
                    mode='markers+text',
                    name=region,
                    text=[region],
                    textposition='top center'
                ))
            risk_performance_fig.update_layout(
                title='Risk vs Performance Analysis',
                xaxis_title='Security Risk Level',
                yaxis_title='Workload Optimization Score'
            )
            
            # Create trend correlation plot
            trend_correlation_fig = go.Figure()
            for region in regions:
                # Calculate moving averages
                workload_ma = workload_patterns[f'{region}_workload'].rolling(window=30).mean()
                threat_ma = security_threats[f'{region}_severity'].rolling(window=30).mean()
                
                trend_correlation_fig.add_trace(go.Scatter(
                    x=workload_patterns.index,
                    y=workload_ma,
                    name=f'{region} Workload Trend',
                    mode='lines'
                ))
                trend_correlation_fig.add_trace(go.Scatter(
                    x=security_threats.index,
                    y=threat_ma,
                    name=f'{region} Threat Trend',
                    mode='lines'
                ))
            trend_correlation_fig.update_layout(
                title='Workload and Threat Trend Correlation',
                xaxis_title='Time',
                yaxis_title='Value'
            )
            
            # Create predictive insights plot
            predictive_insights_fig = go.Figure()
            for region in regions:
                # Simple linear regression for predictions
                x = np.arange(len(workload_patterns))
                y_workload = workload_patterns[f'{region}_workload'].values
                y_threat = security_threats[f'{region}_severity'].values
                
                z_workload = np.polyfit(x, y_workload, 1)
                z_threat = np.polyfit(x, y_threat, 1)
                
                p_workload = np.poly1d(z_workload)
                p_threat = np.poly1d(z_threat)
                
                # Extend predictions for next 6 months
                future_x = np.arange(len(workload_patterns), len(workload_patterns) + 180)
                future_workload = p_workload(future_x)
                future_threat = p_threat(future_x)
                
                predictive_insights_fig.add_trace(go.Scatter(
                    x=pd.date_range(start=workload_patterns.index[-1], periods=181, freq='D')[1:],
                    y=future_workload,
                    name=f'{region} Predicted Workload',
                    mode='lines',
                    line=dict(dash='dash')
                ))
                predictive_insights_fig.add_trace(go.Scatter(
                    x=pd.date_range(start=security_threats.index[-1], periods=181, freq='D')[1:],
                    y=future_threat,
                    name=f'{region} Predicted Threat',
                    mode='lines',
                    line=dict(dash='dot')
                ))
            predictive_insights_fig.update_layout(
                title='Predictive Insights',
                xaxis_title='Time',
                yaxis_title='Value'
            )
            
            return correlation_fig, cost_comparison_fig, strategy_overlap_fig, risk_performance_fig, trend_correlation_fig, predictive_insights_fig
    
    def run_server(self, debug: bool = True, port: int = 8050):
        """Run the dashboard server"""
        self.app.run_server(debug=debug, port=port)

if __name__ == "__main__":
    dashboard = DataCenterDashboard()
    dashboard.run_server() 