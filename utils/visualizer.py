"""
Visualization Utilities
=======================
Functions for creating charts and visualizations using Plotly.
"""

import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Any, Optional


class DataVisualizer:
    """
    Data visualization class for creating professional charts using Plotly.
    """
    
    def __init__(self, df: pd.DataFrame, output_dir: str = "outputs/charts"):
        self.df = df
        self.output_dir = output_dir
        self.charts = []
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Color palette
        self.colors = px.colors.qualitative.Set2
        self.color_scale = "RdBu_r"
    
    def create_histogram(self, 
                        column: str, 
                        title: str = None,
                        nbins: int = 30,
                        show_stats: bool = True) -> go.Figure:
        """
        Create a histogram for a numerical column.
        
        Args:
            column: Column name
            title: Chart title
            nbins: Number of bins
            show_stats: Whether to show mean/median lines
            
        Returns:
            Plotly Figure
        """
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found")
        
        title = title or f"Distribution of {column}"
        
        fig = px.histogram(
            self.df, 
            x=column, 
            nbins=nbins,
            title=title,
            color_discrete_sequence=[self.colors[0]]
        )
        
        if show_stats:
            mean_val = self.df[column].mean()
            median_val = self.df[column].median()
            
            fig.add_vline(
                x=mean_val, 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"Mean: {mean_val:.2f}"
            )
            fig.add_vline(
                x=median_val, 
                line_dash="dot", 
                line_color="green",
                annotation_text=f"Median: {median_val:.2f}"
            )
        
        fig.update_layout(
            xaxis_title=column,
            yaxis_title="Count",
            template="plotly_white",
            height=500,
            width=800
        )
        
        self.charts.append({"name": f"histogram_{column}", "figure": fig})
        return fig
    
    def create_correlation_heatmap(self, 
                                   title: str = "Correlation Matrix",
                                   annotate: bool = True) -> go.Figure:
        """
        Create a correlation heatmap for numerical columns.
        
        Args:
            title: Chart title
            annotate: Whether to annotate values
            
        Returns:
            Plotly Figure
        """
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) < 2:
            raise ValueError("Need at least 2 numerical columns for correlation")
        
        corr_matrix = self.df[numerical_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale=self.color_scale,
            zmin=-1,
            zmax=1,
            text=corr_matrix.values.round(2) if annotate else None,
            texttemplate="%{text}" if annotate else None,
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=title,
            template="plotly_white",
            height=600,
            width=800,
            xaxis_title="",
            yaxis_title=""
        )
        
        self.charts.append({"name": "correlation_heatmap", "figure": fig})
        return fig
    
    def create_bar_chart(self,
                        column: str,
                        title: str = None,
                        top_n: int = 10,
                        horizontal: bool = True) -> go.Figure:
        """
        Create a bar chart for categorical data.
        
        Args:
            column: Column name
            title: Chart title
            top_n: Number of top values to show
            horizontal: Whether to make horizontal bars
            
        Returns:
            Plotly Figure
        """
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found")
        
        title = title or f"Top {top_n} Values - {column}"
        
        value_counts = self.df[column].value_counts().head(top_n)
        
        if horizontal:
            fig = go.Figure(go.Bar(
                x=value_counts.values,
                y=value_counts.index,
                orientation='h',
                marker_color=self.colors[1]
            ))
            fig.update_layout(
                xaxis_title="Count",
                yaxis_title=column,
                yaxis={'categoryorder': 'total ascending'}
            )
        else:
            fig = go.Figure(go.Bar(
                x=value_counts.index,
                y=value_counts.values,
                marker_color=self.colors[1]
            ))
            fig.update_layout(
                xaxis_title=column,
                yaxis_title="Count"
            )
        
        fig.update_layout(
            title=title,
            template="plotly_white",
            height=500,
            width=800
        )
        
        self.charts.append({"name": f"bar_{column}", "figure": fig})
        return fig
    
    def create_scatter_plot(self,
                           x_column: str,
                           y_column: str,
                           color_column: str = None,
                           title: str = None,
                           add_trendline: bool = True) -> go.Figure:
        """
        Create a scatter plot between two variables.
        
        Args:
            x_column: X-axis column
            y_column: Y-axis column
            color_column: Column for color encoding
            title: Chart title
            add_trendline: Whether to add OLS trendline
            
        Returns:
            Plotly Figure
        """
        title = title or f"{y_column} vs {x_column}"
        
        trendline = "ols" if add_trendline else None
        
        fig = px.scatter(
            self.df,
            x=x_column,
            y=y_column,
            color=color_column,
            title=title,
            trendline=trendline,
            color_discrete_sequence=self.colors
        )
        
        fig.update_layout(
            template="plotly_white",
            height=500,
            width=800
        )
        
        self.charts.append({"name": f"scatter_{x_column}_{y_column}", "figure": fig})
        return fig
    
    def create_box_plot(self,
                       column: str,
                       group_by: str = None,
                       title: str = None) -> go.Figure:
        """
        Create a box plot for a numerical column.
        
        Args:
            column: Numerical column
            group_by: Categorical column for grouping
            title: Chart title
            
        Returns:
            Plotly Figure
        """
        title = title or f"Box Plot - {column}"
        
        fig = px.box(
            self.df,
            x=group_by,
            y=column,
            title=title,
            color=group_by,
            color_discrete_sequence=self.colors
        )
        
        fig.update_layout(
            template="plotly_white",
            height=500,
            width=800
        )
        
        self.charts.append({"name": f"box_{column}", "figure": fig})
        return fig
    
    def create_line_chart(self,
                         x_column: str,
                         y_column: str,
                         title: str = None) -> go.Figure:
        """
        Create a line chart for time series data.
        
        Args:
            x_column: X-axis column (usually date/time)
            y_column: Y-axis column
            title: Chart title
            
        Returns:
            Plotly Figure
        """
        title = title or f"{y_column} over {x_column}"
        
        fig = px.line(
            self.df.sort_values(x_column),
            x=x_column,
            y=y_column,
            title=title,
            color_discrete_sequence=[self.colors[2]]
        )
        
        fig.update_layout(
            template="plotly_white",
            height=500,
            width=800
        )
        
        self.charts.append({"name": f"line_{x_column}_{y_column}", "figure": fig})
        return fig
    
    def create_pie_chart(self,
                        column: str,
                        title: str = None,
                        top_n: int = 10) -> go.Figure:
        """
        Create a pie chart for categorical data.
        
        Args:
            column: Column name
            title: Chart title
            top_n: Number of top categories
            
        Returns:
            Plotly Figure
        """
        title = title or f"Distribution of {column}"
        
        value_counts = self.df[column].value_counts().head(top_n)
        
        fig = px.pie(
            values=value_counts.values,
            names=value_counts.index,
            title=title,
            color_discrete_sequence=self.colors
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        
        fig.update_layout(
            template="plotly_white",
            height=500,
            width=800
        )
        
        self.charts.append({"name": f"pie_{column}", "figure": fig})
        return fig
    
    def create_kpi_cards(self, metrics: Dict[str, Any]) -> go.Figure:
        """
        Create KPI cards dashboard.
        
        Args:
            metrics: Dictionary of metric names and values
            
        Returns:
            Plotly Figure
        """
        n_metrics = len(metrics)
        
        fig = make_subplots(
            rows=1, 
            cols=n_metrics,
            specs=[[{"type": "indicator"}] * n_metrics]
        )
        
        for i, (name, value) in enumerate(metrics.items()):
            fig.add_trace(
                go.Indicator(
                    mode="number",
                    value=value if isinstance(value, (int, float)) else 0,
                    title={"text": name},
                    number={"font": {"size": 40}}
                ),
                row=1, col=i+1
            )
        
        fig.update_layout(
            title="Key Metrics Dashboard",
            template="plotly_white",
            height=300,
            width=200 * n_metrics
        )
        
        self.charts.append({"name": "kpi_cards", "figure": fig})
        return fig
    
    def create_dashboard(self) -> go.Figure:
        """
        Create a comprehensive dashboard with multiple charts.
        
        Returns:
            Plotly Figure
        """
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # Determine layout
        n_numerical = min(len(numerical_cols), 2)
        has_correlation = len(numerical_cols) >= 2
        has_categorical = len(categorical_cols) > 0
        
        total_plots = n_numerical + (1 if has_correlation else 0) + (1 if has_categorical else 0)
        
        rows = (total_plots + 1) // 2
        
        fig = make_subplots(
            rows=rows, 
            cols=2,
            subplot_titles=[
                *[f"Distribution: {col}" for col in numerical_cols[:n_numerical]],
                "Correlation Matrix" if has_correlation else None,
                f"Categories: {categorical_cols[0]}" if has_categorical else None
            ],
            specs=[[{"type": "xy"}, {"type": "xy"}]] * rows
        )
        
        plot_idx = 0
        
        # Add histograms
        for i, col in enumerate(numerical_cols[:n_numerical]):
            row = (plot_idx // 2) + 1
            col_pos = (plot_idx % 2) + 1
            
            fig.add_trace(
                go.Histogram(x=self.df[col], name=col, marker_color=self.colors[i]),
                row=row, col=col_pos
            )
            plot_idx += 1
        
        fig.update_layout(
            title="Data Overview Dashboard",
            template="plotly_white",
            height=400 * rows,
            width=1200,
            showlegend=False
        )
        
        self.charts.append({"name": "dashboard", "figure": fig})
        return fig
    
    def auto_generate_charts(self) -> List[go.Figure]:
        """
        Automatically generate appropriate charts based on data types.
        
        Returns:
            List of Plotly Figures
        """
        figures = []
        
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # Histograms for first 3 numerical columns
        for col in numerical_cols[:3]:
            try:
                fig = self.create_histogram(col)
                figures.append(fig)
            except Exception:
                pass
        
        # Correlation heatmap if we have multiple numerical columns
        if len(numerical_cols) >= 2:
            try:
                fig = self.create_correlation_heatmap()
                figures.append(fig)
            except Exception:
                pass
        
        # Bar charts for first 2 categorical columns
        for col in categorical_cols[:2]:
            try:
                fig = self.create_bar_chart(col)
                figures.append(fig)
            except Exception:
                pass
        
        # Scatter plot for first two numerical columns
        if len(numerical_cols) >= 2:
            try:
                fig = self.create_scatter_plot(
                    numerical_cols[0], 
                    numerical_cols[1],
                    add_trendline=True
                )
                figures.append(fig)
            except Exception:
                pass
        
        # Box plot for first numerical with first categorical
        if len(numerical_cols) >= 1 and len(categorical_cols) >= 1:
            try:
                fig = self.create_box_plot(
                    numerical_cols[0],
                    group_by=categorical_cols[0]
                )
                figures.append(fig)
            except Exception:
                pass
        
        return figures
    
    def save_charts(self, format: str = "html") -> List[str]:
        """
        Save all generated charts to files.
        
        Args:
            format: "html" or "png"
            
        Returns:
            List of saved file paths
        """
        saved_files = []
        
        for chart in self.charts:
            file_name = f"{chart['name']}.{format}"
            file_path = os.path.join(self.output_dir, file_name)
            
            if format == "html":
                chart['figure'].write_html(file_path)
            elif format == "png":
                chart['figure'].write_image(file_path, width=1200, height=800)
            
            saved_files.append(file_path)
        
        return saved_files
    
    def get_chart_summary(self) -> str:
        """
        Get a text summary of all generated charts.
        
        Returns:
            Summary string
        """
        if not self.charts:
            return "No charts generated yet."
        
        summary_parts = ["Generated Visualizations:"]
        
        for i, chart in enumerate(self.charts, 1):
            summary_parts.append(f"  {i}. {chart['name']}")
        
        return "\n".join(summary_parts)


def auto_visualize(df: pd.DataFrame, output_dir: str = "outputs/charts") -> DataVisualizer:
    """
    Quick utility to auto-generate visualizations.
    
    Args:
        df: DataFrame to visualize
        output_dir: Directory to save charts
        
    Returns:
        DataVisualizer instance with generated charts
    """
    visualizer = DataVisualizer(df, output_dir)
    visualizer.auto_generate_charts()
    return visualizer
