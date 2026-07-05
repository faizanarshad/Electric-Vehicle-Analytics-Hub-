#!/usr/bin/env python3
"""
🚗 Modern AI-Powered Electric Vehicle Analysis Dashboard using Gradio
Features: Beautiful UI, Natural Language Queries, AI Insights, Interactive Analytics
"""

import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')
import functools
import argparse

# Load and preprocess data
@functools.lru_cache(maxsize=1)
def load_data():
    """Load and cache data efficiently"""
    try:
        df = pd.read_csv('data/raw/electric_vehicles_spec_2025.csv.csv')
        
        # Data cleaning and preprocessing
        numeric_columns = [
            'top_speed_kmh', 'battery_capacity_kWh', 'number_of_cells', 'torque_nm',
            'efficiency_wh_per_km', 'range_km', 'acceleration_0_100_s', 
            'fast_charging_power_kw_dc', 'towing_capacity_kg', 'cargo_volume_l',
            'seats', 'length_mm', 'width_mm', 'height_mm'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Create derived features
        df['segment'] = df['segment'].str.strip()
        df['segment_category'] = df['segment'].str.split(' - ').str[0]
        df['range_per_kwh'] = df['range_km'] / df['battery_capacity_kWh']
        df['power_to_weight_ratio'] = df['torque_nm'] / (df['length_mm'] * df['width_mm'] * df['height_mm'] * 1e-9)
        
        return df
    except Exception as e:
        return pd.DataFrame()

# Load data
df = load_data()

# AI Analysis Functions
@functools.lru_cache(maxsize=8)
def get_kmeans_model(n_clusters):
    df = load_data()
    features = ['range_km', 'top_speed_kmh', 'battery_capacity_kWh', 'efficiency_wh_per_km']
    data = df[features].dropna()
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(scaled_data)
    return scaler, kmeans

def perform_clustering(df, n_clusters=4):
    """Perform K-means clustering on EV data"""
    features = ['range_km', 'top_speed_kmh', 'battery_capacity_kWh', 'efficiency_wh_per_km']
    data = df[features].dropna()
    scaler, kmeans = get_kmeans_model(n_clusters)
    scaled_data = scaler.transform(data)
    clusters = kmeans.predict(scaled_data)
    data = data.copy()
    data['cluster'] = clusters
    return data

@functools.lru_cache(maxsize=1)
def get_regression_model():
    df = load_data()
    model_data = df[['battery_capacity_kWh', 'efficiency_wh_per_km', 'range_km']].dropna()
    X = model_data[['battery_capacity_kWh', 'efficiency_wh_per_km']]
    y = model_data['range_km']
    model = LinearRegression()
    model.fit(X, y)
    return model, X, y

def predict_range(battery_capacity, efficiency):
    """Predict range based on battery capacity and efficiency"""
    model, X, y = get_regression_model()
    prediction = model.predict([[battery_capacity, efficiency]])
    r2_score = model.score(X, y)
    return prediction[0], r2_score

def _fmt_vehicle_row(row):
    return (f"{row['brand']} {row['model']}: {row['range_km']:.0f} km range, "
            f"{row['top_speed_kmh']:.0f} km/h top speed, {row['battery_capacity_kWh']:.1f} kWh battery, "
            f"{row['efficiency_wh_per_km']:.0f} Wh/km efficiency")


def _brand_summary(df, brand):
    subset = df[df['brand'].str.lower() == brand]
    if subset.empty:
        return None
    return (f"🏷️ **{subset['brand'].iloc[0]}** ({len(subset)} models):\n"
            f"• Avg range: {subset['range_km'].mean():.0f} km\n"
            f"• Avg top speed: {subset['top_speed_kmh'].mean():.0f} km/h\n"
            f"• Avg battery: {subset['battery_capacity_kWh'].mean():.1f} kWh\n"
            f"• Avg efficiency: {subset['efficiency_wh_per_km'].mean():.0f} Wh/km\n"
            f"• Best model by range: {subset.loc[subset['range_km'].idxmax(), 'model']} "
            f"({subset['range_km'].max():.0f} km)")


def analyze_query(query, df):
    """Answer natural-language questions by computing directly over the EV dataset."""
    if not query or not query.strip():
        return "🤖 Ask a question like *'which brand has the longest range?'* or *'compare Audi and BMW'*."

    q = query.lower()
    known_brands = sorted(df['brand'].dropna().str.lower().unique(), key=len, reverse=True)
    mentioned_brands = [b for b in known_brands if b in q]

    superlative = 'most' in q or 'longest' in q or 'highest' in q or 'best' in q or 'biggest' in q or 'largest' in q or 'fastest' in q or 'quickest' in q
    inferior = 'least' in q or 'shortest' in q or 'lowest' in q or 'worst' in q or 'smallest' in q or 'slowest' in q

    # Compare two specific brands
    if len(mentioned_brands) >= 2 and ('compare' in q or 'vs' in q or 'versus' in q or True):
        parts = [_brand_summary(df, b) for b in mentioned_brands[:2]]
        parts = [p for p in parts if p]
        if parts:
            return "🔍 **Brand Comparison**\n\n" + "\n\n".join(parts)

    # Single brand lookup
    if len(mentioned_brands) == 1:
        summary = _brand_summary(df, mentioned_brands[0])
        if summary:
            return summary

    # "Which brand has the longest/shortest range" -> brand-level ranking, not a single vehicle
    if 'brand' in q and 'range' in q and (superlative or inferior):
        brand_avg = df.groupby('brand')['range_km'].mean().sort_values(ascending=inferior)
        top = brand_avg.head(5)
        label = "Shortest" if inferior else "Longest"
        return (f"🔋 **{label} average range by brand**: **{top.index[0]}** leads at {top.iloc[0]:.0f} km.\n\n"
                + "\n".join([f"• {b}: {v:.0f} km" for b, v in top.items()]))

    # Range questions (single vehicle)
    if 'range' in q and ('longest' in q or 'highest' in q or 'best' in q or 'most' in q or 'max' in q):
        row = df.loc[df['range_km'].idxmax()]
        return f"🔋 **Longest range (single model)**: {_fmt_vehicle_row(row)}"
    if 'range' in q and inferior:
        row = df.loc[df['range_km'].idxmin()]
        return f"🔋 **Shortest range (single model)**: {_fmt_vehicle_row(row)}"

    # "Which brand is fastest / has the highest battery" -> brand-level ranking
    if 'brand' in q and ('speed' in q or 'fastest' in q) and (superlative or inferior):
        brand_avg = df.groupby('brand')['top_speed_kmh'].mean().sort_values(ascending=inferior)
        top = brand_avg.head(5)
        return (f"🏁 **{'Slowest' if inferior else 'Fastest'} average top speed by brand**: "
                f"**{top.index[0]}** leads at {top.iloc[0]:.0f} km/h.\n\n"
                + "\n".join([f"• {b}: {v:.0f} km/h" for b, v in top.items()]))
    if 'brand' in q and 'battery' in q and (superlative or inferior):
        brand_avg = df.groupby('brand')['battery_capacity_kWh'].mean().sort_values(ascending=inferior)
        top = brand_avg.head(5)
        return (f"⚡ **{'Smallest' if inferior else 'Largest'} average battery by brand**: "
                f"**{top.index[0]}** leads at {top.iloc[0]:.1f} kWh.\n\n"
                + "\n".join([f"• {b}: {v:.1f} kWh" for b, v in top.items()]))

    # Speed / performance questions (single vehicle)
    if ('speed' in q or 'fastest' in q or 'quickest' in q) and (superlative or 'speed' in q):
        row = df.loc[df['top_speed_kmh'].idxmax()]
        return f"🏁 **Fastest top speed (single model)**: {_fmt_vehicle_row(row)}"
    if 'acceleration' in q or '0-100' in q or '0 to 100' in q:
        row = df.loc[df['acceleration_0_100_s'].idxmin()]
        return f"🏎️ **Quickest 0-100 km/h**: {_fmt_vehicle_row(row)} ({row['acceleration_0_100_s']:.1f}s)"

    # Battery questions
    if 'battery' in q and superlative:
        row = df.loc[df['battery_capacity_kWh'].idxmax()]
        return f"⚡ **Largest battery**: {_fmt_vehicle_row(row)}"
    if 'battery' in q and inferior:
        row = df.loc[df['battery_capacity_kWh'].idxmin()]
        return f"⚡ **Smallest battery**: {_fmt_vehicle_row(row)}"

    # Efficiency questions (lower Wh/km = more efficient)
    if 'efficien' in q or 'consumption' in q:
        if inferior or 'worst' in q:
            row = df.loc[df['efficiency_wh_per_km'].idxmax()]
            label = "Least efficient"
        else:
            row = df.loc[df['efficiency_wh_per_km'].idxmin()]
            label = "Most efficient"
        return f"⚡ **{label}**: {_fmt_vehicle_row(row)}"

    # Range vs battery relationship
    if 'range' in q and 'battery' in q:
        corr = df['range_km'].corr(df['battery_capacity_kWh'])
        return (f"🔋 **Battery vs Range Analysis**: EVs with larger batteries generally have longer ranges. "
                f"The actual correlation across all {len(df)} models in this dataset is **{corr:.2f}**.")

    # Brand ranking / comparison
    if 'brand' in q and ('compare' in q or 'top' in q or 'best' in q or 'rank' in q):
        top_brands = df.groupby('brand')['range_km'].mean().sort_values(ascending=False).head(5)
        return "🏷️ **Top 5 Brands by Average Range**:\n" + "\n".join(
            [f"• {brand}: {range_val:.0f} km" for brand, range_val in top_brands.items()])

    # Segment questions
    if 'segment' in q:
        segment_stats = df.groupby('segment_category').agg({
            'range_km': 'mean',
            'battery_capacity_kWh': 'mean',
            'top_speed_kmh': 'mean'
        }).round(1).sort_values('range_km', ascending=False)
        return f"📦 **Segment Analysis** (avg range/battery/speed):\n{segment_stats.to_string()}"

    # Charging questions
    if 'charging' in q or 'charge' in q:
        row = df.loc[df['fast_charging_power_kw_dc'].idxmax()]
        return (f"🔌 **Charging**: Fast-charging power in this dataset ranges from "
                f"{df['fast_charging_power_kw_dc'].min():.0f} kW to {df['fast_charging_power_kw_dc'].max():.0f} kW "
                f"(avg {df['fast_charging_power_kw_dc'].mean():.0f} kW). "
                f"Fastest-charging model: {row['brand']} {row['model']} at {row['fast_charging_power_kw_dc']:.0f} kW.")

    # Market trend / general summary fallback, still grounded in real numbers
    total = len(df)
    return (f"🤖 **Market Snapshot** ({total} models, {df['brand'].nunique()} brands): "
            f"avg range {df['range_km'].mean():.0f} km, avg battery {df['battery_capacity_kWh'].mean():.1f} kWh, "
            f"avg top speed {df['top_speed_kmh'].mean():.0f} km/h. "
            f"Try asking about a specific brand, or 'longest range', 'fastest', 'most efficient', or 'compare X and Y'.")

def create_overview_dashboard():
    """Create overview dashboard with key metrics"""
    # Key metrics
    total_evs = len(df)
    avg_range = df['range_km'].mean()
    avg_battery = df['battery_capacity_kWh'].mean()
    avg_speed = df['top_speed_kmh'].mean()
    
    # Brand distribution
    brand_counts = df['brand'].value_counts().head(10)
    fig_brand = px.pie(
        values=brand_counts.values,
        names=brand_counts.index,
        title="🚗 Top 10 Brands Distribution",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig_brand.update_traces(textposition='inside', textinfo='percent+label')
    fig_brand.update_layout(
        title_font_size=20,
        title_font_color='#1e3c72',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    # Performance scatter
    fig_performance = px.scatter(
        df, x='range_km', y='top_speed_kmh',
        color='brand', size='battery_capacity_kWh',
        hover_data=['model', 'segment_category'],
        title="⚡ Performance Analysis: Range vs Speed",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig_performance.update_layout(
        title_font_size=20,
        title_font_color='#1e3c72',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    # Segment distribution
    segment_counts = df['segment_category'].value_counts().reset_index()
    segment_counts.columns = ['segment_category', 'count']
    fig_segment = px.bar(
        segment_counts,
        x='segment_category',
        y='count',
        title="📦 Segment Distribution",
        color_discrete_sequence=['#1e3c72']
    )
    fig_segment.update_layout(
        title_font_size=20,
        title_font_color='#1e3c72',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig_brand, fig_performance, fig_segment, f"""
    ## 📊 Market Overview
    
    **Key Metrics:**
    - 🚗 **Total Vehicles**: {total_evs:,}
    - 🔋 **Average Range**: {avg_range:.0f} km
    - ⚡ **Average Battery**: {avg_battery:.1f} kWh
    - 🏁 **Average Speed**: {avg_speed:.0f} km/h
    """

def create_ml_insights(n_clusters):
    """Create machine learning insights"""
    clustered_data = perform_clustering(df, n_clusters)
    
    # Clustering scatter plot
    fig_cluster = px.scatter(
        clustered_data, x='range_km', y='top_speed_kmh',
        color='cluster', size='battery_capacity_kWh',
        title=f"🤖 EV Clusters (K={n_clusters})",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig_cluster.update_layout(
        title_font_size=20,
        title_font_color='#1e3c72',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    # Cluster characteristics
    cluster_centers = clustered_data.groupby('cluster').mean()
    fig_radar = go.Figure()
    for cluster in range(n_clusters):
        center = cluster_centers.loc[cluster]
        fig_radar.add_trace(go.Scatterpolar(
            r=[center['range_km'], center['top_speed_kmh'], 
               center['battery_capacity_kWh'], center['efficiency_wh_per_km']],
            theta=['Range', 'Speed', 'Battery', 'Efficiency'],
            fill='toself',
            name=f'Cluster {cluster}',
            line_color=px.colors.qualitative.Set3[cluster]
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, cluster_centers.max().max()])),
        title="📊 Cluster Characteristics",
        title_font_size=20,
        title_font_color='#1e3c72',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    # Cluster insights
    insights = []
    for cluster in range(n_clusters):
        cluster_data = clustered_data[clustered_data['cluster'] == cluster]
        insights.append(f"""
        **Cluster {cluster}** ({len(cluster_data)} vehicles):
        - 🔋 Avg Range: {cluster_data['range_km'].mean():.0f} km
        - 🏁 Avg Speed: {cluster_data['top_speed_kmh'].mean():.0f} km/h
        - ⚡ Avg Battery: {cluster_data['battery_capacity_kWh'].mean():.1f} kWh
        """)
    
    return fig_cluster, fig_radar, "\n".join(insights)

def create_prediction_analysis(battery_capacity, efficiency):
    """Create prediction analysis"""
    predicted_range, r2_score = predict_range(battery_capacity, efficiency)
    
    # Create scatter plot with prediction
    model_data = df[['battery_capacity_kWh', 'efficiency_wh_per_km', 'range_km']].dropna()
    fig_scatter = px.scatter(
        model_data, x='battery_capacity_kWh', y='range_km',
        color='efficiency_wh_per_km',
        title=f"🔮 Range Prediction Model (R² = {r2_score:.3f})",
        color_continuous_scale='viridis'
    )
    fig_scatter.add_trace(go.Scatter(
        x=[battery_capacity], y=[predicted_range],
        mode='markers',
        marker=dict(size=20, color='red', symbol='star'),
        name='Prediction'
    ))
    fig_scatter.update_layout(
        title_font_size=20,
        title_font_color='#1e3c72',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    # Surface plot
    x_range = np.linspace(30, 150, 20)
    y_range = np.linspace(100, 300, 20)
    X, Y = np.meshgrid(x_range, y_range)
    model, X_train, y_train = get_regression_model()
    Z = model.predict(np.column_stack([X.ravel(), Y.ravel()])).reshape(X.shape)
    
    fig_surface = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale='viridis')])
    fig_surface.update_layout(
        title="🌐 Range Prediction Surface",
        scene=dict(
            xaxis_title="Battery Capacity (kWh)",
            yaxis_title="Efficiency (Wh/km)",
            zaxis_title="Predicted Range (km)"
        ),
        title_font_size=20,
        title_font_color='#1e3c72'
    )
    
    return f"{predicted_range:.0f} km", fig_scatter, fig_surface

def create_advanced_analytics():
    """Create advanced analytics visualizations"""
    # Correlation heatmap
    features = ['range_km', 'top_speed_kmh', 'battery_capacity_kWh', 
                'efficiency_wh_per_km', 'acceleration_0_100_s']
    corr_matrix = df[features].corr()
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="📊 Feature Correlation Matrix",
        color_continuous_scale="RdBu"
    )
    fig_corr.update_layout(
        title_font_size=20,
        title_font_color='#1e3c72',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    # Distribution plots
    fig_dist = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Range Distribution', 'Speed Distribution', 
                       'Battery Distribution', 'Efficiency Distribution')
    )
    fig_dist.add_trace(go.Histogram(x=df['range_km'].dropna(), name='Range'), row=1, col=1)
    fig_dist.add_trace(go.Histogram(x=df['top_speed_kmh'].dropna(), name='Speed'), row=1, col=2)
    fig_dist.add_trace(go.Histogram(x=df['battery_capacity_kWh'].dropna(), name='Battery'), row=2, col=1)
    fig_dist.add_trace(go.Histogram(x=df['efficiency_wh_per_km'].dropna(), name='Efficiency'), row=2, col=2)
    fig_dist.update_layout(
        height=600, 
        showlegend=False,
        title_font_size=20,
        title_font_color='#1e3c72',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    # Box plots
    fig_box = px.box(
        df, x='segment_category', y='range_km',
        title="📦 Range Distribution by Segment",
        color_discrete_sequence=['#1e3c72']
    )
    fig_box.update_layout(
        title_font_size=20,
        title_font_color='#1e3c72',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig_corr, fig_dist, fig_box

def create_interactive_explorer(brands, segments, min_range, min_speed):
    """Create interactive explorer visualizations"""
    filtered_df = df.copy()
    # Ensure brands and segments are lists for .isin
    if brands is not None and not isinstance(brands, list):
        brands = list(brands)
    if segments is not None and not isinstance(segments, list):
        segments = list(segments)
    if brands:
        filtered_df = filtered_df[filtered_df['brand'].isin(brands)]
    if segments:
        filtered_df = filtered_df[filtered_df['segment_category'].isin(segments)]
    filtered_df = filtered_df[
        (filtered_df['range_km'] >= min_range) &
        (filtered_df['top_speed_kmh'] >= min_speed)
    ]
    # Ensure filtered_df is a DataFrame
    if not isinstance(filtered_df, pd.DataFrame):
        filtered_df = pd.DataFrame(filtered_df)
    # 2D scatter
    fig_scatter = px.scatter(
        filtered_df, x='top_speed_kmh', y='range_km',
        color='brand', size='battery_capacity_kWh',
        hover_data=['model', 'segment_category'],
        title=f"🔍 Filtered Data ({len(filtered_df)} vehicles)",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig_scatter.update_layout(
        title_font_size=20,
        title_font_color='#1e3c72',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    # 3D scatter
    fig_3d = px.scatter_3d(
        filtered_df, x='battery_capacity_kWh', y='range_km', z='top_speed_kmh',
        color='segment_category', size='efficiency_wh_per_km',
        hover_data=['brand', 'model'],
        title="🌐 3D View: Battery vs Range vs Speed"
    )
    fig_3d.update_layout(
        title_font_size=20,
        title_font_color='#1e3c72'
    )
    # Summary
    summary = f"""
    ## 🔍 Data Explorer Results
    **Filtered Data Summary:**
    - 📊 **Total Vehicles**: {len(filtered_df)}
    - 🔋 **Average Range**: {filtered_df['range_km'].mean():.0f} km
    - ⚡ **Average Battery**: {filtered_df['battery_capacity_kWh'].mean():.1f} kWh
    - 🏁 **Average Speed**: {filtered_df['top_speed_kmh'].mean():.0f} km/h
    **Top Models by Range:**
    """
    if isinstance(filtered_df, pd.DataFrame) and not filtered_df.empty:
        top_models = filtered_df.nlargest(5, 'range_km')[['brand', 'model', 'range_km']]
        for _, row in top_models.iterrows():
            summary += f"- {row['brand']} {row['model']}: {row['range_km']:.0f} km\n"
    return fig_scatter, fig_3d, summary

# Create Gradio interface with modern design
def create_dashboard():
    """Create the main dashboard interface with modern design"""
    
    # Modern CSS styling
    custom_css = """
.gradio-container {
    background: linear-gradient(120deg, #f8fafc 0%, #e0e7ff 100%);
    min-height: 100vh;
    padding: 2.5rem 1rem 1.5rem 1rem;
    font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

.main-header {
    background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
    padding: 2.5rem 1rem;
    border-radius: 24px;
    margin-bottom: 2.5rem;
    box-shadow: 0 12px 40px rgba(30,60,114,0.10);
    text-align: center;
    color: white;
    border: 1px solid rgba(255,255,255,0.12);
}

.main-header h1 {
    font-size: 3.8rem;
    font-weight: 900;
    margin: 0;
    letter-spacing: -2px;
    text-shadow: 2px 2px 8px rgba(30,60,114,0.10);
    background: linear-gradient(45deg, #fff, #e0e7ff 80%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.main-header p {
    font-size: 1.35rem;
    margin: 1.2rem 0 0 0;
    opacity: 0.97;
    font-weight: 400;
    letter-spacing: 0.5px;
}

.metric-card {
    background: rgba(255, 255, 255, 0.99);
    border-radius: 22px;
    padding: 2.2rem 1.5rem;
    margin: 1.2rem 0;
    box-shadow: 0 8px 32px rgba(30,60,114,0.08);
    border: 1px solid rgba(30,60,114,0.07);
    transition: transform 0.3s, box-shadow 0.3s;
}
.metric-card:hover {
    transform: translateY(-6px) scale(1.01);
    box-shadow: 0 16px 48px rgba(30,60,114,0.13);
}

.gradio-tab-nav {
    background: rgba(30,60,114,0.07) !important;
    border-radius: 18px !important;
    padding: 0.7rem !important;
    margin: 1.2rem 0 !important;
    backdrop-filter: blur(8px) !important;
}
.gradio-tab-nav button {
    background: rgba(255,255,255,0.92) !important;
    border-radius: 14px !important;
    border: none !important;
    padding: 1.1rem 2.2rem !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    transition: all 0.3s !important;
    margin: 0 0.6rem !important;
    color: #1e3c72 !important;
    letter-spacing: 0.5px;
}
.gradio-tab-nav button:hover {
    background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%) !important;
    color: #fff !important;
    transform: translateY(-2px) scale(1.03) !important;
    box-shadow: 0 8px 24px rgba(30,60,114,0.13) !important;
}
.gradio-tab-nav button.selected {
    background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%) !important;
    color: #fff !important;
    box-shadow: 0 8px 24px rgba(30,60,114,0.13) !important;
}

.gradio-button {
    background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%) !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 1.1rem 2.2rem !important;
    font-weight: 700 !important;
    color: #fff !important;
    font-size: 1.1rem !important;
    transition: all 0.3s !important;
    box-shadow: 0 4px 15px rgba(30,60,114,0.10) !important;
}
.gradio-button:hover {
    transform: translateY(-2px) scale(1.03) !important;
    box-shadow: 0 12px 32px rgba(30,60,114,0.15) !important;
}

.gradio-input, .gradio-textbox, .gradio-slider, .gradio-dropdown {
    border-radius: 14px !important;
    border: 2px solid rgba(30,60,114,0.10) !important;
    background: rgba(255,255,255,0.98) !important;
    font-size: 1.08rem !important;
    padding: 0.7rem 1rem !important;
    backdrop-filter: blur(8px) !important;
}
.gradio-input:focus, .gradio-textbox:focus, .gradio-slider:focus, .gradio-dropdown:focus {
    border-color: #2a5298 !important;
    box-shadow: 0 0 0 3px rgba(30,60,114,0.10) !important;
}

.gradio-plot {
    background: rgba(255,255,255,0.99) !important;
    border-radius: 22px !important;
    padding: 1.7rem !important;
    box-shadow: 0 8px 32px rgba(30,60,114,0.08) !important;
    border: 1px solid rgba(30,60,114,0.07) !important;
    backdrop-filter: blur(8px) !important;
}

.gradio-markdown {
    background: rgba(255,255,255,0.99) !important;
    border-radius: 22px !important;
    padding: 2.2rem !important;
    box-shadow: 0 8px 32px rgba(30,60,114,0.08) !important;
    border: 1px solid rgba(30,60,114,0.07) !important;
    backdrop-filter: blur(8px) !important;
}

.gradio-row {
    margin: 1.2rem 0 !important;
}
.gradio-column {
    padding: 0.7rem !important;
}

/* Loading spinner */
.loading-spinner {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 60px;
}
.loading-spinner .spinner {
    border: 6px solid #e0e7ff;
    border-top: 6px solid #2a5298;
    border-radius: 50%;
    width: 40px;
    height: 40px;
    animation: spin 1s linear infinite;
}
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

@media (max-width: 900px) {
    .main-header h1 { font-size: 2.2rem; }
    .main-header { padding: 1.2rem 0.5rem; }
    .metric-card { padding: 1.2rem 0.7rem; }
}
"""
    
    with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
        # Beautiful Header
        gr.HTML("""
        <div class="main-header">
            <h1>🚗 EV Analytics Hub</h1>
            <p>AI-Powered Electric Vehicle Analysis • Natural Language Queries • Machine Learning Insights</p>
        </div>
        """)
        
        # Main tabs with modern design
        with gr.Tabs():
            # Overview Tab
            with gr.Tab("📊 Market Overview"):
                with gr.Row():
                    with gr.Column(scale=1):
                        overview_btn = gr.Button(
                            "🔄 Generate Overview", 
                            variant="primary", 
                            size="lg",
                            elem_classes=["gradio-button"]
                        )
                        overview_output = gr.HTML(elem_classes=["gradio-markdown"])
                
                with gr.Row():
                    with gr.Column(scale=1):
                        brand_plot = gr.Plot(label="Brand Distribution", elem_classes=["gradio-plot"])
                    with gr.Column(scale=1):
                        performance_plot = gr.Plot(label="Performance Analysis", elem_classes=["gradio-plot"])
                
                with gr.Row():
                    with gr.Column(scale=1):
                        segment_plot = gr.Plot(label="Segment Distribution", elem_classes=["gradio-plot"])
            
            # AI Insights Tab
            with gr.Tab("🤖 AI Assistant"):
                with gr.Row():
                    with gr.Column(scale=2):
                        query_input = gr.Textbox(
                            label="💬 Ask me anything about EVs:",
                            placeholder="e.g., 'Compare brands by range' or 'Analyze efficiency trends'",
                            lines=3,
                            elem_classes=["gradio-textbox"]
                        )
                        query_btn = gr.Button("🔍 Analyze", variant="primary", elem_classes=["gradio-button"])
                        query_output = gr.Markdown(elem_classes=["gradio-markdown"])
                
                with gr.Row():
                    with gr.Column(scale=1):
                        cluster_slider = gr.Slider(
                            minimum=2, maximum=6, value=4, step=1,
                            label="Number of Clusters",
                            elem_classes=["gradio-slider"]
                        )
                        ml_btn = gr.Button("🤖 Generate ML Insights", variant="primary", elem_classes=["gradio-button"])
                        ml_output = gr.Markdown(elem_classes=["gradio-markdown"])
                
                with gr.Row():
                    with gr.Column(scale=1):
                        cluster_plot = gr.Plot(label="Clustering Analysis", elem_classes=["gradio-plot"])
                    with gr.Column(scale=1):
                        radar_plot = gr.Plot(label="Cluster Characteristics", elem_classes=["gradio-plot"])
            
            # Predictive Analytics Tab
            with gr.Tab("🔮 Predictions"):
                with gr.Row():
                    with gr.Column(scale=1):
                        battery_input = gr.Slider(
                            minimum=30, maximum=150, value=75, step=5,
                            label="🔋 Battery Capacity (kWh)",
                            elem_classes=["gradio-slider"]
                        )
                        efficiency_input = gr.Slider(
                            minimum=100, maximum=300, value=200, step=10,
                            label="⚡ Efficiency (Wh/km)",
                            elem_classes=["gradio-slider"]
                        )
                        predict_btn = gr.Button("🔮 Predict Range", variant="primary", elem_classes=["gradio-button"])
                        predict_output = gr.Markdown(elem_classes=["gradio-markdown"])
                
                with gr.Row():
                    with gr.Column(scale=1):
                        prediction_plot = gr.Plot(label="Prediction Model", elem_classes=["gradio-plot"])
                    with gr.Column(scale=1):
                        surface_plot = gr.Plot(label="Prediction Surface", elem_classes=["gradio-plot"])
            
            # Advanced Analytics Tab
            with gr.Tab("📈 Analytics"):
                with gr.Row():
                    analytics_btn = gr.Button("📊 Generate Analytics", variant="primary", size="lg", elem_classes=["gradio-button"])
                
                with gr.Row():
                    with gr.Column(scale=1):
                        corr_plot = gr.Plot(label="Correlation Matrix", elem_classes=["gradio-plot"])
                
                with gr.Row():
                    with gr.Column(scale=1):
                        dist_plot = gr.Plot(label="Feature Distributions", elem_classes=["gradio-plot"])
                    with gr.Column(scale=1):
                        box_plot = gr.Plot(label="Segment Analysis", elem_classes=["gradio-plot"])
            
            # Interactive Explorer Tab
            with gr.Tab("🔍 Explorer"):
                with gr.Row():
                    with gr.Column(scale=1):
                        brand_filter = gr.Dropdown(
                            choices=sorted(df['brand'].unique()),
                            label="🏷️ Select Brands",
                            multiselect=True,
                            elem_classes=["gradio-dropdown"]
                        )
                        segment_filter = gr.Dropdown(
                            choices=sorted(df['segment_category'].unique()),
                            label="📦 Select Segments",
                            multiselect=True,
                            elem_classes=["gradio-dropdown"]
                        )
                    with gr.Column(scale=1):
                        range_filter = gr.Slider(
                            minimum=0, maximum=int(df['range_km'].max()), value=0,
                            label="🔋 Min Range (km)",
                            elem_classes=["gradio-slider"]
                        )
                        speed_filter = gr.Slider(
                            minimum=0, maximum=int(df['top_speed_kmh'].max()), value=0,
                            label="🏁 Min Speed (km/h)",
                            elem_classes=["gradio-slider"]
                        )
                
                with gr.Row():
                    explore_btn = gr.Button("🔍 Explore Data", variant="primary", elem_classes=["gradio-button"])
                    explore_output = gr.Markdown(elem_classes=["gradio-markdown"])
                
                with gr.Row():
                    with gr.Column(scale=1):
                        scatter_plot = gr.Plot(label="Interactive Scatter", elem_classes=["gradio-plot"])
                    with gr.Column(scale=1):
                        plot_3d = gr.Plot(label="3D Analysis", elem_classes=["gradio-plot"])
        
        # Event handlers
        overview_btn.click(
            fn=create_overview_dashboard,
            outputs=[brand_plot, performance_plot, segment_plot, overview_output]
        )
        
        query_btn.click(
            fn=lambda q: analyze_query(q, df),
            inputs=[query_input],
            outputs=[query_output]
        )
        
        ml_btn.click(
            fn=create_ml_insights,
            inputs=[cluster_slider],
            outputs=[cluster_plot, radar_plot, ml_output]
        )
        
        predict_btn.click(
            fn=create_prediction_analysis,
            inputs=[battery_input, efficiency_input],
            outputs=[predict_output, prediction_plot, surface_plot]
        )
        
        analytics_btn.click(
            fn=create_advanced_analytics,
            outputs=[corr_plot, dist_plot, box_plot]
        )
        
        explore_btn.click(
            fn=create_interactive_explorer,
            inputs=[brand_filter, segment_filter, range_filter, speed_filter],
            outputs=[scatter_plot, plot_3d, explore_output]
        )
    
    return demo

# Launch the dashboard
if __name__ == "__main__":
    print("🚗 Starting Modern AI-Powered Electric Vehicle Analysis Dashboard (Gradio)...")
    print("🤖 Loading AI models and machine learning insights...")
    print("🌐 Dashboard will open in your browser at http://localhost:<PORT>")
    print("⏹️  Press Ctrl+C to stop the dashboard")
    print("-" * 70)
    # Parse command line arguments for port
    parser = argparse.ArgumentParser(description="Run Gradio EV Dashboard")
    parser.add_argument('--port', type=int, default=7861, help='Port to run the dashboard on')
    args = parser.parse_args()
    # Check dependencies
    print("🔍 Checking dependencies...")
    try:
        import gradio
        import plotly
        import sklearn
        print("✅ All dependencies are available")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("💡 Install dependencies with:")
        print("   pip install -r gradio_requirements.txt")
        exit(1)
    print("🚀 Launching modern AI dashboard...")
    # Create and launch the dashboard
    demo = create_dashboard()
    demo.launch(
        server_name="localhost",
        server_port=args.port,
        share=False,
        show_error=True,
        quiet=False
    ) 