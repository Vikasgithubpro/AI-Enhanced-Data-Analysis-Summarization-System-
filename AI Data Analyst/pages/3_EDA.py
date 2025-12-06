import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils import (
    df_shape, df_missing, df_duplicates, df_summary,
    value_counts, corr_matrix, plot_histogram, plot_bar, plot_trend,
    enhanced_data_quality_report, detect_anomalies, forecast_trend
)

st.set_page_config(page_title="🧪 Enhanced EDA", layout="wide")
st.title("🧪 Enhanced Exploratory Data Analysis")

if "df" not in st.session_state:
    st.warning("⚠️ Please upload a dataset first (📥 Upload).")
    st.stop()

df = st.session_state.df

# Enhanced Data Quality Report
st.markdown("## 📋 Comprehensive Data Quality Report")
if st.button("🔄 Generate Quality Report"):
    with st.spinner("Analyzing data quality..."):
        quality_report = enhanced_data_quality_report(df)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", quality_report['overview']['rows'])
        with col2:
            st.metric("Columns", quality_report['overview']['columns'])
        with col3:
            st.metric("Data Completeness", f"{quality_report['overview']['completeness']:.1f}%")
        with col4:
            st.metric("Missing Cells", quality_report['overview']['missing_cells'])
        
        # Quality Issues
        if quality_report['quality_issues']:
            st.warning("🚨 Data Quality Issues Found:")
            for issue in quality_report['quality_issues']:
                st.write(f"• {issue}")
        else:
            st.success("✅ No major data quality issues detected!")
        
        # Column Analysis
        st.markdown("### 📊 Column Analysis")
        for col, analysis in quality_report['column_analysis'].items():
            with st.expander(f"🔍 {col} ({analysis['dtype']})"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Missing", f"{analysis['missing']} ({analysis['missing_percent']:.1f}%)")
                with col2:
                    st.metric("Unique Values", analysis['unique_values'])
                with col3:
                    if analysis['sample_values']:
                        st.write("Sample:", analysis['sample_values'][:3])

# Quick Facts
st.markdown("## 📈 Dataset Overview")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Rows", df.shape[0])
with col2:
    st.metric("Columns", df.shape[1])
with col3:
    st.metric("Duplicates", int(df_duplicates(df)))
with col4:
    st.metric("Missing Cells", int(df.isna().sum().sum()))

# Summary Statistics
st.markdown("### 📊 Summary Statistics")
st.dataframe(df_summary(df), use_container_width=True)

# Missing Values Analysis
st.markdown("### ❓ Missing Values Analysis")
missing_df = df_missing(df)
fig_missing = px.bar(missing_df, x='Column', y='Percentage', 
                     title='Missing Values Percentage by Column')
st.plotly_chart(fig_missing, use_container_width=True)
st.dataframe(missing_df, use_container_width=True)

# Enhanced Correlation Analysis
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
if len(numeric_cols) > 1:
    st.markdown("### 🔗 Advanced Correlation Analysis")
    
    corr = corr_matrix(df)
    fig_corr = px.imshow(corr, title="Correlation Heatmap", 
                        aspect="auto", color_continuous_scale="RdBu")
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Find strong correlations
    strong_corrs = []
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            if abs(corr.iloc[i, j]) > 0.7:
                strong_corrs.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))
    
    if strong_corrs:
        st.markdown("#### 💪 Strong Correlations (|r| > 0.7)")
        for col1, col2, corr_val in strong_corrs:
            st.write(f"• **{col1}** ↔ **{col2}**: {corr_val:.3f}")

# Enhanced Categorical Analysis
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
if categorical_cols:
    st.markdown("### 🏷️ Enhanced Categorical Analysis")
    col = st.selectbox("Select categorical column", categorical_cols, key="vc_col")
    n = st.slider("Top N categories", 5, 50, 10, key="vc_n")
    
    if st.button("📊 Analyze Categorical Distribution"):
        value_counts_df = value_counts(df, col, n=n).reset_index()
        value_counts_df.columns = [col, 'Count']
        
        fig = px.pie(value_counts_df, values='Count', names=col, 
                    title=f'Distribution of {col}')
        st.plotly_chart(fig, use_container_width=True)
        
        fig_bar = px.bar(value_counts_df, x=col, y='Count',
                        title=f'Top {n} Categories in {col}')
        st.plotly_chart(fig_bar, use_container_width=True)

# Enhanced Numeric Analysis
if numeric_cols:
    st.markdown("### 📈 Enhanced Numeric Analysis")
    col = st.selectbox("Select numeric column", numeric_cols, key="hist_col")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 Show Distribution"):
            fig_hist = px.histogram(df, x=col, nbins=30, 
                                  title=f'Distribution of {col}')
            st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        if st.button("📈 Show Box Plot"):
            fig_box = px.box(df, y=col, title=f'Box Plot of {col}')
            st.plotly_chart(fig_box, use_container_width=True)

# Anomaly Detection
if numeric_cols:
    st.markdown("### 🚨 AI-Powered Anomaly Detection")
    if st.button("🔍 Detect Anomalies"):
        with st.spinner("Detecting anomalies using AI..."):
            anomalies = detect_anomalies(df, numeric_cols)
            if not anomalies.empty:
                st.warning(f"🚨 Found {len(anomalies)} potential anomalies!")
                st.dataframe(anomalies, use_container_width=True)
                
                # Plot anomalies
                if len(numeric_cols) >= 2:
                    fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], 
                                   title="Anomaly Detection")
                    # Highlight anomalies
                    anomaly_indices = anomalies.index
                    fig.add_trace(go.Scatter(
                        x=df.loc[anomaly_indices, numeric_cols[0]],
                        y=df.loc[anomaly_indices, numeric_cols[1]],
                        mode='markers',
                        marker=dict(color='red', size=10, symbol='x'),
                        name='Anomalies'
                    ))
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("✅ No anomalies detected or insufficient data for analysis.")

# Time Series Analysis
date_cols = [c for c in df.columns if any(pattern in c.lower() for pattern in ['date', 'time', 'year'])]
if date_cols and numeric_cols:
    st.markdown("### ⏳ Advanced Time Series Analysis")
    date_col = st.selectbox("Select date column", date_cols, key="date_col")
    value_col = st.selectbox("Select value column", numeric_cols, key="value_col")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📈 Show Time Trend"):
            if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            
            time_series = df.groupby(date_col)[value_col].mean().reset_index()
            fig = px.line(time_series, x=date_col, y=value_col,
                         title=f'{value_col} Trend Over Time')
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if st.button("🔮 Forecast Trend"):
            forecast_fig = forecast_trend(df, date_col, value_col)
            st.plotly_chart(forecast_fig, use_container_width=True)