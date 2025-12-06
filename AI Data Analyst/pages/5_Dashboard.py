import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import enhanced_data_quality_report, detect_anomalies, forecast_trend

st.set_page_config(page_title="📊 AI Dashboard", layout="wide")
st.title("📊 AI-Enhanced Analytics Dashboard")

if "df" not in st.session_state:
    st.warning("⚠️ Please upload a dataset first (📥 Upload).")
    st.stop()

df = st.session_state.df

# Dashboard Overview
st.markdown("## 📈 Real-time Analytics Dashboard")

# Key Metrics Row
st.markdown("### 🎯 Key Metrics")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Records", f"{len(df):,}")
with col2:
    completeness = (1 - df.isna().sum().sum() / (len(df) * len(df.columns))) * 100
    st.metric("Data Quality", f"{completeness:.1f}%")
with col3:
    st.metric("Numeric Features", len(df.select_dtypes(include=['number']).columns))
with col4:
    st.metric("Categorical Features", len(df.select_dtypes(include=['object']).columns))
with col5:
    st.metric("Duplicates", df.duplicated().sum())

# Data Distribution Overview
st.markdown("### 📊 Data Distribution Overview")

# Column type distribution - FIXED: Convert dtypes to strings for Plotly
col_types_data = {
    'Type': ['Numeric', 'Categorical', 'Date', 'Other'],
    'Count': [
        len(df.select_dtypes(include=['number']).columns),
        len(df.select_dtypes(include=['object']).columns),
        len([c for c in df.columns if 'date' in c.lower()]),
        len(df.columns) - len(df.select_dtypes(include=['number', 'object']).columns)
    ]
}

col_types = pd.DataFrame(col_types_data)

# Only create pie chart if we have data
if col_types['Count'].sum() > 0:
    fig_types = px.pie(col_types, values='Count', names='Type', 
                       title='Column Type Distribution')
    st.plotly_chart(fig_types, use_container_width=True)
else:
    st.info("No data available for column type distribution")

# Quick Analysis Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📋 Data Health", "📈 Trends", "🚨 Anomalies", "🔮 Predictions"])

with tab1:
    st.markdown("#### 📋 Data Health Monitor")
    
    # Missing values by column - FIXED: Ensure data is serializable
    missing_data = df.isna().sum().reset_index()
    missing_data.columns = ['Column', 'Missing_Count']
    missing_data = missing_data[missing_data['Missing_Count'] > 0]
    
    if not missing_data.empty:
        # Convert columns to strings for Plotly
        missing_data['Column'] = missing_data['Column'].astype(str)
        fig_missing = px.bar(missing_data, x='Column', y='Missing_Count',
                           title='Missing Values by Column')
        st.plotly_chart(fig_missing, use_container_width=True)
    else:
        st.success("✅ No missing values found in the dataset!")
    
    # Data types overview - FIXED: Convert dtypes to strings
    dtype_counts = df.dtypes.astype(str).value_counts().reset_index()
    dtype_counts.columns = ['Data Type', 'Count']
    
    if not dtype_counts.empty:
        fig_dtypes = px.bar(dtype_counts, x='Data Type', y='Count',
                           title='Data Types Distribution')
        st.plotly_chart(fig_dtypes, use_container_width=True)
    else:
        st.info("No data types information available")

with tab2:
    st.markdown("#### 📈 Trend Analysis")
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    date_cols = [c for c in df.columns if any(pattern in c.lower() for pattern in ['date', 'time'])]
    
    if numeric_cols and date_cols:
        col1, col2 = st.columns(2)
        
        with col1:
            selected_date = st.selectbox("Select Date Column", date_cols, key='trend_date')
        with col2:
            selected_value = st.selectbox("Select Value Column", numeric_cols, key='trend_value')
        
        if selected_date and selected_value:
            try:
                if not pd.api.types.is_datetime64_any_dtype(df[selected_date]):
                    df[selected_date] = pd.to_datetime(df[selected_date], errors='coerce')
                
                # Remove rows where date or value is NaN
                temp_df = df[[selected_date, selected_value]].dropna()
                
                if not temp_df.empty:
                    time_series = temp_df.groupby(selected_date)[selected_value].mean().reset_index()
                    fig_trend = px.line(time_series, x=selected_date, y=selected_value,
                                      title=f'{selected_value} Trend Over Time')
                    st.plotly_chart(fig_trend, use_container_width=True)
                else:
                    st.warning("No valid data available for trend analysis after cleaning")
                
            except Exception as e:
                st.error(f"Error generating trend: {e}")
    else:
        st.info("ℹ️ Need both date and numeric columns for trend analysis")

with tab3:
    st.markdown("#### 🚨 Anomaly Detection")
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if len(numeric_cols) >= 2:
        if st.button("🔍 Run Anomaly Detection", key='anomaly_detection'):
            with st.spinner("Detecting anomalies..."):
                anomalies = detect_anomalies(df, numeric_cols)
                
                if not anomalies.empty:
                    st.warning(f"🚨 Found {len(anomalies)} potential anomalies!")
                    
                    # Display anomalies in a table
                    st.dataframe(anomalies.head(10), use_container_width=True)
                    
                    # Visualize anomalies - FIXED: Ensure we have valid data
                    if len(numeric_cols) >= 2:
                        # Use the first two numeric columns for visualization
                        col1, col2 = numeric_cols[0], numeric_cols[1]
                        
                        # Create a clean dataframe for plotting
                        plot_df = df[[col1, col2]].dropna()
                        anomaly_indices = anomalies.index.intersection(plot_df.index)
                        
                        if not plot_df.empty:
                            fig = px.scatter(plot_df, x=col1, y=col2, 
                                           title="Anomaly Detection Scatter Plot")
                            
                            # Highlight anomalies if we have any in the current view
                            if not anomaly_indices.empty:
                                fig.add_trace(go.Scatter(
                                    x=plot_df.loc[anomaly_indices, col1],
                                    y=plot_df.loc[anomaly_indices, col2],
                                    mode='markers',
                                    marker=dict(color='red', size=8, symbol='x', line=dict(width=2)),
                                    name='Detected Anomalies'
                                ))
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No valid data available for anomaly visualization")
                else:
                    st.success("✅ No anomalies detected!")
    else:
        st.info("ℹ️ Need at least 2 numeric columns for anomaly detection")

with tab4:
    st.markdown("#### 🔮 Predictive Insights")
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    date_cols = [c for c in df.columns if any(pattern in c.lower() for pattern in ['date', 'time'])]
    
    if numeric_cols and date_cols:
        st.info("📈 Trend forecasting available for time series data")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            forecast_date = st.selectbox("Date Column", date_cols, key='forecast_date')
        with col2:
            forecast_value = st.selectbox("Value Column", numeric_cols, key='forecast_value')
        with col3:
            forecast_periods = st.slider("Forecast Periods", 7, 90, 30, key='forecast_periods')
        
        if st.button("📊 Generate Forecast", key='generate_forecast'):
            with st.spinner("Generating forecast..."):
                forecast_fig = forecast_trend(df, forecast_date, forecast_value, forecast_periods)
                if forecast_fig and len(forecast_fig.data) > 0:
                    st.plotly_chart(forecast_fig, use_container_width=True)
                else:
                    st.warning("Could not generate forecast with the selected data")
    else:
        st.info("ℹ️ Need both date and numeric columns for forecasting")

# Real-time Data Summary
st.markdown("---")
st.markdown("### 📋 Real-time Data Summary")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 📊 Numeric Columns Summary")
    numeric_summary = df.describe()
    # Ensure all data is serializable
    st.dataframe(numeric_summary, use_container_width=True)

with col2:
    st.markdown("#### 📝 Categorical Columns Summary")
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        cat_summary_data = []
        for col in categorical_cols:
            unique_count = df[col].nunique()
            most_common = "N/A"
            if not df[col].mode().empty:
                most_common = str(df[col].mode().iloc[0])
            
            cat_summary_data.append({
                'Column': col,
                'Unique Values': unique_count,
                'Most Common': most_common
            })
        
        cat_summary = pd.DataFrame(cat_summary_data)
        st.dataframe(cat_summary, use_container_width=True)
    else:
        st.info("No categorical columns found")

# Data Quality Insights
st.markdown("---")
st.markdown("### 🔍 Data Quality Insights")

if st.button("🔄 Run Data Quality Analysis", key='quality_analysis'):
    with st.spinner("Analyzing data quality..."):
        quality_report = enhanced_data_quality_report(df)
        
        # Display key quality metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Overall Completeness", f"{quality_report['overview']['completeness']:.1f}%")
        with col2:
            st.metric("Total Issues", len(quality_report['quality_issues']))
        with col3:
            quality_score = max(0, 100 - len(quality_report['quality_issues']) * 5)
            st.metric("Quality Score", f"{quality_score:.1f}%")
        
        # Show quality issues
        if quality_report['quality_issues']:
            st.warning("#### 🚨 Data Quality Issues")
            for issue in quality_report['quality_issues'][:5]:  # Show top 5 issues
                st.write(f"• {issue}")
            
            if len(quality_report['quality_issues']) > 5:
                st.info(f"... and {len(quality_report['quality_issues']) - 5} more issues")
        else:
            st.success("#### ✅ Excellent data quality!")

# Auto-refresh option
st.markdown("---")
if st.button("🔄 Refresh Dashboard", key='refresh_dashboard'):
    st.rerun()

st.caption("Dashboard updates automatically with new data uploads")