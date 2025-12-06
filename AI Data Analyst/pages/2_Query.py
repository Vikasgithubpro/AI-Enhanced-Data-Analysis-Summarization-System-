import streamlit as st
import re
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from utils import GroqChatWrapper, nl_to_sql, connect_duckdb_from_csv, generate_ai_insights

st.set_page_config(page_title="🔍 AI-Powered Query & Analysis", layout="wide")
st.title("🔍 AI-Powered Query & Analysis")

if "df" not in st.session_state or "csv_path" not in st.session_state:
    st.warning("⚠️ Please upload a dataset first (📥 Upload).")
    st.stop()

if "con" not in st.session_state:
    st.session_state.con = connect_duckdb_from_csv(st.session_state.csv_path)

con = st.session_state.con
columns = st.session_state.columns
df = st.session_state.df

# AI Analysis Section
st.markdown("### 🤖 AI-Powered Data Analysis")
if st.button("🎯 Generate AI Insights", key="ai_insights"):
    if "groq_key" not in st.session_state:
        st.error("Groq API key missing. Set it in 📥 Upload page.")
    else:
        with st.spinner("AI is analyzing your data..."):
            groq_model = GroqChatWrapper(model="llama-3.3-70b-versatile", api_key=st.session_state.groq_key)
            insights = generate_ai_insights(groq_model, df)
            st.markdown("#### 📋 AI Analysis Report")
            st.success(insights)

st.markdown("---")

# Query Section
st.markdown("### 💬 Natural Language Query")
col1, col2 = st.columns([3, 1])
with col1:
    user_query = st.text_area("Ask a question about your data:", height=100, 
                             placeholder="E.g.: Show me the top 10 products by sales, What's the average age by category?")
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    run_query = st.button("🚀 Run Analysis", use_container_width=True)

st.caption("💡 Examples: *Top 10 customers by revenue*, *Average sales by month*, *Correlation between age and spending*")

# Predefined analysis templates
st.markdown("### 📊 Quick Analysis Templates")
template_col1, template_col2, template_col3 = st.columns(3)

with template_col1:
    if st.button("📈 Summary Statistics", use_container_width=True):
        user_query = "Show summary statistics for all numeric columns"

with template_col2:
    if st.button("🔍 Find Anomalies", use_container_width=True):
        user_query = "Find unusual patterns or outliers in the data"

with template_col3:
    if st.button("📅 Time Trends", use_container_width=True):
        user_query = "Show trends over time for date columns"

if run_query and user_query.strip():
    try:
        q = user_query.strip()
        is_sql = bool(re.match(r"^(select|with|create|describe|pragma|explain)\b", q, flags=re.IGNORECASE))

        if is_sql:
            sql = q
        else:
            if "groq_key" not in st.session_state:
                st.error("Groq API key missing. Set it in 📥 Upload page.")
                st.stop()
            groq_model = GroqChatWrapper(model="llama-3.3-70b-versatile", api_key=st.session_state.groq_key)
            sql = nl_to_sql(groq_model, q, columns, df.head(3))

        with st.expander("📜 Executed SQL", expanded=True):
            st.code(sql, language="sql")

        # Fix regex for DuckDB
        sql = sql.replace("REGEXP", "regexp_matches")
        
        with st.spinner("Executing query..."):
            result_df = con.execute(sql).fetchdf()

        st.success(f"✅ Query executed successfully. Found {len(result_df)} records.")

        # Results Section
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Data", "📈 Visualization", "📋 Summary", "💾 Export"])

        with tab1:
            st.dataframe(result_df, use_container_width=True)

        with tab2:
            if not result_df.empty:
                st.markdown("#### 📊 Auto Visualization")
                try:
                    if result_df.shape[1] == 2:
                        col1, col2 = result_df.columns
                        
                        # Determine best chart type
                        if (pd.api.types.is_numeric_dtype(result_df[col2]) and 
                            pd.api.types.is_object_dtype(result_df[col1])):
                            # Bar chart for categorical vs numeric
                            fig = px.bar(result_df, x=col1, y=col2, 
                                       title=f"{col2} by {col1}")
                            st.plotly_chart(fig, use_container_width=True)
                            
                        elif (pd.api.types.is_datetime64_any_dtype(result_df[col1]) and 
                              pd.api.types.is_numeric_dtype(result_df[col2])):
                            # Line chart for time series
                            fig = px.line(result_df, x=col1, y=col2, 
                                        title=f"Trend of {col2} over {col1}")
                            st.plotly_chart(fig, use_container_width=True)
                            
                        elif (pd.api.types.is_numeric_dtype(result_df[col1]) and 
                              pd.api.types.is_numeric_dtype(result_df[col2])):
                            # Scatter plot for numeric vs numeric
                            fig = px.scatter(result_df, x=col1, y=col2, 
                                           title=f"{col2} vs {col1}")
                            st.plotly_chart(fig, use_container_width=True)

                    elif all(pd.api.types.is_numeric_dtype(result_df[col]) for col in result_df.columns):
                        # Correlation heatmap for multiple numeric columns
                        corr = result_df.corr()
                        fig = px.imshow(corr, title="Correlation Matrix", 
                                      aspect="auto", color_continuous_scale="RdBu")
                        st.plotly_chart(fig, use_container_width=True)

                except Exception as viz_err:
                    st.warning(f"⚠️ Could not generate visualization: {viz_err}")

        with tab3:
            if not result_df.empty:
                st.markdown("#### 📋 Result Summary")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Rows", len(result_df))
                with col2:
                    st.metric("Total Columns", len(result_df.columns))
                with col3:
                    numeric_cols = result_df.select_dtypes(include=['number']).columns
                    st.metric("Numeric Columns", len(numeric_cols))
                with col4:
                    st.metric("Memory Usage", f"{result_df.memory_usage(deep=True).sum() / 1024:.1f} KB")
                
                # Column info
                st.markdown("##### Column Information")
                for col in result_df.columns:
                    with st.expander(f"📌 {col} ({result_df[col].dtype})"):
                        st.write(f"Missing values: {result_df[col].isna().sum()} ({result_df[col].isna().sum()/len(result_df)*100:.1f}%)")
                        if pd.api.types.is_numeric_dtype(result_df[col]):
                            st.write(f"Range: {result_df[col].min():.2f} to {result_df[col].max():.2f}")
                            st.write(f"Mean: {result_df[col].mean():.2f}")
                        else:
                            st.write(f"Unique values: {result_df[col].nunique()}")
                            st.write(f"Most common: {result_df[col].mode().iloc[0] if not result_df[col].mode().empty else 'N/A'}")

        with tab4:
            st.markdown("#### 💾 Export Results")
            export_col1, export_col2, export_col3 = st.columns(3)
            
            with export_col1:
                csv_data = result_df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download CSV", csv_data, "ai_analysis_results.csv", "text/csv")
            
            with export_col2:
                json_data = result_df.to_json(orient="records", indent=2)
                st.download_button("📥 Download JSON", json_data, "ai_analysis_results.json", "application/json")
            
            with export_col3:
                # Generate summary report
                summary = f"""
AI Analysis Report
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Original Query: {user_query}
SQL Executed: {sql}
Results: {len(result_df)} rows, {len(result_df.columns)} columns

Column Summary:
"""
                for col in result_df.columns:
                    summary += f"- {col}: {result_df[col].dtype}\n"
                
                st.download_button("📥 Download Summary", summary, "analysis_summary.txt", "text/plain")

    except Exception as e:
        st.error(f"❌ Error executing query: {e}")
        st.info("💡 Tip: Try rephrasing your question or check if the column names exist in your dataset.")