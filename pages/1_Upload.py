import streamlit as st
import pandas as pd
import numpy as np
from utils import preprocess_and_save, connect_duckdb_from_csv, enhanced_data_quality_report, generate_ai_insights, GroqChatWrapper

st.set_page_config(page_title="📥 AI Data Upload & Preparation", layout="wide")
st.title("📥 AI-Powered Data Upload & Preparation")

# Sidebar for API Key and AI Settings
with st.sidebar:
    st.header("🔑 AI Configuration")
    
    groq_key = st.text_input("Enter your Groq API key:", type="password", 
                           help="Required for AI-powered insights and natural language queries")
    if groq_key:
        st.session_state.groq_key = groq_key
        st.success("✅ Groq API key saved!")
    else:
        st.warning("⚠️ Enter Groq API key to enable AI features")
    
    st.markdown("---")
    st.header("⚙️ Analysis Settings")
    
    sample_size = st.slider("Sample Size for AI Analysis", 100, 5000, 1000,
                          help="Larger samples provide better insights but take longer")
    st.session_state.sample_size = sample_size
    
    auto_analyze = st.checkbox("Auto-analyze on upload", value=True,
                             help="Automatically generate AI insights when data is uploaded")

# Main upload section
st.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white;">
<h3 style="color: white; margin: 0;">🚀 Upload Your Dataset</h3>
<p style="margin: 10px 0 0 0;">Upload CSV or Excel files for AI-powered analysis and insights</p>
</div>
""", unsafe_allow_html=True)

# File upload with enhanced options
uploaded_file = st.file_uploader(
    "📂 Choose a CSV or Excel file", 
    type=["csv", "xlsx"],
    help="Supported formats: CSV, Excel (.xlsx)"
)

# Process uploaded file
if uploaded_file is not None:
    # Show file info
    file_details = {
        "Filename": uploaded_file.name,
        "File size": f"{uploaded_file.size / 1024:.2f} KB",
        "File type": uploaded_file.type
    }
    
    st.markdown("### 📋 File Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("File Name", uploaded_file.name)
    with col2:
        st.metric("File Size", f"{uploaded_file.size / 1024:.2f} KB")
    with col3:
        st.metric("File Type", uploaded_file.type.split('/')[-1].upper())
    
    # Process the file
    with st.spinner("🔄 Processing your data with AI-enhanced cleaning..."):
        path, cols, df = preprocess_and_save(uploaded_file)
        
        if path and cols and df is not None:
            # Store in session state
            st.session_state.csv_path = path
            st.session_state.columns = cols
            st.session_state.df = df
            st.session_state.con = connect_duckdb_from_csv(path)
            
            st.success("✅ File processed successfully with AI-enhanced cleaning!")
            
            # Enhanced Data Overview
            st.markdown("## 🎯 Data Overview")
            
            # Quick stats in columns
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Rows", f"{len(df):,}")
            with col2:
                st.metric("Total Columns", len(df.columns))
            with col3:
                numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
                st.metric("Numeric Columns", numeric_cols)
            with col4:
                categorical_cols = len(df.select_dtypes(include=['object']).columns)
                st.metric("Categorical Columns", categorical_cols)
            
            # Data Preview Tabs
            st.markdown("### 👀 Data Preview")
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Cleaned Data", "🔍 Data Types", "📈 Sample Data", "🗃️ Database View"])
            
            with tab1:
                st.markdown("#### Cleaned Dataset (First 20 rows)")
                st.dataframe(df.head(20), use_container_width=True)
                st.caption(f"Columns: {', '.join(cols)}")
            
            with tab2:
                st.markdown("#### Column Data Types")
                dtype_info = pd.DataFrame({
                    'Column': df.columns,
                    'Data Type': df.dtypes,
                    'Non-Null Count': df.count(),
                    'Null Count': df.isnull().sum(),
                    'Null Percentage': (df.isnull().sum() / len(df) * 100).round(2)
                })
                st.dataframe(dtype_info, use_container_width=True)
                
                # Data type distribution
                dtype_counts = df.dtypes.value_counts()
                if not dtype_counts.empty:
                    st.markdown("##### Data Type Distribution")
                    for dtype, count in dtype_counts.items():
                        st.write(f"- **{dtype}**: {count} columns")
            
            with tab3:
                st.markdown("#### Random Sample (10 rows)")
                sample_df = df.sample(min(10, len(df)))
                st.dataframe(sample_df, use_container_width=True)
            
            with tab4:
                st.markdown("#### DuckDB Database Preview")
                try:
                    preview_df = st.session_state.con.execute("SELECT * FROM uploaded_data LIMIT 10").fetchdf()
                    st.dataframe(preview_df, use_container_width=True)
                    st.success("✅ Data successfully loaded into DuckDB for SQL queries")
                except Exception as e:
                    st.error(f"❌ Error connecting to DuckDB: {e}")
            
            # Enhanced Data Quality Report
            st.markdown("## 🔍 AI-Powered Data Quality Assessment")
            
            if st.button("🔄 Generate Data Quality Report", type="primary"):
                with st.spinner("🤖 AI is analyzing data quality..."):
                    quality_report = enhanced_data_quality_report(df)
                    
                    # Display quality metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Data Completeness", f"{quality_report['overview']['completeness']:.1f}%")
                    with col2:
                        st.metric("Missing Cells", quality_report['overview']['missing_cells'])
                    with col3:
                        st.metric("Total Columns", quality_report['overview']['columns'])
                    with col4:
                        st.metric("Quality Score", f"{(quality_report['overview']['completeness'] / 100 * 95):.1f}%")
                    
                    # Quality Issues
                    if quality_report['quality_issues']:
                        st.warning("### 🚨 Data Quality Issues Found")
                        for issue in quality_report['quality_issues']:
                            st.write(f"• {issue}")
                    else:
                        st.success("### ✅ No major data quality issues detected!")
                    
                    # Column Analysis
                    st.markdown("### 📊 Detailed Column Analysis")
                    for col, analysis in quality_report['column_analysis'].items():
                        with st.expander(f"🔍 {col} ({analysis['dtype']})"):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Missing Values", 
                                        f"{analysis['missing']} ({analysis['missing_percent']:.1f}%)")
                            with col2:
                                st.metric("Unique Values", analysis['unique_values'])
                            with col3:
                                if analysis['sample_values']:
                                    st.write("Sample Values:", analysis['sample_values'][:3])
            
            # AI Insights Section
            st.markdown("## 🤖 AI-Powered Data Insights")
            
            if auto_analyze or st.button("🎯 Generate AI Insights"):
                if "groq_key" not in st.session_state:
                    st.error("🔑 Groq API key required for AI insights. Please enter your API key in the sidebar.")
                else:
                    with st.spinner("🤖 AI is analyzing your data and generating insights..."):
                        try:
                            groq_model = GroqChatWrapper(
                                model="llama-3.3-70b-versatile", 
                                api_key=st.session_state.groq_key
                            )
                            insights = generate_ai_insights(groq_model, df, st.session_state.sample_size)
                            
                            st.success("### 📋 AI Analysis Report")
                            
                            # Display insights in a nice format
                            insight_container = st.container()
                            with insight_container:
                                st.markdown("""
                                <div style="background: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #667eea;">
                                """, unsafe_allow_html=True)
                                st.markdown(insights)
                                st.markdown("</div>", unsafe_allow_html=True)
                            
                            # Store insights for later use
                            st.session_state.ai_insights = insights
                            
                        except Exception as e:
                            st.error(f"❌ Error generating AI insights: {e}")
                            st.info("💡 Please check your API key and try again.")
            
            # Next Steps Recommendations
            st.markdown("## 🚀 Recommended Next Steps")
            
            rec_col1, rec_col2, rec_col3 = st.columns(3)
            
            with rec_col1:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 15px; border-radius: 10px; color: white; height: 150px;">
                <h4 style="color: white;">🔍 Explore Data</h4>
                <p>Use the EDA page to explore distributions, correlations, and patterns in your data.</p>
                </div>
                """, unsafe_allow_html=True)
            
            with rec_col2:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 15px; border-radius: 10px; color: white; height: 150px;">
                <h4 style="color: white;">💬 Ask Questions</h4>
                <p>Use natural language to query your data and get instant answers with AI.</p>
                </div>
                """, unsafe_allow_html=True)
            
            with rec_col3:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); padding: 15px; border-radius: 10px; color: white; height: 150px;">
                <h4 style="color: white;">📊 Generate Reports</h4>
                <p>Create comprehensive analysis reports with automated insights and visualizations.</p>
                </div>
                """, unsafe_allow_html=True)

# Handle case where data is already in session state
elif "df" in st.session_state:
    st.info("ℹ️ Using previously uploaded dataset from session.")
    df = st.session_state.df
    
    st.markdown("### 📋 Current Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", f"{len(df):,}")
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
        st.metric("Numeric Columns", numeric_cols)
    with col4:
        categorical_cols = len(df.select_dtypes(include=['object']).columns)
        st.metric("Categorical Columns", categorical_cols)
    
    st.markdown("#### 👀 Data Preview (First 20 rows)")
    st.dataframe(df.head(20), use_container_width=True)
    
    # Show AI insights if available
    if "ai_insights" in st.session_state:
        st.markdown("### 🤖 Previous AI Insights")
        st.info(st.session_state.ai_insights)

else:
    # Welcome and instructions
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 15px; color: white; text-align: center;">
    <h2 style="color: white;">Welcome to AI-Enhanced Data Analysis! 🚀</h2>
    <p style="font-size: 18px;">Upload your dataset to unlock powerful AI-powered insights and analysis capabilities.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📁 Supported Formats
        - **CSV Files** (.csv) - Comma separated values
        - **Excel Files** (.xlsx) - Microsoft Excel workbooks
        
        ### 🎯 What to Expect
        - **Automated Data Cleaning**
        - **AI-Powered Insights**
        - **Data Quality Assessment**
        - **Interactive Visualizations**
        - **Natural Language Queries**
        """)
    
    with col2:
        st.markdown("""
        ### 🔧 Key Features
        - **Smart Data Preprocessing**
        - **Automatic Type Detection**
        - **Missing Value Analysis**
        - **Pattern Recognition**
        - **Anomaly Detection**
        - **Trend Forecasting**
        
        ### 🚀 Get Started
        1. Upload your dataset
        2. Configure AI settings
        3. Explore insights
        4. Generate reports
        """)

# Footer with status
st.markdown("---")
if "df" in st.session_state:
    st.success(f"✅ Dataset loaded: {len(st.session_state.df):,} rows × {len(st.session_state.df.columns)} columns")
    if "ai_insights" in st.session_state:
        st.info("🤖 AI insights available - Check the analysis pages!")
else:
    st.warning("📂 Please upload a dataset to get started with AI-powered analysis.")