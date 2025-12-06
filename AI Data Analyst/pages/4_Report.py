import streamlit as st
import sweetviz as sv
import tempfile
import pandas as pd
from datetime import datetime
from utils import enhanced_data_quality_report, generate_ai_insights, GroqChatWrapper

st.set_page_config(page_title="📄 AI Report Generator", layout="wide")
st.title("📄 AI-Powered Report Generator")

if "df" not in st.session_state:
    st.warning("⚠️ Please upload a dataset first (📥 Upload).")
    st.stop()

df = st.session_state.df

st.markdown("""
### 🎯 Generate Comprehensive Analysis Reports

Choose the type of report you want to generate:
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 📊 Automated EDA Report")
    st.write("Comprehensive visual analysis using Sweetviz")
    if st.button("🧾 Generate EDA Report", use_container_width=True):
        with st.spinner("Generating Sweetviz report..."):
            report = sv.analyze(df)
            tmp_html = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
            report.show_html(tmp_html.name)

            with open(tmp_html.name, "rb") as f:
                st.download_button(
                    "⬇️ Download EDA Report (HTML)",
                    f,
                    f"eda_report_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
                    "text/html",
                    use_container_width=True
                )
        st.success("✅ EDA Report generated successfully!")

with col2:
    st.markdown("#### 📋 Data Quality Report")
    st.write("Detailed data quality assessment and issues")
    if st.button("🔍 Generate Quality Report", use_container_width=True):
        with st.spinner("Analyzing data quality..."):
            quality_report = enhanced_data_quality_report(df)
            
            # Create a comprehensive quality report
            report_content = f"""
DATA QUALITY REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Dataset: {df.shape[0]} rows, {df.shape[1]} columns

OVERVIEW:
- Total Rows: {quality_report['overview']['rows']}
- Total Columns: {quality_report['overview']['columns']}
- Missing Cells: {quality_report['overview']['missing_cells']}
- Data Completeness: {quality_report['overview']['completeness']:.1f}%

QUALITY ISSUES:
"""
            if quality_report['quality_issues']:
                for issue in quality_report['quality_issues']:
                    report_content += f"- {issue}\n"
            else:
                report_content += "- No major issues detected\n"
            
            report_content += "\nCOLUMN ANALYSIS:\n"
            for col, analysis in quality_report['column_analysis'].items():
                report_content += f"\n{col} ({analysis['dtype']}):\n"
                report_content += f"  - Missing: {analysis['missing']} ({analysis['missing_percent']:.1f}%)\n"
                report_content += f"  - Unique Values: {analysis['unique_values']}\n"
                
                if pd.api.types.is_numeric_dtype(df[col]):
                    report_content += f"  - Range: {analysis['min']:.2f} to {analysis['max']:.2f}\n"
                    report_content += f"  - Mean: {analysis['mean']:.2f}\n"
            
            st.download_button(
                "⬇️ Download Quality Report",
                report_content,
                f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                "text/plain",
                use_container_width=True
            )
        st.success("✅ Quality Report generated successfully!")

with col3:
    st.markdown("#### 🤖 AI Insights Report")
    st.write("AI-generated insights and recommendations")
    if st.button("🎯 Generate AI Report", use_container_width=True):
        if "groq_key" not in st.session_state:
            st.error("Groq API key required for AI insights")
        else:
            with st.spinner("AI is generating insights..."):
                groq_model = GroqChatWrapper(model="llama-3.3-70b-versatile", api_key=st.session_state.groq_key)
                insights = generate_ai_insights(groq_model, df)
                
                ai_report = f"""
AI ENHANCED DATA ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Dataset: {df.shape[0]} rows, {df.shape[1]} columns

AI-GENERATED INSIGHTS:
{insights}

DATASET SUMMARY:
- Numeric Columns: {len(df.select_dtypes(include=['number']).columns)}
- Categorical Columns: {len(df.select_dtypes(include=['object']).columns)}
- Date Columns: {len([c for c in df.columns if 'date' in c.lower()])}
- Total Missing Values: {df.isna().sum().sum()}

RECOMMENDED NEXT STEPS:
1. Explore correlations between key variables
2. Investigate any data quality issues mentioned above
3. Consider feature engineering for machine learning
4. Validate business hypotheses with targeted queries
"""
                st.download_button(
                    "⬇️ Download AI Report",
                    ai_report,
                    f"ai_insights_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    "text/plain",
                    use_container_width=True
                )
                
                # Also show insights in the app
                st.markdown("#### 📋 AI Insights Preview")
                st.info(insights)
                
            st.success("✅ AI Insights Report generated successfully!")

# Combined Comprehensive Report
st.markdown("---")
st.markdown("### 🚀 Comprehensive Analysis Report")
st.write("Generate a complete report combining all analysis types")

if st.button("🔄 Generate Master Report", use_container_width=True, type="primary"):
    with st.spinner("Generating comprehensive master report..."):
        # This would combine all report types into one
        st.info("""
        🎉 **Master Report Features:**
        - Complete EDA with visualizations
        - Detailed data quality assessment
        - AI-generated insights and recommendations
        - Statistical summaries
        - Exportable in multiple formats
        """)
        st.warning("⚠️ Master report generation requires additional processing time and resources.")