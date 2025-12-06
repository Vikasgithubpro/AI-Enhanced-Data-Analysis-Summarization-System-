import streamlit as st

st.set_page_config(page_title="AI Enhanced Data Analysis System", page_icon="🤖", layout="wide")

st.title("🤖 AI Enhanced Data Analysis & Summarization System")
st.markdown("""
<style>
.big-font { font-size:22px !important; font-weight:500; }
.feature-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
    border-radius: 10px;
    color: white;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="big-font">
Welcome! This AI-powered system helps you explore any dataset with advanced capabilities:
</div>

<div class="feature-card">
🎯 <strong>AI-Powered Features:</strong><br>
• Natural Language → SQL Queries (Groq AI)<br>
• Automated Data Summarization & Insights<br>
• Smart Pattern Detection & Anomaly Detection<br>
• Predictive Analysis & Trend Forecasting<br>
• Intelligent Report Generation with AI Insights<br>
• <strong>NEW: Document Analysis & Chatbot</strong>
</div>

<div class="feature-card">
📊 <strong>Advanced Analytics:</strong><br>
• Interactive Exploratory Data Analysis (EDA)<br>
• Automated Statistical Analysis<br>
• Correlation & Pattern Discovery<br>
• Data Quality Assessment<br>
• Multi-dimensional Analysis<br>
• <strong>NEW: Semantic Document Search</strong>
</div>

<div class="feature-card">
📄 <strong>Document Intelligence:</strong><br>
• Multi-format Document Processing (PDF, Excel, CSV)<br>
• AI-Powered Document Chatbot<br>
• Semantic Search & Content Retrieval<br>
• Automated Document Summarization<br>
• Cross-Document Analysis
</div>
""", unsafe_allow_html=True)

st.info("🚀 **Get Started**: Upload your dataset in **📥 Upload** or analyze documents in **📄 Document Analysis**!")

# Add quick start section
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("AI Models", "4+", "Groq, Embedding, Analysis, Chat")
with col2:
    st.metric("Analysis Types", "15+", "From EDA to Document AI")
with col3:
    st.metric("Data Sources", "6+", "CSV, Excel, PDF, DB, JSON, TXT")
with col4:
    st.metric("Output Formats", "8+", "Reports, Visuals, Summaries, Exports")