import pandas as pd
import tempfile
import csv
import re
import logging
import duckdb
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from datetime import datetime, timedelta
import json
import numpy as np

# --------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --------- Simplified Groq wrapper ----------
class GroqChatWrapper:
    def __init__(self, model: str = "llama-3.3-70b-versatile", api_key: str = None):
        self.model = model
        self.api_key = api_key
        if api_key:
            self.client = Groq(api_key=api_key)
        else:
            self.client = None

    def run(self, messages: list[dict], **kwargs) -> str:
        if not self.client:
            raise ValueError("Groq client not initialized. Please provide API key.")
        
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            **kwargs
        )
        return resp.choices[0].message.content.strip()

# --------- AI-Powered Analysis Functions ----------
def generate_ai_insights(groq_model: GroqChatWrapper, df: pd.DataFrame, sample_size: int = 1000) -> str:
    """Generate AI-powered insights about the dataset."""
    sample_df = df.sample(min(sample_size, len(df)))
    dataset_info = f"""
    Dataset Shape: {df.shape}
    Columns: {list(df.columns)}
    Numeric Columns: {list(df.select_dtypes(include=['number']).columns)}
    Categorical Columns: {list(df.select_dtypes(include=['object']).columns)}
    Sample Data:
    {sample_df.head(5).to_string()}
    """
    
    system_prompt = """You are a senior data analyst. Analyze the dataset and provide:
    1. Key patterns and trends
    2. Potential business insights
    3. Data quality issues
    4. Recommended next analyses
    5. Interesting correlations to explore
    
    Be concise but insightful. Focus on actionable insights."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Analyze this dataset:\n{dataset_info}"}
    ]
    
    return groq_model.run(messages)

def detect_anomalies(df: pd.DataFrame, numeric_columns: list) -> pd.DataFrame:
    """Detect anomalies using Isolation Forest."""
    if not numeric_columns:
        return pd.DataFrame()
    
    # Select only numeric columns
    numeric_df = df[numeric_columns].dropna()
    
    if len(numeric_df) < 10:
        return pd.DataFrame()
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)
    
    # Fit Isolation Forest
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    anomalies = iso_forest.fit_predict(scaled_data)
    
    # Create results
    result_df = numeric_df.copy()
    result_df['is_anomaly'] = anomalies == -1
    result_df['anomaly_score'] = iso_forest.decision_function(scaled_data)
    
    return result_df[result_df['is_anomaly']].head(20)

def forecast_trend(df: pd.DataFrame, date_column: str, value_column: str, periods: int = 30) -> go.Figure:
    """Forecast future trends using simple linear regression."""
    try:
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        
        # Aggregate by date
        daily_data = df.groupby(date_column)[value_column].mean().reset_index()
        daily_data = daily_data.sort_values(date_column)
        
        # Create numeric index for regression
        daily_data['day_num'] = (daily_data[date_column] - daily_data[date_column].min()).dt.days
        
        # Fit linear regression
        X = sm.add_constant(daily_data['day_num'])
        y = daily_data[value_column]
        model = sm.OLS(y, X).fit()
        
        # Forecast future values
        last_date = daily_data[date_column].max()
        future_days = range(daily_data['day_num'].max() + 1, daily_data['day_num'].max() + periods + 1)
        future_dates = [last_date + timedelta(days=int(x - daily_data['day_num'].max())) for x in future_days]
        
        future_X = sm.add_constant(list(future_days))
        future_y = model.predict(future_X)
        
        # Create plot
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=daily_data[date_column],
            y=daily_data[value_column],
            mode='lines+markers',
            name='Historical',
            line=dict(color='blue')
        ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=future_y,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title=f'{value_column} Trend Forecast',
            xaxis_title='Date',
            yaxis_title=value_column,
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Forecasting error: {e}")
        return go.Figure()

# --------- Enhanced Preprocessing ----------
def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names for safe SQL usage."""
    df = df.copy()
    df.columns = (
        df.columns
          .str.strip()
          .str.replace(r"\s+", "_", regex=True)
          .str.replace(r"[^A-Za-z0-9_]", "", regex=True)
          .str.replace(r"^_+", "", regex=True)
          .str.replace(r"_+", "_", regex=True)
    )
    return df

def enhanced_data_quality_report(df: pd.DataFrame) -> dict:
    """Generate comprehensive data quality report."""
    report = {
        'overview': {
            'rows': len(df),
            'columns': len(df.columns),
            'missing_cells': df.isna().sum().sum(),
            'completeness': (1 - df.isna().sum().sum() / (len(df) * len(df.columns))) * 100
        },
        'column_analysis': {},
        'quality_issues': []
    }
    
    for col in df.columns:
        col_data = df[col]
        col_analysis = {
            'dtype': str(col_data.dtype),
            'missing': col_data.isna().sum(),
            'missing_percent': (col_data.isna().sum() / len(df)) * 100,
            'unique_values': col_data.nunique(),
            'sample_values': col_data.dropna().head(3).tolist() if col_data.dtype == 'object' else None
        }
        
        if pd.api.types.is_numeric_dtype(col_data):
            col_analysis.update({
                'mean': col_data.mean(),
                'std': col_data.std(),
                'min': col_data.min(),
                'max': col_data.max()
            })
        
        report['column_analysis'][col] = col_analysis
        
        # Detect quality issues
        if col_analysis['missing_percent'] > 50:
            report['quality_issues'].append(f"High missing values in {col}: {col_analysis['missing_percent']:.1f}%")
        if col_analysis['unique_values'] == 1:
            report['quality_issues'].append(f"Constant values in {col}")
        if col_analysis['unique_values'] == len(df):
            report['quality_issues'].append(f"All unique values in {col} - potential ID column")
    
    return report

def preprocess_and_save(file):
    """Clean dataset and save to temporary CSV for DuckDB."""
    try:
        if file.name.endswith(".csv"):
            try:
                df = pd.read_csv(file, encoding="utf-8", na_values=["NA", "N/A", "missing", ""])
            except UnicodeDecodeError:
                file.seek(0)
                df = pd.read_csv(file, encoding="latin1", na_values=["NA", "N/A", "missing", ""])
            except pd.errors.ParserError:
                file.seek(0)
                df = pd.read_csv(file, sep=None, engine="python", na_values=["NA", "N/A", "missing", ""])
            if df.shape[1] == 0:
                file.seek(0)
                df = pd.read_csv(file, header=None)
                df.columns = [f"Column_{i+1}" for i in range(df.shape[1])]

        elif file.name.endswith(".xlsx"):
            df = pd.read_excel(file, na_values=["NA", "N/A", "missing", ""])
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None, None, None

        df = clean_column_names(df)

        # Enhanced preprocessing
        for col in df.select_dtypes(include=["object"]):
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace(["nan", "None", "null", "NULL"], "")

        # Convert possible date columns
        date_patterns = ['date', 'time', 'year', 'month', 'day']
        for col in df.columns:
            if any(pattern in col.lower() for pattern in date_patterns):
                df[col] = pd.to_datetime(df[col], errors="coerce")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            temp_path = tmp.name
            df.to_csv(temp_path, index=False, quoting=csv.QUOTE_ALL)

        return temp_path, df.columns.tolist(), df

    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None, None, None

def connect_duckdb_from_csv(path: str):
    """Create DuckDB connection from a saved CSV file."""
    con = duckdb.connect()
    con.execute(f"""
        CREATE OR REPLACE TABLE uploaded_data AS
        SELECT * FROM read_csv_auto(
            '{path}',
            header=True,
            ignore_errors=True
        );
    """)
    return con

def nl_to_sql(groq_model: GroqChatWrapper, user_question: str, columns: list[str], df_sample: pd.DataFrame = None) -> str:
    """Convert natural language to SQL using Groq (DuckDB compatible)."""
    quoted_cols = ", ".join([f'"{c}"' for c in columns])
    
    # Add sample data context
    sample_context = ""
    if df_sample is not None and len(df_sample) > 0:
        sample_context = f"\nSample data (first 3 rows):\n{df_sample.head(3).to_string()}"
    
    system = (
        "You are a senior data analyst. "
        "The dataset table is called \"uploaded_data\". "
        f"Columns are: {quoted_cols}. "
        f"{sample_context}"
        "⚠️ Rules:\n"
        "- Always wrap table and column names in double quotes.\n"
        "- For text equality, use TRIM(LOWER()).\n"
        "- For partial matches in text fields, use ILIKE with wildcards.\n"
        "- Do not invent columns.\n"
        "- DuckDB does not support `REGEXP`; use regexp_matches() or regexp_extract().\n"
        "- For Duration columns: extract numbers using CAST(regexp_extract(\"Duration\", '[0-9]+') AS INTEGER).\n"
        "- For DATE columns: use year(), month(), day(). If regex needed, CAST to VARCHAR first.\n"
        "Return ONLY SQL wrapped in ```sql ... ```."
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_question}
    ]
    text = groq_model.run(messages)

    m = re.search(r"```sql\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text.strip()

# --------- EDA Helpers ----------
def df_shape(df):
    """Return dataset shape as (rows, cols)."""
    return df.shape if df is not None else (0, 0)

def df_missing(df):
    """Missing values per column."""
    missing_df = df.isna().sum().reset_index()
    missing_df.columns = ["Column", "Missing"]
    missing_df["Percentage"] = (missing_df["Missing"] / len(df)) * 100
    return missing_df

def df_duplicates(df):
    """Count duplicate rows."""
    return df.duplicated().sum()

def df_summary(df):
    """Summary statistics for numeric and categorical columns."""
    return df.describe(include="all").transpose()

def value_counts(df, column, n=10):
    """Top N value counts for a categorical column."""
    return df[column].value_counts().head(n)

def corr_matrix(df):
    """Correlation matrix for numeric columns."""
    num_df = df.select_dtypes(include=["number"])
    return num_df.corr()

def plot_histogram(df, column):
    """Plot histogram for a numeric column."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df[column].dropna(), bins=30, kde=True, ax=ax)
    ax.set_title(f"Distribution of {column}")
    st.pyplot(fig)

def plot_bar(df, column, n=10):
    """Plot bar chart for top categories in a categorical column."""
    fig, ax = plt.subplots(figsize=(12, 6))
    df[column].value_counts().head(n).plot(kind="bar", ax=ax)
    ax.set_title(f"Top {n} categories in {column}")
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

def plot_trend(df, date_column):
    """Plot trend over years for a date column."""
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
    trend = df[date_column].dt.year.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    trend.plot(kind="line", marker="o", ax=ax)
    ax.set_title(f"Trend of records over years ({date_column})")
    ax.set_xlabel("Year")
    ax.set_ylabel("Count")
    st.pyplot(fig)