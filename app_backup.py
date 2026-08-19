"""
Multi-Agent Data Analyst - Streamlit Web UI
============================================
Modern, professional web interface for AI-powered data analysis.
"""

import os
import streamlit as st
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from utils.data_processor import DataProcessor
from utils.visualizer import DataVisualizer

# Page configuration
st.set_page_config(
    page_title="🤖 AI Data Analyst Pro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern Custom CSS
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main Header */
    .hero-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    
    .hero-title {
        font-size: 2.8rem;
        font-weight: 700;
        color: white;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .hero-subtitle {
        font-size: 1.1rem;
        color: rgba(255,255,255,0.9);
        margin-top: 0.5rem;
    }
    
    /* Card Styles */
    .glass-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
        margin-bottom: 1rem;
    }
    
    /* Metric Cards */
    .metric-container {
        background: linear-gradient(145deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 16px;
        padding: 1.2rem;
        text-align: center;
        border-left: 4px solid #667eea;
        transition: transform 0.3s ease;
    }
    
    .metric-container:hover {
        transform: translateY(-5px);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin: 0;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.3rem;
    }
    
    /* Status Badges */
    .status-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 50px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    
    .status-success {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
    }
    
    .status-warning {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    
    .status-info {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #1a1a2e;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #667eea;
    }
    
    /* Agent Cards */
    .agent-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid;
        transition: all 0.3s ease;
    }
    
    .agent-card:hover {
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
    }
    
    .agent-engineer { border-color: #667eea; }
    .agent-analyst { border-color: #11998e; }
    .agent-visualizer { border-color: #f093fb; }
    .agent-writer { border-color: #f5576c; }
    
    /* Upload Area */
    .upload-area {
        border: 2px dashed #667eea;
        border-radius: 16px;
        padding: 3rem 2rem;
        text-align: center;
        background: linear-gradient(145deg, #f8f9fa 0%, #ffffff 100%);
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: #764ba2;
        background: linear-gradient(145deg, #ffffff 0%, #f0f0ff 100%);
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.7rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(102, 126, 234, 0.5);
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #f8f9fa;
        border-radius: 8px 8px 0 0;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Progress Animation */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .processing {
        animation: pulse 1.5s infinite;
    }
    
    /* Data Table Styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'df': None,
        'df_cleaned': None,
        'processor': None,
        'profile': None,
        'analysis_results': None,
        'charts': [],
        'current_page': 'home'
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sidebar():
    """Render the modern sidebar with navigation."""
    with st.sidebar:
        # Logo and branding
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h1 style="font-size: 2.5rem; margin: 0;">🧠</h1>
            <h2 style="font-size: 1.2rem; font-weight: 600; color: #667eea; margin: 0.5rem 0;">
                AI Data Analyst
            </h2>
            <p style="font-size: 0.75rem; color: #6c757d; margin: 0;">
                Powered by Groq LLM
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation Menu
        st.markdown("### 🧭 Navigation")
        
        pages = {
            "🏠 Home": "home",
            "📁 Upload Data": "upload",
            "🔍 Data Profile": "profile", 
            "📊 Analysis": "analysis",
            "📈 Visualizations": "viz",
            "🤖 AI Report": "report"
        }
        
        page = st.radio("Navigation Menu", list(pages.keys()), label_visibility="collapsed")
        
        st.markdown("---")
        
        # API Status Panel
        st.markdown("### 🔌 Connection Status")
        
        groq_key = os.getenv("GROQ_API_KEY", "")
        if groq_key and groq_key.startswith("gsk_"):
            st.markdown("""
            <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                        padding: 0.5rem 1rem; border-radius: 8px; margin: 0.3rem 0;">
                <span style="color: white; font-size: 0.85rem;">✅ Groq AI Connected</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                        padding: 0.5rem 1rem; border-radius: 8px; margin: 0.3rem 0;">
                <span style="color: white; font-size: 0.85rem;">⚠️ API Key Required</span>
            </div>
            """, unsafe_allow_html=True)
        
        # Data Status
        if st.session_state.df is not None:
            rows, cols = st.session_state.df.shape
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 0.5rem 1rem; border-radius: 8px; margin: 0.3rem 0;">
                <span style="color: white; font-size: 0.85rem;">📊 Data: {rows:,} × {cols}</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Quick Info
        with st.expander("ℹ️ About"):
            st.markdown("""
            **4 AI Agents:**
            - 🔧 Data Engineer
            - 📊 Data Analyst  
            - 📈 Visualizer
            - 📝 Report Writer
            
            **Features:**
            - Auto data cleaning
            - Statistical analysis
            - Interactive charts
            - AI-generated reports
            """)
        
        return pages[page]


def render_home_page():
    """Render the home/landing page."""
    # Hero Section
    st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">🧠 AI Data Analyst Pro</h1>
        <p class="hero-subtitle">
            Transform your data into actionable insights with our multi-agent AI system. 
            Upload any dataset and let 4 specialized AI agents analyze, visualize, and report.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature Cards
    col1, col2, col3, col4 = st.columns(4)
    
    features = [
        ("🔧", "Data Engineer", "Cleans & profiles your data automatically", "#667eea"),
        ("📊", "Data Analyst", "Statistical analysis & correlations", "#11998e"),
        ("📈", "Visualizer", "Beautiful interactive charts", "#f093fb"),
        ("📝", "Report Writer", "Executive-ready business reports", "#f5576c")
    ]
    
    for col, (icon, title, desc, color) in zip([col1, col2, col3, col4], features):
        with col:
            st.markdown(f"""
            <div class="agent-card" style="border-color: {color}; text-align: center; padding: 1.5rem;">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">{icon}</div>
                <h3 style="margin: 0.5rem 0; font-size: 1rem; color: #1a1a2e;">{title}</h3>
                <p style="font-size: 0.8rem; color: #6c757d; margin: 0;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Quick Start Section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: #1a1a2e; margin-top: 0;">🚀 Quick Start</h3>
            <ol style="color: #495057; line-height: 2;">
                <li><strong>Upload</strong> your CSV or Excel file</li>
                <li><strong>Profile</strong> - Auto-clean and analyze data structure</li>
                <li><strong>Analyze</strong> - Explore correlations and patterns</li>
                <li><strong>Visualize</strong> - Generate interactive charts</li>
                <li><strong>AI Report</strong> - Get comprehensive insights</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="glass-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
            <h3 style="color: white; margin-top: 0;">💡 Pro Tip</h3>
            <p style="color: rgba(255,255,255,0.9); font-size: 0.9rem;">
                For best results, ensure your data has clear column headers and consistent formatting.
            </p>
        </div>
        """, unsafe_allow_html=True)


def render_upload_page():
    """Render the data upload page."""
    st.markdown("""
    <div class="hero-container" style="padding: 1.5rem;">
        <h2 style="color: white; margin: 0;">📁 Upload Your Dataset</h2>
        <p style="color: rgba(255,255,255,0.8); margin: 0.5rem 0 0 0;">
            Drag & drop or browse to upload CSV/Excel files
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Drop your file here",
            type=['csv', 'xlsx', 'xls'],
            help="Supported: CSV, Excel (.xlsx, .xls)"
        )
        
        if uploaded_file is not None:
            try:
                processor = DataProcessor()
                df = processor.load_from_uploaded_file(uploaded_file)
                
                st.session_state.df = df
                st.session_state.processor = processor
                
                st.markdown(f"""
                <div style="background: #d4edda; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                    <strong>✅ Success!</strong> Loaded {df.shape[0]:,} rows × {df.shape[1]} columns
                </div>
                """, unsafe_allow_html=True)
                
                # Preview
                st.markdown("#### 📋 Data Preview")
                st.dataframe(df.head(10), width='stretch')
                
                # Action buttons
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("🧹 Clean & Profile Data", type="primary", use_container_width=True):
                        with st.spinner("Processing..."):
                            df_cleaned = processor.clean_data()
                            profile = processor.generate_profile_text()
                            st.session_state.df_cleaned = df_cleaned
                            st.session_state.profile = profile
                            st.success("✅ Data cleaned and profiled!")
                            st.balloons()
                
                with col_b:
                    if st.button("📊 Quick Stats", use_container_width=True):
                        st.markdown("**Quick Statistics:**")
                        st.write(df.describe())
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="glass-card">
            <h4 style="margin-top: 0;">📌 Supported Formats</h4>
            <ul style="color: #495057;">
                <li>CSV (.csv)</li>
                <li>Excel (.xlsx, .xls)</li>
            </ul>
            <h4>💡 Tips</h4>
            <ul style="color: #495057; font-size: 0.9rem;">
                <li>First row = headers</li>
                <li>Max file size: 200MB</li>
                <li>UTF-8 encoding preferred</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Sample data button
        if st.button("🎯 Load Sample Data", use_container_width=True):
            import random
            sample_data = {
                'CustomerID': range(1, 101),
                'Name': [f'Customer_{i}' for i in range(1, 101)],
                'Age': [random.randint(22, 65) for _ in range(100)],
                'Salary': [random.randint(35000, 150000) for _ in range(100)],
                'Department': random.choices(['Sales', 'Marketing', 'IT', 'HR', 'Finance'], k=100),
                'Performance': [round(random.uniform(2.5, 5.0), 1) for _ in range(100)],
                'YearsExp': [random.randint(1, 25) for _ in range(100)]
            }
            df = pd.DataFrame(sample_data)
            processor = DataProcessor()
            processor.df = df
            processor.file_path = "sample_data.csv"
            st.session_state.df = df
            st.session_state.processor = processor
            st.success("✅ Sample data loaded!")
            st.rerun()


def render_profile_page():
    """Render the data profile page."""
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data first from the Upload page.")
        return
    
    st.markdown("""
    <div class="hero-container" style="padding: 1.5rem;">
        <h2 style="color: white; margin: 0;">🔍 Data Profile</h2>
        <p style="color: rgba(255,255,255,0.8); margin: 0.5rem 0 0 0;">
            Comprehensive overview of your dataset
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    processor = st.session_state.processor
    df = st.session_state.df_cleaned if st.session_state.df_cleaned is not None else st.session_state.df
    
    # Auto-generate profile if needed
    if st.session_state.profile is None and processor:
        with st.spinner("Generating profile..."):
            if st.session_state.df_cleaned is None:
                processor.clean_data()
                st.session_state.df_cleaned = processor.df_cleaned
                df = processor.df_cleaned
            st.session_state.profile = processor.generate_profile_text()
    
    # Key Metrics Row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    metrics = [
        ("📊", "Rows", f"{df.shape[0]:,}"),
        ("📋", "Columns", str(df.shape[1])),
        ("🔢", "Numeric", str(len(df.select_dtypes(include=['number']).columns))),
        ("📝", "Text", str(len(df.select_dtypes(exclude=['number']).columns))),
        ("❓", "Missing", str(df.isnull().sum().sum()))
    ]
    
    for col, (icon, label, value) in zip([col1, col2, col3, col4, col5], metrics):
        with col:
            st.markdown(f"""
            <div class="metric-container">
                <div style="font-size: 1.5rem;">{icon}</div>
                <p class="metric-value">{value}</p>
                <p class="metric-label">{label}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Overview", "📊 Data Types", "❓ Missing Analysis", "📈 Statistics"])
    
    with tab1:
        st.markdown("#### Data Sample (First 20 rows)")
        st.dataframe(df.head(20), width='stretch')
    
    with tab2:
        st.markdown("#### Column Information")
        dtype_data = []
        for col in df.columns:
            dtype_data.append({
                'Column': col,
                'Type': str(df[col].dtype),
                'Non-Null': int(df[col].count()),
                'Null': int(df[col].isnull().sum()),
                'Unique': int(df[col].nunique())
            })
        dtype_df = pd.DataFrame(dtype_data)
        st.dataframe(dtype_df, width='stretch')
    
    with tab3:
        st.markdown("#### Missing Values Analysis")
        if processor:
            missing_df = processor.get_missing_values()
            st.dataframe(missing_df, width='stretch')
        
        # Visual
        missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
        if missing_pct.sum() > 0:
            import plotly.express as px
            fig = px.bar(x=missing_pct.index, y=missing_pct.values, 
                        title="Missing Values by Column (%)",
                        labels={'x': 'Column', 'y': 'Missing %'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing values in dataset!")
    
    with tab4:
        st.markdown("#### Numerical Statistics")
        if processor:
            stats_df = processor.get_numerical_stats()
            st.dataframe(stats_df, width='stretch')
    
    # Full Report Expander
    with st.expander("📜 View Full Profile Report"):
        if st.session_state.profile:
            st.code(st.session_state.profile)


def render_analysis_page():
    """Render the analysis page."""
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data first from the Upload page.")
        return
    
    st.markdown("""
    <div class="hero-container" style="padding: 1.5rem;">
        <h2 style="color: white; margin: 0;">📊 Data Analysis</h2>
        <p style="color: rgba(255,255,255,0.8); margin: 0.5rem 0 0 0;">
            Explore correlations, patterns, and insights
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    processor = st.session_state.processor
    df = st.session_state.df_cleaned if st.session_state.df_cleaned is not None else st.session_state.df
    
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    
    import plotly.express as px
    
    # Correlation Section
    if len(numerical_cols) >= 2:
        st.markdown("### 🔗 Correlation Matrix")
        
        corr_matrix = df[numerical_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            aspect='auto',
            title=""
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Strong correlations
        if processor:
            strong_corr = processor.get_strong_correlations()
            if strong_corr:
                st.markdown("#### 🎯 Strong Correlations (|r| > 0.5)")
                corr_df = pd.DataFrame(strong_corr)
                st.dataframe(corr_df, width='stretch')
    
    st.markdown("---")
    
    # Segmentation Analysis
    st.markdown("### 📊 Segmentation Analysis")
    
    if categorical_cols and numerical_cols:
        col1, col2 = st.columns(2)
        with col1:
            group_col = st.selectbox("Group by:", categorical_cols, key="seg_group")
        with col2:
            agg_col = st.selectbox("Measure:", numerical_cols, key="seg_agg")
        
        if group_col and agg_col:
            grouped = df.groupby(group_col)[agg_col].agg(['mean', 'median', 'std', 'count'])
            grouped = grouped.round(2).sort_values('mean', ascending=False)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.dataframe(grouped, width='stretch')
            
            with col2:
                fig = px.bar(
                    grouped.reset_index(),
                    x=group_col,
                    y='mean',
                    color='mean',
                    color_continuous_scale='Viridis',
                    title=f"Mean {agg_col} by {group_col}"
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need both categorical and numerical columns for segmentation analysis.")


def render_visualization_page():
    """Render the visualization page."""
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data first from the Upload page.")
        return
    
    st.markdown("""
    <div class="hero-container" style="padding: 1.5rem;">
        <h2 style="color: white; margin: 0;">📈 Interactive Visualizations</h2>
        <p style="color: rgba(255,255,255,0.8); margin: 0.5rem 0 0 0;">
            Create beautiful charts from your data
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    df = st.session_state.df_cleaned if st.session_state.df_cleaned is not None else st.session_state.df
    
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    
    import plotly.express as px
    
    # Chart selector
    chart_options = ["📊 Histogram", "📈 Scatter Plot", "📉 Line Chart", 
                     "📊 Bar Chart", "🥧 Pie Chart", "📦 Box Plot"]
    
    selected_chart = st.selectbox("Select Chart Type", chart_options)
    
    st.markdown("---")
    
    if selected_chart == "📊 Histogram" and numerical_cols:
        col1, col2 = st.columns([3, 1])
        with col2:
            col = st.selectbox("Column", numerical_cols)
            bins = st.slider("Bins", 10, 100, 30)
        with col1:
            if col:
                fig = px.histogram(df, x=col, nbins=bins, title=f"Distribution of {col}",
                                  color_discrete_sequence=['#667eea'])
                fig.update_layout(height=450)
                st.plotly_chart(fig, use_container_width=True)
    
    elif selected_chart == "📈 Scatter Plot" and len(numerical_cols) >= 2:
        col1, col2 = st.columns([3, 1])
        with col2:
            x_col = st.selectbox("X-axis", numerical_cols, key="sc_x")
            y_col = st.selectbox("Y-axis", numerical_cols, key="sc_y")
            color_col = st.selectbox("Color", ["None"] + categorical_cols, key="sc_c")
            trendline = st.checkbox("Add Trendline", value=True)
        with col1:
            fig = px.scatter(
                df, x=x_col, y=y_col,
                color=None if color_col == "None" else color_col,
                trendline="ols" if trendline else None,
                title=f"{y_col} vs {x_col}"
            )
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)
    
    elif selected_chart == "📉 Line Chart" and numerical_cols:
        col1, col2 = st.columns([3, 1])
        with col2:
            x_col = st.selectbox("X-axis", df.columns.tolist(), key="ln_x")
            y_col = st.selectbox("Y-axis", numerical_cols, key="ln_y")
        with col1:
            fig = px.line(df.sort_values(x_col), x=x_col, y=y_col, title=f"{y_col} over {x_col}",
                         color_discrete_sequence=['#667eea'])
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)
    
    elif selected_chart == "📊 Bar Chart" and categorical_cols:
        col1, col2 = st.columns([3, 1])
        with col2:
            col = st.selectbox("Column", categorical_cols, key="bar_col")
            top_n = st.slider("Top N", 5, 20, 10)
        with col1:
            value_counts = df[col].value_counts().head(top_n)
            fig = px.bar(x=value_counts.index, y=value_counts.values,
                        title=f"Top {top_n} - {col}",
                        color=value_counts.values,
                        color_continuous_scale='Viridis')
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)
    
    elif selected_chart == "🥧 Pie Chart" and categorical_cols:
        col1, col2 = st.columns([3, 1])
        with col2:
            col = st.selectbox("Column", categorical_cols, key="pie_col")
        with col1:
            fig = px.pie(df, names=col, title=f"Distribution of {col}")
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)
    
    elif selected_chart == "📦 Box Plot" and numerical_cols:
        col1, col2 = st.columns([3, 1])
        with col2:
            num_col = st.selectbox("Value", numerical_cols, key="box_num")
            cat_col = st.selectbox("Group by", ["None"] + categorical_cols, key="box_cat")
        with col1:
            fig = px.box(df, y=num_col, x=None if cat_col == "None" else cat_col,
                        title=f"Box Plot of {num_col}", color=None if cat_col == "None" else cat_col)
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)
    
    # Auto-generate button
    st.markdown("---")
    if st.button("🎨 Auto-Generate All Charts", type="primary"):
        with st.spinner("Creating visualizations..."):
            visualizer = DataVisualizer(df)
            figures = visualizer.auto_generate_charts()
            
            for fig in figures:
                st.plotly_chart(fig, use_container_width=True)
            
            st.success(f"✅ Generated {len(figures)} charts!")


def render_report_page():
    """Render the AI report page."""
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data first from the Upload page.")
        return
    
    st.markdown("""
    <div class="hero-container" style="padding: 1.5rem;">
        <h2 style="color: white; margin: 0;">🤖 AI-Powered Analysis</h2>
        <p style="color: rgba(255,255,255,0.8); margin: 0.5rem 0 0 0;">
            Let our AI agents analyze your data and generate insights
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # API Check
    groq_key = os.getenv("GROQ_API_KEY", "")
    
    if not groq_key or not groq_key.startswith("gsk_"):
        st.markdown("""
        <div class="glass-card" style="border-left: 4px solid #f5576c;">
            <h4 style="color: #f5576c; margin-top: 0;">⚠️ API Key Required</h4>
            <p>To use AI features, you need a FREE Groq API key:</p>
            <ol>
                <li>Visit <a href="https://console.groq.com" target="_blank">console.groq.com</a></li>
                <li>Create an account (free)</li>
                <li>Generate an API key</li>
                <li>Add to your <code>.env</code> file</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # AI Agents Info
    st.markdown("### 🤖 AI Agents")
    
    col1, col2, col3, col4 = st.columns(4)
    
    agents = [
        ("🔧", "Data Engineer", "Validates & cleans", "agent-engineer"),
        ("📊", "Data Analyst", "Statistical analysis", "agent-analyst"),
        ("📈", "Visualizer", "Chart recommendations", "agent-visualizer"),
        ("📝", "Report Writer", "Business insights", "agent-writer")
    ]
    
    for col, (icon, name, desc, cls) in zip([col1, col2, col3, col4], agents):
        with col:
            st.markdown(f"""
            <div class="agent-card {cls}" style="text-align: center;">
                <span style="font-size: 1.8rem;">{icon}</span>
                <p style="font-weight: 600; margin: 0.3rem 0;">{name}</p>
                <p style="font-size: 0.75rem; color: #6c757d; margin: 0;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Generate Button
    if st.button("🚀 Run AI Analysis", type="primary", use_container_width=True):
        with st.spinner("🤖 AI agents are analyzing your data..."):
            try:
                from agents.crew_agents import DataAnalysisCrew
                
                processor = st.session_state.processor
                profile = st.session_state.profile
                
                if profile is None:
                    processor.clean_data()
                    profile = processor.generate_profile_text()
                    st.session_state.profile = profile
                
                # Progress indicators
                progress = st.progress(0, text="Starting AI analysis...")
                
                crew = DataAnalysisCrew(processor.file_path or "uploaded_data")
                
                progress.progress(25, text="🔧 Data Engineer analyzing...")
                results = crew.run_full_analysis(profile)
                
                progress.progress(100, text="✅ Complete!")
                
                st.session_state.analysis_results = results
                
                st.success("✅ AI Analysis Complete!")
                st.balloons()
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)
    
    # Display Results
    if st.session_state.analysis_results:
        st.markdown("---")
        st.markdown("### 📋 Analysis Results")
        
        results = st.session_state.analysis_results
        
        tabs = st.tabs(["📊 Analysis", "📝 Report", "💻 Viz Code"])
        
        with tabs[0]:
            st.markdown(results.get("analysis", "No analysis available"))
        
        with tabs[1]:
            report = results.get("report", "No report available")
            st.markdown(report)
            
            # Download button
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                "📥 Download Report",
                report,
                f"ai_report_{timestamp}.md",
                "text/markdown",
                use_container_width=True
            )
        
        with tabs[2]:
            st.code(results.get("visualization", "No code available"), language="python")


def main():
    """Main application entry point."""
    init_session_state()
    
    page = render_sidebar()
    
    if page == "home":
        render_home_page()
    elif page == "upload":
        render_upload_page()
    elif page == "profile":
        render_profile_page()
    elif page == "analysis":
        render_analysis_page()
    elif page == "viz":
        render_visualization_page()
    elif page == "report":
        render_report_page()


if __name__ == "__main__":
    main()
