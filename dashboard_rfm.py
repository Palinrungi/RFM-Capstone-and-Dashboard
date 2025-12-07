import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import datetime as dt
import warnings
import os
warnings.filterwarnings('ignore')

# ============================
# PAGE CONFIG
# ============================
st.set_page_config(
    page_title="RFM Analysis Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================
# THEME MANAGEMENT
# ============================
# Initialize theme in session state
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

def toggle_theme():
    """Toggle between light and dark theme"""
    st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'

# Theme colors
THEMES = {
    'light': {
        'bg_color': '#ffffff',
        'secondary_bg': '#f0f2f6',
        'text_color': '#262730',
        'card_bg': '#ffffff',
        'border_color': '#e0e0e0',
        'metric_bg': '#f0f2f6',
        'header_gradient': 'linear-gradient(90deg, #667eea 0%, #764ba2 100%)',
        'chart_template': 'plotly_white'
    },
    'dark': {
        'bg_color': '#0e1117',
        'secondary_bg': '#262730',
        'text_color': '#fafafa',
        'card_bg': '#1e1e1e',
        'border_color': '#3a3a3a',
        'metric_bg': '#262730',
        'header_gradient': 'linear-gradient(90deg, #4a5568 0%, #2d3748 100%)',
        'chart_template': 'plotly_dark'
    }
}

current_theme = THEMES[st.session_state.theme]

def apply_theme_to_chart(fig):
    """Apply current theme to plotly chart"""
    fig.update_layout(
        template=current_theme['chart_template'],
        paper_bgcolor=current_theme['card_bg'],
        plot_bgcolor=current_theme['card_bg'],
        font=dict(color=current_theme['text_color'])
    )
    return fig

# ============================
# CUSTOM CSS
# ============================
st.markdown(f"""
    <style>
    /* Main background */
    .stApp {{
        background-color: {current_theme['bg_color']};
        color: {current_theme['text_color']};
    }}
    
    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {current_theme['secondary_bg']};
    }}
    
    /* Main header */
    .main-header {{
        font-size: 42px;
        font-weight: bold;
        text-align: center;
        padding: 20px;
        background: {current_theme['header_gradient']};
        color: white;
        border-radius: 10px;
        margin-bottom: 30px;
    }}
    
    /* Metric cards */
    .metric-card {{
        background-color: {current_theme['metric_bg']};
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }}
    
    /* Segment cards */
    .segment-card {{
        background-color: {current_theme['card_bg']};
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
        border: 1px solid {current_theme['border_color']};
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        background-color: {current_theme['secondary_bg']};
        border-radius: 5px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        color: {current_theme['text_color']};
    }}
    
    /* Metrics */
    [data-testid="stMetricValue"] {{
        color: {current_theme['text_color']};
    }}
    
    /* Text elements */
    h1, h2, h3, h4, h5, h6, p, label, span {{
        color: {current_theme['text_color']} !important;
    }}
    
    /* Dataframes */
    .dataframe {{
        color: {current_theme['text_color']} !important;
        background-color: {current_theme['card_bg']} !important;
    }}
    
    /* Expander */
    .streamlit-expanderHeader {{
        background-color: {current_theme['secondary_bg']};
        color: {current_theme['text_color']};
    }}
    </style>
""", unsafe_allow_html=True)

# ============================
# LOAD DATA FUNCTION
# ============================
@st.cache_data
def load_data():
    """Load and preprocess data"""
    try:
        # Path untuk file lokal
        csv_file = 'E-Commerce Data.csv'
        
        # Jika file tidak ada, download dari Google Drive
        if not os.path.exists(csv_file):
            with st.spinner('📥 Downloading data from Google Drive...'):
                # File ID dari Google Drive
                file_id = '1tEDiDR8IPIwDbxLryW1RnTBJdjR_J2v-'
                url = f'https://drive.google.com/uc?id={file_id}'
                
                try:
                    import gdown
                    gdown.download(url, csv_file, quiet=False)
                    st.success('✅ Data downloaded successfully!')
                except ImportError:
                    st.error('❌ gdown library not installed. Please add to requirements.txt')
                    return None, None, None, None
        
        # Load CSV dengan separator semicolon
        df = pd.read_csv(csv_file, sep=';', encoding='latin-1')
        
        # Data Cleaning
        df_clean = df.copy()
        df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])
        
        # Konversi Quantity dan UnitPrice ke numeric
        df_clean['Quantity'] = pd.to_numeric(df_clean['Quantity'], errors='coerce')
        df_clean['UnitPrice'] = pd.to_numeric(df_clean['UnitPrice'], errors='coerce')
        
        df_clean = df_clean.dropna(subset=['CustomerID'])
        df_clean = df_clean.drop_duplicates()
        df_clean = df_clean[(df_clean['Quantity'] > 0) & (df_clean['UnitPrice'] > 0)]
        df_clean['TotalPrice'] = df_clean['Quantity'] * df_clean['UnitPrice']
        
        # Split data UK vs Non-UK
        df_uk = df_clean[df_clean['Country'] == 'United Kingdom'].copy()
        df_nonuk = df_clean[df_clean['Country'] != 'United Kingdom'].copy()
        
        return df, df_clean, df_uk, df_nonuk
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None

# ============================
# RFM CALCULATION FUNCTION
# ============================
@st.cache_data
def calculate_rfm(df_market):
    """Calculate RFM metrics for specified market"""
    latest_date = df_market['InvoiceDate'].max()
    
    rfm = df_market.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (latest_date - x.max()).days,
        'InvoiceNo': 'count',
        'TotalPrice': 'sum'
    }).reset_index()
    
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    
    # RFM Scoring
    rfm['R_Score'] = pd.qcut(rfm['Recency'], q=5, labels=[5, 4, 3, 2, 1])
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5])
    rfm['M_Score'] = pd.qcut(rfm['Monetary'], q=5, labels=[1, 2, 3, 4, 5])
    
    rfm[['R_Score', 'F_Score', 'M_Score']] = rfm[['R_Score', 'F_Score', 'M_Score']].astype(int)
    rfm['RFM_Score'] = rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']
    
    # Customer Segmentation
    def segment_customers(row):
        if row['R_Score'] >= 4 and row['F_Score'] >= 4 and row['M_Score'] >= 4:
            return 'Champions'
        elif row['R_Score'] >= 3 and row['F_Score'] >= 3 and row['M_Score'] >= 3:
            return 'Loyal Customers'
        elif row['R_Score'] >= 4 and row['F_Score'] >= 2:
            return 'Potential Loyalists'
        elif row['R_Score'] >= 4 and row['F_Score'] == 1:
            return 'New Customers'
        elif row['R_Score'] >= 3 and row['F_Score'] <= 2 and row['M_Score'] <= 2:
            return 'Promising'
        elif row['R_Score'] >= 2 and row['F_Score'] >= 2 and row['M_Score'] >= 2:
            return 'Need Attention'
        elif row['R_Score'] >= 2 and row['F_Score'] <= 2:
            return 'About to Sleep'
        elif row['R_Score'] <= 2 and row['F_Score'] >= 3:
            return 'Cannot Lose Them'
        elif row['R_Score'] <= 2 and row['F_Score'] == 2:
            return 'Hibernating'
        else:
            return 'Lost'
    
    rfm['Customer_Segment'] = rfm.apply(segment_customers, axis=1)
    
    # CLV Calculation
    rfm['CLV'] = rfm['Monetary'] * rfm['Frequency'] * 0.5
    rfm['CLV_segment'] = pd.qcut(rfm['CLV'], 3, labels=['Low Value', 'Medium Value', 'High Value'])
    
    return rfm

# ============================
# KMEANS CLUSTERING FUNCTION
# ============================
@st.cache_data
def perform_clustering(rfm, n_clusters):
    """Perform K-Means clustering"""
    scaler = StandardScaler()
    scaled_rfm = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    rfm['Cluster'] = kmeans.fit_predict(scaled_rfm)
    
    # PCA for visualization
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(scaled_rfm)
    
    return rfm, scaled_rfm, pca_result, kmeans

# ============================
# MAIN APP
# ============================
def main():
    # Header
    st.markdown('<div class="main-header"> RFM Analysis Dashboard</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    
    # Theme Toggle
    theme_icon = "🌙" if st.session_state.theme == 'light' else "☀️"
    theme_text = "Dark Mode" if st.session_state.theme == 'light' else "Light Mode"
    
    if st.sidebar.button(f"{theme_icon} {theme_text}", use_container_width=True):
        toggle_theme()
        st.rerun()
    
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Select Page",
        ["Overview", "RFM Analysis", "Customer Segments", "K-Means Clustering", "CLV Analysis", "Market Basket Analysis"]
    )
    
    # Load Data
    with st.spinner("Loading data..."):
        df, df_clean, df_uk, df_nonuk = load_data()
    
    if df is None or df_clean is None:
        st.error("Failed to load data. Please check your data source.")
        return
    
    # Market Selection
    st.sidebar.markdown("---")
    market = st.sidebar.selectbox(" Select Market", ["UK", "Non-UK", "Both"])
    
    # ============================
    # PAGE: OVERVIEW
    # ============================
    if page == "Overview":
        st.title("Business Overview")
        
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Revenue",
                value=f"${df_clean['TotalPrice'].sum():,.0f}",
                delta="All Time"
            )
        
        with col2:
            st.metric(
                label="Total Customers",
                value=f"{df_clean['CustomerID'].nunique():,}",
                delta=None
            )
        
        with col3:
            st.metric(
                label="Total Transactions",
                value=f"{len(df_clean):,}",
                delta=None
            )
        
        with col4:
            st.metric(
                label="Avg Order Value",
                value=f"${df_clean['TotalPrice'].mean():.2f}",
                delta=None
            )
        
        st.markdown("---")
        
        # Market Distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Revenue by Market")
            market_revenue = df_clean.groupby('Country')['TotalPrice'].sum().nlargest(10).reset_index()
            fig = px.bar(market_revenue, x='Country', y='TotalPrice', 
                        title="Top 10 Countries by Revenue",
                        labels={'TotalPrice': 'Revenue ($)', 'Country': 'Country'})
            fig.update_traces(marker_color='#667eea')
            apply_theme_to_chart(fig)
            st.plotly_chart(fig, width="stretch")
        
        with col2:
            st.subheader("Monthly Revenue Trend")
            df_clean['Month'] = df_clean['InvoiceDate'].dt.to_period('M').astype(str)
            monthly_revenue = df_clean.groupby('Month')['TotalPrice'].sum().reset_index()
            fig = px.line(monthly_revenue, x='Month', y='TotalPrice',
                         title="Revenue Trend Over Time",
                         labels={'TotalPrice': 'Revenue ($)', 'Month': 'Month'})
            fig.update_traces(line_color='#764ba2', line_width=3)
            apply_theme_to_chart(fig)
            st.plotly_chart(fig, width="stretch")
        
        # Customer Behavior
        st.markdown("---")
        st.subheader("👥 Customer Behavior Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            customer_orders = df_clean.groupby('CustomerID')['InvoiceNo'].nunique()
            fig = px.histogram(customer_orders, nbins=30, 
                              title="Distribution of Orders per Customer",
                              labels={'value': 'Number of Orders', 'count': 'Number of Customers'})
            fig.update_traces(marker_color='lightblue')
            apply_theme_to_chart(fig)
            st.plotly_chart(fig, width="stretch")
        
        with col2:
            customer_spending = df_clean.groupby('CustomerID')['TotalPrice'].sum()
            fig = px.histogram(customer_spending, nbins=30,
                              title="Distribution of Total Spending per Customer",
                              labels={'value': 'Total Spending ($)', 'count': 'Number of Customers'})
            fig.update_traces(marker_color='lightcoral')
            apply_theme_to_chart(fig)
            st.plotly_chart(fig, width="stretch")
    
    # ============================
    # PAGE: RFM ANALYSIS
    # ============================
    elif page == "RFM Analysis":
        st.title("RFM Analysis")
        
        if market == "Both":
            tab1, tab2 = st.tabs(["UK Market", "Non-UK Market"])
            
            with tab1:
                rfm_uk = calculate_rfm(df_uk)
                display_rfm_analysis(rfm_uk, "UK")
            
            with tab2:
                rfm_nonuk = calculate_rfm(df_nonuk)
                display_rfm_analysis(rfm_nonuk, "Non-UK")
        else:
            df_market = df_uk if market == "UK" else df_nonuk
            rfm = calculate_rfm(df_market)
            display_rfm_analysis(rfm, market)
    
    # ============================
    # PAGE: CUSTOMER SEGMENTS
    # ============================
    elif page == "Customer Segments":
        st.title("Customer Segmentation")
        
        if market == "Both":
            tab1, tab2 = st.tabs(["UK Market", "Non-UK Market"])
            
            with tab1:
                rfm_uk = calculate_rfm(df_uk)
                display_segment_analysis(rfm_uk, "UK")
            
            with tab2:
                rfm_nonuk = calculate_rfm(df_nonuk)
                display_segment_analysis(rfm_nonuk, "Non-UK")
        else:
            df_market = df_uk if market == "UK" else df_nonuk
            rfm = calculate_rfm(df_market)
            display_segment_analysis(rfm, market)
    
    # ============================
    # PAGE: K-MEANS CLUSTERING
    # ============================
    elif page == "K-Means Clustering":
        st.title("K-Means Clustering Analysis")
        
        if market == "Both":
            tab1, tab2 = st.tabs(["UK Market", "Non-UK Market"])
            
            with tab1:
                rfm_uk = calculate_rfm(df_uk)
                n_clusters_uk = st.slider("Select number of clusters (UK)", 2, 10, 5, key="uk_clusters")
                display_clustering_analysis(rfm_uk, n_clusters_uk, "UK")
            
            with tab2:
                rfm_nonuk = calculate_rfm(df_nonuk)
                n_clusters_nonuk = st.slider("Select number of clusters (Non-UK)", 2, 10, 4, key="nonuk_clusters")
                display_clustering_analysis(rfm_nonuk, n_clusters_nonuk, "Non-UK")
        else:
            df_market = df_uk if market == "UK" else df_nonuk
            rfm = calculate_rfm(df_market)
            n_clusters = st.sidebar.slider("Select number of clusters", 2, 10, 5 if market == "UK" else 4)
            display_clustering_analysis(rfm, n_clusters, market)
    
    # ============================
    # PAGE: CLV ANALYSIS
    # ============================
    elif page == "CLV Analysis":
        st.title("Customer Lifetime Value Analysis")
        
        if market == "Both":
            tab1, tab2 = st.tabs(["UK Market", "Non-UK Market"])
            
            with tab1:
                rfm_uk = calculate_rfm(df_uk)
                display_clv_analysis(rfm_uk, "UK")
            
            with tab2:
                rfm_nonuk = calculate_rfm(df_nonuk)
                display_clv_analysis(rfm_nonuk, "Non-UK")
        else:
            df_market = df_uk if market == "UK" else df_nonuk
            rfm = calculate_rfm(df_market)
            display_clv_analysis(rfm, market)
    
    # ============================
    # PAGE: MARKET BASKET ANALYSIS
    # ============================
    elif page == "Market Basket Analysis":
        st.title("Market Basket Analysis")
        
        if market == "Both":
            tab1, tab2 = st.tabs(["UK Market", "Non-UK Market"])
            
            with tab1:
                display_association_rules(df_uk, "UK")
            
            with tab2:
                display_association_rules(df_nonuk, "Non-UK")
        else:
            df_market = df_uk if market == "UK" else df_nonuk
            display_association_rules(df_market, market)

# ============================
# DISPLAY FUNCTIONS
# ============================
def display_rfm_analysis(rfm, market_name):
    """Display RFM metrics and distributions"""
    st.subheader(f"RFM Metrics Summary - {market_name}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Avg Recency", f"{rfm['Recency'].mean():.1f} days")
        st.metric("Min Recency", f"{rfm['Recency'].min()} days")
        st.metric("Max Recency", f"{rfm['Recency'].max()} days")
    
    with col2:
        st.metric("Avg Frequency", f"{rfm['Frequency'].mean():.1f} orders")
        st.metric("Min Frequency", f"{rfm['Frequency'].min()} orders")
        st.metric("Max Frequency", f"{rfm['Frequency'].max()} orders")
    
    with col3:
        st.metric("Avg Monetary", f"${rfm['Monetary'].mean():,.2f}")
        st.metric("Min Monetary", f"${rfm['Monetary'].min():,.2f}")
        st.metric("Max Monetary", f"${rfm['Monetary'].max():,.2f}")
    
    st.markdown("---")
    
    # RFM Distributions
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(rfm, x='Recency', nbins=30,
                          title="Recency Distribution",
                          labels={'Recency': 'Days Since Last Purchase'})
        fig.update_traces(marker_color='skyblue')
        apply_theme_to_chart(fig)
        st.plotly_chart(fig, width="stretch")
        
        fig = px.histogram(rfm, x='Monetary', nbins=30,
                          title="Monetary Distribution",
                          labels={'Monetary': 'Total Spending ($)'})
        fig.update_traces(marker_color='salmon')
        apply_theme_to_chart(fig)
        st.plotly_chart(fig, width="stretch")
    
    with col2:
        fig = px.histogram(rfm, x='Frequency', nbins=30,
                          title="Frequency Distribution",
                          labels={'Frequency': 'Number of Orders'})
        fig.update_traces(marker_color='lightgreen')
        apply_theme_to_chart(fig)
        st.plotly_chart(fig, width="stretch")
        
        fig = px.bar(rfm['RFM_Score'].value_counts().sort_index(),
                    title="RFM Score Distribution",
                    labels={'index': 'RFM Score', 'value': 'Number of Customers'})
        fig.update_traces(marker_color='purple')
        apply_theme_to_chart(fig)
        st.plotly_chart(fig, width="stretch")
    
    # Correlation Matrix
    st.subheader("RFM Correlation Matrix")
    corr_matrix = rfm[['Recency', 'Frequency', 'Monetary']].corr()
    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                   title="RFM Metrics Correlation",
                   color_continuous_scale='RdBu_r')
    apply_theme_to_chart(fig)
    st.plotly_chart(fig, width="stretch")

def display_segment_analysis(rfm, market_name):
    """Display customer segment analysis"""
    st.subheader(f"Customer Segments - {market_name}")
    
    segment_counts = rfm['Customer_Segment'].value_counts()
    segment_percentages = (segment_counts / len(rfm) * 100).round(2)
    
    # Segment Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(values=segment_counts.values, names=segment_counts.index,
                    title="Customer Segment Distribution",
                    hole=0.4)
        apply_theme_to_chart(fig)
        st.plotly_chart(fig, width="stretch")
    
    with col2:
        fig = px.bar(x=segment_counts.index, y=segment_counts.values,
                    title="Customer Segment Counts",
                    labels={'x': 'Segment', 'y': 'Number of Customers'})
        fig.update_traces(marker_color='lightblue')
        apply_theme_to_chart(fig)
        st.plotly_chart(fig, width="stretch")
    
    # Segment Metrics
    st.markdown("---")
    st.subheader("Segment Performance Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        segment_monetary = rfm.groupby('Customer_Segment')['Monetary'].mean().sort_values(ascending=False)
        fig = px.bar(x=segment_monetary.index, y=segment_monetary.values,
                    title="Average Monetary Value by Segment",
                    labels={'x': 'Segment', 'y': 'Average Spending ($)'})
        fig.update_traces(marker_color='lightcoral')
        apply_theme_to_chart(fig)
        st.plotly_chart(fig, width="stretch")
    
    with col2:
        segment_frequency = rfm.groupby('Customer_Segment')['Frequency'].mean().sort_values(ascending=False)
        fig = px.bar(x=segment_frequency.index, y=segment_frequency.values,
                    title="Average Frequency by Segment",
                    labels={'x': 'Segment', 'y': 'Average Orders'})
        fig.update_traces(marker_color='lightgreen')
        apply_theme_to_chart(fig)
        st.plotly_chart(fig, width="stretch")
    
    # Detailed Segment Summary
    st.markdown("---")
    st.subheader("Detailed Segment Summary")
    
    segment_summary = []
    for segment in segment_counts.index:
        segment_data = rfm[rfm['Customer_Segment'] == segment]
        segment_summary.append({
            'Segment': segment,
            'Count': len(segment_data),
            'Percentage': f"{(len(segment_data) / len(rfm) * 100):.2f}%",
            'Avg Recency': f"{segment_data['Recency'].mean():.1f} days",
            'Avg Frequency': f"{segment_data['Frequency'].mean():.1f} orders",
            'Avg Monetary': f"${segment_data['Monetary'].mean():,.2f}",
            'Total Revenue': f"${segment_data['Monetary'].sum():,.2f}"
        })
    
    summary_df = pd.DataFrame(segment_summary)
    st.dataframe(summary_df, width="stretch")

def display_clustering_analysis(rfm, n_clusters, market_name):
    """Display K-Means clustering analysis"""
    rfm_clustered, scaled_data, pca_result, kmeans = perform_clustering(rfm, n_clusters)
    
    st.subheader(f"K-Means Clustering ({n_clusters} clusters) - {market_name}")
    
    # Cluster Statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Number of Clusters", n_clusters)
    
    with col2:
        silhouette = silhouette_score(scaled_data, rfm_clustered['Cluster'])
        st.metric("Silhouette Score", f"{silhouette:.4f}")
    
    with col3:
        st.metric("Total Customers", len(rfm_clustered))
    
    # Cluster Distribution
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        cluster_counts = rfm_clustered['Cluster'].value_counts().sort_index()
        fig = px.bar(x=cluster_counts.index, y=cluster_counts.values,
                    title="Customers per Cluster",
                    labels={'x': 'Cluster', 'y': 'Number of Customers'})
        apply_theme_to_chart(fig)
        st.plotly_chart(fig, width="stretch")
    
    with col2:
        fig = px.pie(values=cluster_counts.values, names=cluster_counts.index,
                    title="Cluster Distribution",
                    hole=0.4)
        apply_theme_to_chart(fig)
        st.plotly_chart(fig, width="stretch")
    
    # 3D Visualization
    st.markdown("---")
    st.subheader("3D Cluster Visualization (PCA)")
    
    pca_df = pd.DataFrame({
        'PC1': pca_result[:, 0],
        'PC2': pca_result[:, 1],
        'PC3': pca_result[:, 2],
        'Cluster': rfm_clustered['Cluster'].astype(str)
    })
    
    fig = px.scatter_3d(pca_df, x='PC1', y='PC2', z='PC3', color='Cluster',
                       title="3D PCA Visualization of Clusters")
    apply_theme_to_chart(fig)
    st.plotly_chart(fig, width="stretch")
    
    # Cluster Characteristics
    st.markdown("---")
    st.subheader("Cluster Characteristics")
    
    cluster_summary = rfm_clustered.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean',
        'CustomerID': 'count'
    }).round(2)
    cluster_summary.columns = ['Avg Recency', 'Avg Frequency', 'Avg Monetary', 'Customer Count']
    st.dataframe(cluster_summary, width="stretch")

def display_clv_analysis(rfm, market_name):
    """Display CLV analysis"""
    st.subheader(f"Customer Lifetime Value - {market_name}")
    
    # CLV Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total CLV", f"${rfm['CLV'].sum():,.0f}")
    
    with col2:
        st.metric("Avg CLV", f"${rfm['CLV'].mean():,.2f}")
    
    with col3:
        st.metric("Max CLV", f"${rfm['CLV'].max():,.2f}")
    
    with col4:
        st.metric("Min CLV", f"${rfm['CLV'].min():,.2f}")
    
    # CLV Segment Distribution
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        clv_counts = rfm['CLV_segment'].value_counts()
        fig = px.pie(values=clv_counts.values, names=clv_counts.index,
                    title="CLV Segment Distribution",
                    color_discrete_map={'High Value': 'green', 'Medium Value': 'yellow', 'Low Value': 'red'})
        apply_theme_to_chart(fig)
        st.plotly_chart(fig, width="stretch")
    
    with col2:
        clv_summary = rfm.groupby('CLV_segment')['CLV'].agg(['count', 'mean']).reset_index()
        fig = px.bar(clv_summary, x='CLV_segment', y='mean',
                    title="Average CLV by Segment",
                    labels={'mean': 'Average CLV ($)', 'CLV_segment': 'Segment'})
        apply_theme_to_chart(fig)
        st.plotly_chart(fig, width="stretch")
    
    # CLV Distribution
    st.markdown("---")
    st.subheader("CLV Distribution")
    
    fig = px.histogram(rfm, x='CLV', nbins=50,
                      title="Distribution of Customer Lifetime Value",
                      labels={'CLV': 'Customer Lifetime Value ($)'})
    fig.update_traces(marker_color='mediumpurple')
    apply_theme_to_chart(fig)
    st.plotly_chart(fig, width="stretch")
    
    # Top Customers by CLV
    st.markdown("---")
    st.subheader("Top 20 Customers by CLV")
    
    top_clv = rfm.nlargest(20, 'CLV')[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'CLV', 'CLV_segment']]
    top_clv['CLV'] = top_clv['CLV'].apply(lambda x: f"${x:,.2f}")
    top_clv['Monetary'] = top_clv['Monetary'].apply(lambda x: f"${x:,.2f}")
    st.dataframe(top_clv, width="stretch")

def display_association_rules(df_market, market_name):
    """Display market basket analysis"""
    st.subheader(f"Market Basket Analysis - {market_name}")
    
    try:
        # Preprocessing
        df_market = df_market.copy()
        df_market['StockCode'] = df_market['StockCode'].astype(str)
        df_market['Description'] = df_market['Description'].astype(str).str.strip()
        
        # Product mapping
        product_map = df_market[['StockCode', 'Description']].dropna().drop_duplicates(subset='StockCode')
        product_map = product_map.set_index('StockCode')['Description'].to_dict()
        
        # Transaction preparation
        trans = df_market.groupby('InvoiceNo')['StockCode'].apply(list).reset_index(name='Items')
        
        # Transaction Encoder
        te = TransactionEncoder()
        te_data = te.fit(trans['Items']).transform(trans['Items'])
        df_te = pd.DataFrame(te_data, columns=te.columns_)
        
        # Apriori
        min_support = st.slider("Minimum Support", 0.001, 0.05, 0.01, 0.001)
        frequent_items = apriori(df_te, min_support=min_support, use_colnames=True)
        
        if len(frequent_items) == 0:
            st.warning("No frequent itemsets found. Try lowering the minimum support.")
            return
        
        # Association Rules
        min_lift = st.slider("Minimum Lift", 1.0, 5.0, 1.0, 0.1)
        rules = association_rules(frequent_items, metric="lift", min_threshold=min_lift)
        
        if len(rules) == 0:
            st.warning("No association rules found. Try lowering the minimum lift.")
            return
        
        # Map product names
        def get_product_names(codes):
            return ', '.join([product_map.get(code, 'Unknown') for code in codes])
        
        rules['Antecedents_Name'] = rules['antecedents'].apply(get_product_names)
        rules['Consequents_Name'] = rules['consequents'].apply(get_product_names)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Transactions", len(trans))
        
        with col2:
            st.metric("Frequent Itemsets", len(frequent_items))
        
        with col3:
            st.metric("Association Rules", len(rules))
        
        # Top Rules
        st.markdown("---")
        st.subheader("Top 20 Product Recommendations")
        
        top_rules = rules.nlargest(20, 'lift')[['Antecedents_Name', 'Consequents_Name', 'support', 'confidence', 'lift']]
        top_rules['support'] = top_rules['support'].apply(lambda x: f"{x:.4f}")
        top_rules['confidence'] = top_rules['confidence'].apply(lambda x: f"{x:.4f}")
        top_rules['lift'] = top_rules['lift'].apply(lambda x: f"{x:.2f}")
        
        st.dataframe(top_rules, width="stretch")
        
        # Visualization
        st.markdown("---")
        st.subheader("Support vs Confidence vs Lift")
        
        fig = px.scatter(rules, x='support', y='confidence', size='lift', 
                        hover_data=['Antecedents_Name', 'Consequents_Name'],
                        title="Association Rules Scatter Plot",
                        labels={'support': 'Support', 'confidence': 'Confidence', 'lift': 'Lift'})
        apply_theme_to_chart(fig)
        st.plotly_chart(fig, width="stretch")
        
    except Exception as e:
        st.error(f"Error in market basket analysis: {e}")

# ============================
# RUN APP
# ============================
if __name__ == "__main__":
    main()
