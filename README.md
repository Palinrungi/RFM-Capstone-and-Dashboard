# RFM Analysis Dashboard

Dashboard interaktif untuk analisis RFM (Recency, Frequency, Monetary) dengan K-Means Clustering, Customer Lifetime Value (CLV), dan Market Basket Analysis.

## Fitur Dashboard

### 1. **Overview**
- Total Revenue, Customers, Transactions
- Revenue by Country
- Monthly Revenue Trend
- Customer Behavior Analysis

### 2. **RFM Analysis**
- RFM Metrics (Recency, Frequency, Monetary)
- Distribusi RFM
- Correlation Matrix
- RFM Score Distribution

### 3. **Customer Segments**
- 10 Customer Segments:
  - Champions
  - Loyal Customers
  - Potential Loyalists
  - New Customers
  - Promising
  - Need Attention
  - About to Sleep
  - Cannot Lose Them
  - Hibernating
  - Lost
- Segment Distribution (Pie & Bar Chart)
- Performance Metrics per Segment
- Detailed Segment Summary

### 4. **K-Means Clustering**
- Elbow Method untuk optimal clusters
- Silhouette Score
- 3D PCA Visualization
- Cluster Characteristics
- Interactive cluster selection

### 5. **CLV Analysis**
- Customer Lifetime Value Calculation
- CLV Segmentation (High/Medium/Low Value)
- CLV Distribution
- Top 20 Customers by CLV

### 6. **Market Basket Analysis**
- Apriori Algorithm
- Association Rules
- Product Recommendations
- Support, Confidence, Lift Metrics
- Interactive visualization

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Dashboard

```bash
streamlit run dashboard_rfm.py
```

Dashboard akan otomatis terbuka di browser pada `http://localhost:8501`

## Quick Start

1. **Clone atau Download Project**
2. **Install Dependencies**: `pip install -r requirements.txt`
3. **Run Dashboard**: `streamlit run dashboard_rfm.py`
4. **Pilih Market**: UK, Non-UK, atau Both
5. **Explore Pages**: Navigate menggunakan sidebar

## Dataset

Dashboard menggunakan dataset E-Commerce dari Google Drive yang berisi:
- InvoiceNo: Invoice number
- StockCode: Product code
- Description: Product name
- Quantity: Quantity of product
- InvoiceDate: Invoice date
- UnitPrice: Unit price
- CustomerID: Customer unique ID
- Country: Customer country

## Pages Available

### Overview Page
Dashboard utama dengan key metrics dan visualisasi bisnis overview

### RFM Analysis Page
Analisis mendalam RFM metrics dengan distribusi dan correlation

### Customer Segments Page
Segmentasi customer dengan 10 kategori dan performance metrics

### K-Means Clustering Page
Machine Learning clustering dengan PCA visualization

### CLV Analysis Page
Customer Lifetime Value analysis dan segmentation

### Market Basket Analysis Page
Product recommendation menggunakan association rules

## Features

- **Interactive Visualizations**: Menggunakan Plotly untuk grafik interaktif
- **Market Selection**: Pilih UK, Non-UK, atau Both markets
- **Real-time Calculations**: Semua metrics dihitung real-time
- **Responsive Design**: Tampilan responsive untuk berbagai ukuran layar
- **Export Ready**: Dapat export data dari tabel
- **Parameter Tuning**: Adjust parameters untuk clustering dan association rules

## Customization

### Ubah Minimum Support (Market Basket Analysis)
```python
min_support = st.slider("Minimum Support", 0.001, 0.05, 0.01, 0.001)
```

### Ubah Jumlah Clusters (K-Means)
```python
n_clusters = st.sidebar.slider("Select number of clusters", 2, 10, 5)
```

### Ubah Data Source
Edit fungsi `load_data()` di `dashboard_rfm.py`:
```python
@st.cache_data
def load_data():
    # Ganti dengan path file Anda
    url = 'your_data_source_here'
    df = pd.read_excel(url)
    ...
```

## Notes

- Dashboard menggunakan caching untuk performa optimal
- Data preprocessing dilakukan otomatis
- Semua visualisasi interaktif dan dapat di-zoom/pan
- Export data dari tabel dengan klik kanan > Save As

## Technology Stack

- **Streamlit**: Web framework
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Plotly**: Interactive visualizations
- **Scikit-learn**: Machine Learning (K-Means, PCA)
- **mlxtend**: Market Basket Analysis (Apriori)
- **Seaborn**: Statistical visualizations

## Support

Jika ada pertanyaan atau issue, silakan buat issue di repository ini.

## License

MIT License - Feel free to use and modify!

---

**Happy Analyzing! **

