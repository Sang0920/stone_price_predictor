"""
Stone Price Predictor - Web Application
Dự đoán giá sản phẩm đá tự nhiên dựa trên dữ liệu Salesforce

Features:
- Load dữ liệu từ Salesforce (PricebookEntry, Contract_Product__c)
- Machine Learning model để dự đoán giá
- Phân tích giá theo phân khúc (Economy, Common, Premium, Super Premium)
- Tìm sản phẩm tương tự với giá đã biết
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import os
from typing import Optional, Dict, Any, List, Tuple
import requests
from io import StringIO

# Import Salesforce data loader
from dotenv import load_dotenv
load_dotenv()  # Load .env file for Salesforce credentials

try:
    from salesforce_loader import SalesforceDataLoader
    SALESFORCE_AVAILABLE = True
except ImportError:
    SALESFORCE_AVAILABLE = False

# ============ Configuration ============
st.set_page_config(
    page_title="Stone Price Predictor",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
    }
    .segment-economy { background-color: #c6efce; color: #006100; padding: 5px 10px; border-radius: 5px; }
    .segment-common { background-color: #ffeb9c; color: #9c5700; padding: 5px 10px; border-radius: 5px; }
    .segment-premium { background-color: #ffc7ce; color: #9c0006; padding: 5px 10px; border-radius: 5px; }
    .segment-super { background-color: #9e7cc1; color: white; padding: 5px 10px; border-radius: 5px; }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
</style>
""", unsafe_allow_html=True)

# ============ Pricing Rules Constants ============
SEGMENT_THRESHOLDS = {
    'Super premium': 1500,  # >= 1500 USD/m3
    'Premium': 800,         # >= 800 USD/m3
    'Common': 400,          # >= 400 USD/m3
    'Economy': 0            # < 400 USD/m3
}

CUSTOMER_PRICING_RULES = {
    'A': {'description': 'Khách thân thiết đặc biệt', 'discount': '1.5-3% so với B', 'years': '>10', 'volume': '50-150 cont'},
    'B': {'description': 'Khách lớn, chuyên nghiệp', 'discount': '10-30 USD/m3 thấp hơn C', 'years': '3-10', 'volume': '20-50 cont'},
    'C': {'description': 'Khách hàng phổ thông', 'discount': 'Giá chuẩn', 'years': '1-5', 'volume': '5-20 cont'},
    'D': {'description': 'Khách mới, size nhỏ', 'discount': '15-45 USD/m3 cao hơn C', 'years': '1', 'volume': '1-10 cont'},
    'E': {'description': 'Sản phẩm mới, cao cấp', 'discount': 'Giá sản phẩm mới', 'years': '1', 'volume': '1-10 cont'},
    'F': {'description': 'Khách hàng dự án', 'discount': 'Tùy dự án', 'years': '1-5', 'volume': '1-50 cont'},
}

PRODUCT_FAMILIES = [
    'Exterior_Tiles', 'Interior_Tiles', 'WALLSTONE', 'PALISADE', 
    'STAIR', 'ART', 'High-Class', 'SKIRTING', 'SLAB'
]

STONE_CLASSES = ['BASALT', 'GRANITE', 'BLUE STONE']

STONE_COLOR_TYPES = [
    'BLACK BASALT', 'BLUESTONE', 'GREY GRANITE', 'ABSOLUTE BASALT',
    'WHITE GRANITE', 'YELLOW GRANITE', 'RED GRANITE', 'PINK GRANITE',
    'WHITE MARBLE', 'YELLOW MARBLE', 'HIVE BASALT'
]

CHARGE_UNITS = ['USD/PC', 'USD/M2', 'USD/TON', 'USD/ML', 'USD/M3']

# Processing codes with English names (for search dropdown)
PROCESSING_CODES = [
    ('', 'All'),
    ('CUA', 'Sawn'),
    ('DOT', 'Flamed'),
    ('DOC', 'Flamed Brush'),
    ('DOX', 'Flamed Water'),
    ('HON', 'Honed'),
    ('CTA', 'Split Handmade'),
    ('CLO', 'Sawn then Cleaved'),
    ('TDE', 'Chiseled'),
    ('GCR', 'Vibrated Honed Tumbled'),
    ('GCT', 'Old Imitation'),
    ('MGI', 'Scraped'),
    ('PCA', 'Sandblasted'),
    ('QME', 'Tumbled'),
    ('TLO', 'Cleaved'),
    ('BON', 'Polished'),
    ('BAM', 'Bush Hammered'),
    ('CHA', 'Brush'),
]


# ============ Data Generation (Simulated Salesforce Data) ============
@st.cache_data(ttl=3600)
def generate_sample_data(n_samples: int = 500) -> pd.DataFrame:
    """
    Generate sample product pricing data for demonstration.
    In production, this would be replaced with Salesforce API calls.
    """
    np.random.seed(42)
    
    data = []
    for i in range(n_samples):
        # Product attributes
        family = np.random.choice(PRODUCT_FAMILIES)
        stone_class = np.random.choice(STONE_CLASSES)
        stone_color = np.random.choice(STONE_COLOR_TYPES)
        
        # Dimensions in cm
        length = np.random.choice([10, 15, 20, 30, 40, 50, 60, 80, 100, 120])
        width = np.random.choice([5, 8, 10, 15, 20, 30, 40, 60])
        height = np.random.choice([2, 2.5, 3, 5, 6, 7, 8, 10, 12, 15, 20])
        
        # Calculate volume in m3
        volume_m3 = (length * width * height) / 1000000
        area_m2 = (length * width) / 10000
        
        # Base price calculation based on product complexity
        base_price_m3 = 350 + np.random.normal(0, 50)
        
        # Adjustments based on product type
        if family in ['STAIR', 'ART', 'High-Class']:
            base_price_m3 *= 2.5
        elif family in ['Interior_Tiles', 'SLAB']:
            base_price_m3 *= 1.8
        elif family == 'Exterior_Tiles':
            base_price_m3 *= 1.2
            
        # Stone type adjustment
        if stone_color in ['ABSOLUTE BASALT', 'WHITE MARBLE']:
            base_price_m3 *= 1.5
        elif stone_color in ['YELLOW MARBLE', 'RED GRANITE']:
            base_price_m3 *= 1.3
            
        # Size adjustment (smaller pieces = higher price per m3)
        if length <= 15 and width <= 15:
            base_price_m3 *= 1.4
        elif length >= 60 or width >= 60:
            base_price_m3 *= 0.9
            
        # Thickness adjustment
        if height <= 2:
            base_price_m3 *= 2.0  # Thin slices are more expensive
        elif height >= 10:
            base_price_m3 *= 0.85
            
        # Add noise
        price_m3 = max(200, base_price_m3 + np.random.normal(0, 80))
        
        # Calculate segment
        if price_m3 >= 1500:
            segment = 'Super premium'
        elif price_m3 >= 800:
            segment = 'Premium'
        elif price_m3 >= 400:
            segment = 'Common'
        else:
            segment = 'Economy'
            
        # Charge unit
        if family in ['PALISADE', 'STAIR']:
            charge_unit = 'USD/ML'
        elif height <= 3:
            charge_unit = 'USD/M2'
        elif length <= 20 and width <= 20:
            charge_unit = 'USD/PC'
        else:
            charge_unit = np.random.choice(['USD/M3', 'USD/TON'])
            
        # Convert price to selected unit
        if charge_unit == 'USD/M2':
            unit_price = price_m3 * height / 100
        elif charge_unit == 'USD/PC':
            unit_price = price_m3 * volume_m3
        elif charge_unit == 'USD/TON':
            specific_gravity = 2.8 if stone_class == 'BASALT' else 2.65
            unit_price = price_m3 / (specific_gravity * 1.1)
        elif charge_unit == 'USD/ML':
            unit_price = price_m3 * width * height / 10000
        else:
            unit_price = price_m3
            
        # Customer type
        customer_type = np.random.choice(['A', 'B', 'C', 'D', 'E', 'F'], 
                                         p=[0.05, 0.15, 0.35, 0.20, 0.10, 0.15])
        
        # Apply customer discount
        if customer_type == 'A':
            discount = np.random.uniform(0.015, 0.03)
        elif customer_type == 'B':
            discount = np.random.uniform(0.02, 0.04)
        elif customer_type == 'C':
            discount = 0
        elif customer_type == 'D':
            discount = -np.random.uniform(0.03, 0.06)  # Premium price
        elif customer_type == 'E':
            discount = -np.random.uniform(0.05, 0.10)
        else:
            discount = np.random.uniform(-0.02, 0.02)
            
        final_price = unit_price * (1 - discount)
        
        data.append({
            'product_id': f'PROD-{i+1:04d}',
            'product_name': f'{stone_color} {family.replace("_", " ")} {length}x{width}x{height}',
            'family': family,
            'stone_class': stone_class,
            'stone_color_type': stone_color,
            'length_cm': length,
            'width_cm': width,
            'height_cm': height,
            'volume_m3': volume_m3,
            'area_m2': area_m2,
            'charge_unit': charge_unit,
            'list_price': round(unit_price, 2),
            'price_m3': round(price_m3, 2),
            'segment': segment,
            'customer_type': customer_type,
            'discount_pct': round(discount * 100, 2),
            'final_price': round(final_price, 2),
            'created_date': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(1, 365))
        })
    
    return pd.DataFrame(data)


# ============ Machine Learning Model ============
class StonePricePredictor:
    """Machine Learning model for stone sales price prediction."""
    
    def __init__(self):
        self.model = None
        self.encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        # NOTE: segment is EXCLUDED to prevent data leakage (segment is derived from price)
        # processing_code is the main surface processing type (e.g., DOT=Flamed, HON=Honed)
        self.categorical_columns = ['family', 'stone_color_type', 'charge_unit', 'processing_code']
        self.numerical_columns = ['length_cm', 'width_cm', 'height_cm', 'volume_m3', 'area_m2']
        
    def clean_data(self, df: pd.DataFrame, target_col: str = 'sales_price') -> pd.DataFrame:
        """Clean data for training: remove invalid, missing, and outlier data."""
        df_clean = df.copy()
        
        # Remove rows with missing or invalid target
        df_clean = df_clean[df_clean[target_col].notna() & (df_clean[target_col] > 0)]
        
        # Clean processing_code: replace empty/Unknown with 'OTHER'
        if 'processing_code' in df_clean.columns:
            df_clean['processing_code'] = df_clean['processing_code'].fillna('OTHER')
            df_clean['processing_code'] = df_clean['processing_code'].replace('', 'OTHER')
            # Keep 'Unknown' as a valid category but standardize empty strings
        
        # Remove rows with missing critical features (excluding processing_code which is handled above)
        for col in self.categorical_columns:
            if col in df_clean.columns and col != 'processing_code':
                df_clean = df_clean[df_clean[col].notna()]
        
        for col in self.numerical_columns:
            if col in df_clean.columns:
                df_clean = df_clean[df_clean[col].notna() & (df_clean[col] >= 0)]
        
        # Remove extreme outliers using IQR method for target variable
        Q1 = df_clean[target_col].quantile(0.01)
        Q3 = df_clean[target_col].quantile(0.99)
        df_clean = df_clean[(df_clean[target_col] >= Q1) & (df_clean[target_col] <= Q3)]
        
        return df_clean
        
    def prepare_features(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """Prepare features for ML model."""
        features = df.copy()
        
        # Encode categorical variables
        for col in self.categorical_columns:
            if col in features.columns:
                if fit:
                    self.encoders[col] = LabelEncoder()
                    features[f'{col}_encoded'] = self.encoders[col].fit_transform(features[col].astype(str))
                else:
                    # Handle unseen categories
                    features[f'{col}_encoded'] = features[col].apply(
                        lambda x: self.encoders[col].transform([str(x)])[0] 
                        if str(x) in self.encoders[col].classes_ else -1
                    )
        
        # Select feature columns
        encoded_cols = [f'{col}_encoded' for col in self.categorical_columns if col in df.columns]
        available_numerical = [col for col in self.numerical_columns if col in df.columns]
        self.feature_columns = available_numerical + encoded_cols
        
        X = features[self.feature_columns].values
        
        if fit:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)
            
        return X
    
    def train(self, df: pd.DataFrame, target_col: str = 'sales_price') -> Dict[str, float]:
        """Train the sales price prediction model with proper data cleaning."""
        # Clean data: remove invalid, missing, and outlier data
        df_clean = self.clean_data(df, target_col)
        
        if len(df_clean) < 50:
            raise ValueError(f"Không đủ dữ liệu hợp lệ để huấn luyện model (chỉ có {len(df_clean)} mẫu, cần ít nhất 50)")
        
        # Prepare features
        X = self.prepare_features(df_clean, fit=True)
        y = df_clean[target_col].values
        
        # Split data with stratification based on charge_unit if possible
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Optimized Gradient Boosting model for price prediction
        # - subsample < 1.0 helps prevent overfitting
        # - n_iter_no_change enables early stopping
        # - lower learning rate with more estimators for better generalization
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,        # Lower learning rate for better generalization
            max_depth=4,               # Shallower trees to prevent overfitting
            min_samples_split=10,      # Require more samples to split
            min_samples_leaf=5,        # Require more samples in leaves
            subsample=0.8,             # Use 80% of data per tree (stochastic GB)
            max_features='sqrt',       # Use sqrt of features for each split
            n_iter_no_change=10,       # Early stopping if no improvement
            validation_fraction=0.1,   # Use 10% for validation
            random_state=42
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Cross-validation for more robust metrics
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='neg_mean_absolute_error')
        cv_r2_scores = cross_val_score(self.model, X, y, cv=5, scoring='r2')
        
        return {
            'mae': mae,
            'r2': r2,
            'cv_mae_mean': -cv_scores.mean(),
            'cv_mae_std': cv_scores.std(),
            'cv_r2_mean': cv_r2_scores.mean(),
            'cv_r2_std': cv_r2_scores.std(),
            'train_samples': len(df_clean),
            'removed_samples': len(df) - len(df_clean),
            'target_col': target_col,
            'n_estimators_used': self.model.n_estimators_
        }
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict sales prices for new data."""
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        X = self.prepare_features(df, fit=False)
        return self.model.predict(X)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the model."""
        if self.model is None:
            return pd.DataFrame()
        
        importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance


# ============ Helper Functions ============
def classify_segment(price_m3: float) -> str:
    """Classify price into segment."""
    if price_m3 >= 1500:
        return 'Super premium'
    elif price_m3 >= 800:
        return 'Premium'
    elif price_m3 >= 400:
        return 'Common'
    else:
        return 'Economy'

def get_segment_color(segment: str) -> str:
    """Get color for segment."""
    colors = {
        'Super premium': '#9e7cc1',
        'Premium': '#ff6b6b',
        'Common': '#ffd93d',
        'Economy': '#6bcb77'
    }
    return colors.get(segment, '#808080')

def calculate_customer_price(base_price: float, customer_type: str) -> Dict[str, float]:
    """Calculate price adjustments for different customer types."""
    adjustments = {
        'A': {'min': -0.03, 'max': -0.015, 'label': 'Bớt 1.5-3%'},
        'B': {'min': -0.04, 'max': -0.02, 'label': 'Thấp hơn 2-4%'},
        'C': {'min': 0, 'max': 0, 'label': 'Giá chuẩn'},
        'D': {'min': 0.03, 'max': 0.06, 'label': 'Cao hơn 3-6%'},
        'E': {'min': 0.05, 'max': 0.10, 'label': 'Cao hơn 5-10%'},
        'F': {'min': -0.02, 'max': 0.02, 'label': 'Tùy dự án'}
    }
    
    adj = adjustments.get(customer_type, adjustments['C'])
    
    return {
        'base_price': base_price,
        'min_price': round(base_price * (1 + adj['min']), 2),
        'max_price': round(base_price * (1 + adj['max']), 2),
        'adjustment_label': adj['label']
    }

def find_similar_products(df: pd.DataFrame, query: Dict, top_n: int = 5) -> pd.DataFrame:
    """Find similar products based on attributes."""
    # Filter by basic criteria
    mask = pd.Series([True] * len(df))
    
    if query.get('stone_color_type'):
        mask &= df['stone_color_type'] == query['stone_color_type']
    
    if query.get('family'):
        mask &= df['family'] == query['family']
    
    filtered_df = df[mask].copy()
    
    if len(filtered_df) == 0:
        return pd.DataFrame()
    
    # Calculate similarity score based on dimensions
    if all(k in query for k in ['length_cm', 'width_cm', 'height_cm']):
        filtered_df['dim_diff'] = (
            abs(filtered_df['length_cm'] - query['length_cm']) +
            abs(filtered_df['width_cm'] - query['width_cm']) +
            abs(filtered_df['height_cm'] - query['height_cm'])
        )
        filtered_df = filtered_df.nsmallest(top_n, 'dim_diff')
    else:
        filtered_df = filtered_df.head(top_n)
    
    return filtered_df


# ============ Streamlit App ============
def main():
    # Header
    st.markdown('<h1 class="main-header">💎 Stone Price Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">Dự đoán giá sản phẩm đá tự nhiên với AI và dữ liệu Salesforce</p>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'model_metrics' not in st.session_state:
        st.session_state.model_metrics = None
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 💎 Stone Price Predictor")
        st.title("⚙️ Cấu hình")
        
        # Data source - Salesforce only
        st.markdown("**Nguồn dữ liệu:** Salesforce Contract Products")
        
        # Optional account code filter for Salesforce
        account_filter = st.text_input(
            "Mã khách hàng (tùy chọn)",
            placeholder="e.g., ACC-001",
            help="Lọc theo Account_Code_C__c"
        )
        
        if st.button("🔄 Tải / Làm mới dữ liệu từ Salesforce", use_container_width=True):
            with st.spinner("Đang tải dữ liệu từ Salesforce..."):
                if SALESFORCE_AVAILABLE:
                    try:
                        loader = SalesforceDataLoader()
                        df = loader.get_contract_products(account_code=account_filter if account_filter else None)
                        if len(df) > 0:
                            st.session_state.data = df
                            st.success(f"✅ Đã tải {len(df)} sản phẩm từ Salesforce!")
                        else:
                            st.warning("⚠️ Không tìm thấy dữ liệu từ Salesforce.")
                    except Exception as e:
                        st.error(f"❌ Lỗi kết nối Salesforce: {str(e)}")
                else:
                    st.error("❌ Salesforce chưa được cấu hình. Vui lòng kiểm tra file .env")
        
        if st.session_state.data is not None:
            if st.button("🤖 Huấn luyện Model ML", use_container_width=True):
                with st.spinner("Đang huấn luyện model..."):
                    predictor = StonePricePredictor()
                    metrics = predictor.train(st.session_state.data)
                    st.session_state.model = predictor
                    st.session_state.model_metrics = metrics
                    st.success("✅ Model đã sẵn sàng!")
        
        st.divider()
        
        # Pricing rules info
        with st.expander("📋 Quy tắc định giá"):
            st.markdown("""
            **Phân khúc giá (USD/m³):**
            - 🟣 Super Premium: ≥ $1,500
            - 🔴 Premium: ≥ $800
            - 🟡 Common: ≥ $400
            - 🟢 Economy: < $400
            """)
            
        with st.expander("👥 Phân loại khách hàng"):
            for code, info in CUSTOMER_PRICING_RULES.items():
                st.markdown(f"**{code}:** {info['description']}")
                st.caption(f"Điều chỉnh: {info['discount']}")
    
    # Main content
    if st.session_state.data is None:
        st.info("👈 Vui lòng tải dữ liệu từ sidebar để bắt đầu")
        
        # Show sample pricing matrix
        st.subheader("📊 Ma trận giá theo phân khúc và loại sản phẩm")
        
        matrix_data = {
            'Loại sản phẩm': ['Đá lát mỏng 1-1.5cm', 'Đá nội ngoại thất 2-5cm', 'Đá bậc thang', 'Đá cây', 'Đá mỹ nghệ'],
            'Economy (<$400/m³)': ['Đá mẻ, đá gõ tay', 'Đá cơ bản', '-', 'Đá cây cưa lột', 'Cơ bản'],
            'Common ($400-800/m³)': ['Đá 1 mặt đốt', 'Đá lát thông dụng', 'Đá nguyên khối', 'Đốt chải', 'Trung bình'],
            'Premium ($800-1500/m³)': ['Đá xử lý đặc biệt', 'Đá cao cấp', 'Đá ốp bậc thang', 'Xử lý nhiều mặt', 'Cao cấp'],
            'Super Premium (>$1500/m³)': ['Đá mỏng đặc biệt', 'Đá nắp tường, hồ bơi', 'Đặc biệt', 'Đặc biệt', 'Mỹ nghệ đặc biệt']
        }
        st.dataframe(pd.DataFrame(matrix_data), use_container_width=True)
        return
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔮 Dự đoán giá", 
        "📊 Phân tích dữ liệu", 
        "🔍 Tìm sản phẩm tương tự",
        "📈 Hiệu suất Model",
        "📋 Dữ liệu chi tiết"
    ])
    
    # Tab 1: Price Prediction
    with tab1:
        st.subheader("🔮 Dự đoán giá sản phẩm")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Thông tin sản phẩm")
            
            family = st.selectbox("Loại sản phẩm (Family)", PRODUCT_FAMILIES)
            stone_class = st.selectbox("Loại đá (Stone Class)", STONE_CLASSES)
            stone_color = st.selectbox("Màu đá (Stone Color)", STONE_COLOR_TYPES)
            
            # Main Processing dropdown
            processing_code = st.selectbox(
                "Main Processing",
                options=[code for code, name in PROCESSING_CODES],
                format_func=lambda x: f"{x} - {dict(PROCESSING_CODES).get(x, 'Other')}" if x else "OTHER - Unknown/Other",
                index=0  # Default to first option (empty = OTHER)
            )
            # Convert empty to 'OTHER' to match model training
            if not processing_code:
                processing_code = 'OTHER'
            
            col_dim1, col_dim2, col_dim3 = st.columns(3)
            with col_dim1:
                length = st.number_input("Dài (cm)", min_value=1, max_value=300, value=30)
            with col_dim2:
                width = st.number_input("Rộng (cm)", min_value=1, max_value=300, value=30)
            with col_dim3:
                height = st.number_input("Dày (cm)", min_value=0.5, max_value=50.0, value=3.0, step=0.5)
            
            charge_unit = st.selectbox("Đơn vị tính giá", CHARGE_UNITS)
            customer_type = st.selectbox(
                "Phân loại khách hàng",
                ['C', 'A', 'B', 'D', 'E', 'F'],
                format_func=lambda x: f"{x} - {CUSTOMER_PRICING_RULES[x]['description']}"
            )
            
            predict_btn = st.button("🎯 Dự đoán giá", type="primary", use_container_width=True)
        
        with col2:
            if predict_btn and st.session_state.model is not None:
                # Prepare input data (segment is NOT included to prevent data leakage)
                volume_m3 = (length * width * height) / 1000000
                area_m2 = (length * width) / 10000
                
                input_data = pd.DataFrame([{
                    'family': family,
                    'stone_class': stone_class,
                    'stone_color_type': stone_color,
                    'processing_code': processing_code,  # Main processing code
                    'length_cm': length,
                    'width_cm': width,
                    'height_cm': height,
                    'volume_m3': volume_m3,
                    'area_m2': area_m2,
                    'charge_unit': charge_unit
                }])
                
                # Predict sales_price directly
                predicted_sales_price = st.session_state.model.predict(input_data)[0]
                
                # Classify segment based on predicted price (using price_m3 equivalent)
                if charge_unit == 'USD/M3':
                    price_for_segment = predicted_sales_price
                elif charge_unit == 'USD/PC':
                    price_for_segment = predicted_sales_price / volume_m3 if volume_m3 > 0 else predicted_sales_price
                else:
                    price_for_segment = predicted_sales_price * 50  # Rough estimate for other units
                
                segment = classify_segment(price_for_segment)
                
                # Customer price adjustment
                price_info = calculate_customer_price(predicted_sales_price, customer_type)
                
                # Display results
                st.markdown("#### 📊 Kết quả dự đoán")
                
                # Segment indicator (derived from predicted price, not input)
                segment_color = get_segment_color(segment)
                st.markdown(f"""
                <div style="background-color: {segment_color}; padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
                    <h3 style="color: white; margin: 0;">Phân khúc dự đoán: {segment}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Main predicted price metric
                st.metric(f"💰 Giá bán dự đoán ({charge_unit})", f"${predicted_sales_price:,.2f}")
                
                st.divider()
                
                st.markdown(f"**👤 Giá theo khách hàng loại {customer_type}:**")
                st.markdown(f"- Giá cơ sở: **${price_info['base_price']:,.2f}**")
                st.markdown(f"- Khoảng giá: **${price_info['min_price']:,.2f}** - **${price_info['max_price']:,.2f}**")
                st.markdown(f"- Điều chỉnh: {price_info['adjustment_label']}")
                
                st.divider()
                
                # Product info summary
                st.markdown("**📦 Thông tin sản phẩm:**")
                st.markdown(f"- Kích thước: {length} x {width} x {height} cm")
                st.markdown(f"- Thể tích: {volume_m3:.6f} m³")
                st.markdown(f"- Diện tích: {area_m2:.4f} m²")
                
        # ============ EXACT MATCH PRODUCTS (Full Width) ============
        # This section is outside the columns for full width display
        if predict_btn and st.session_state.model is not None:
            st.divider()
            st.markdown("#### 📋 Sản phẩm trong hệ thống khớp tiêu chí")
            
            # Find exact matches from database
            df = st.session_state.data.copy()
            df_clean = df[df['sales_price'].notna() & (df['sales_price'] > 0)].copy()
            
            # Build match criteria
            match_mask = (
                (df_clean['family'] == family) &
                (df_clean['stone_color_type'] == stone_color) &
                (df_clean['charge_unit'] == charge_unit) &
                (df_clean['length_cm'] == length) &
                (df_clean['width_cm'] == width) &
                (df_clean['height_cm'] == height)
            )
            
            exact_matches = df_clean[match_mask]
            
            if len(exact_matches) > 0:
                st.success(f"✅ Tìm thấy **{len(exact_matches)}** sản phẩm khớp chính xác trong hệ thống!")
                
                # Statistics
                match_prices = exact_matches['sales_price']
                stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                with stat_col1:
                    st.metric("Thấp nhất", f"${match_prices.min():,.2f}")
                with stat_col2:
                    st.metric("Cao nhất", f"${match_prices.max():,.2f}")
                with stat_col3:
                    st.metric("Trung bình", f"${match_prices.mean():,.2f}")
                with stat_col4:
                    st.metric("Trung vị", f"${match_prices.median():,.2f}")
                
                # Compare with prediction
                diff = predicted_sales_price - match_prices.mean()
                if abs(diff) < 1:
                    st.info(f"📊 Giá dự đoán gần với giá trung bình thực tế (chênh lệch: ${diff:+.2f})")
                elif diff > 0:
                    st.warning(f"📊 Giá dự đoán cao hơn giá trung bình thực tế ${diff:+.2f}")
                else:
                    st.warning(f"📊 Giá dự đoán thấp hơn giá trung bình thực tế ${diff:+.2f}")
                
                # Show table of ALL matches with ALL fields
                # Define display columns in logical order
                display_cols = [
                    'contract_product_name', 'contract_name', 'account_code',
                    'sku', 'processing_code', 'processing_name',
                    'stone_color_type', 'family', 'segment',
                    'length_cm', 'width_cm', 'height_cm',
                    'charge_unit', 'sales_price', 'price_m3',
                    'created_date', 'fy_year',
                    'quantity', 'm2', 'm3', 'total_price_usd'
                ]
                available_cols = [col for col in display_cols if col in exact_matches.columns]
                
                # Column config for English headers
                col_config = {
                    'sku': st.column_config.TextColumn('SKU'),
                    'processing_code': st.column_config.TextColumn('Main Processing Code'),
                    'processing_name': st.column_config.TextColumn('Main Processing'),
                }
                
                with st.expander(f"📋 Xem danh sách {len(exact_matches)} sản phẩm khớp", expanded=True):
                    st.dataframe(exact_matches[available_cols], use_container_width=True, height=300, column_config=col_config)
            else:
                # Try partial match
                partial_mask = (
                    (df_clean['family'] == family) &
                    (df_clean['stone_color_type'] == stone_color) &
                    (df_clean['charge_unit'] == charge_unit)
                )
                partial_matches = df_clean[partial_mask]
                
                if len(partial_matches) > 0:
                    st.info(f"⚠️ Không tìm thấy sản phẩm khớp chính xác kích thước. Có **{len(partial_matches)}** sản phẩm cùng loại/màu/đơn vị.")
                    match_prices = partial_matches['sales_price']
                    st.caption(f"Giá tham khảo: ${match_prices.min():,.2f} - ${match_prices.max():,.2f} (TB: ${match_prices.mean():,.2f})")
                    
                    # Show partial matches with all fields
                    display_cols = [
                        'contract_product_name', 'contract_name', 'account_code',
                        'sku', 'processing_code', 'processing_name',
                        'stone_color_type', 'family', 'segment',
                        'length_cm', 'width_cm', 'height_cm',
                        'charge_unit', 'sales_price', 'price_m3',
                        'created_date', 'fy_year'
                    ]
                    available_cols = [col for col in display_cols if col in partial_matches.columns]
                    col_config = {
                        'sku': st.column_config.TextColumn('SKU'),
                        'processing_code': st.column_config.TextColumn('Main Processing Code'),
                        'processing_name': st.column_config.TextColumn('Main Processing'),
                    }
                    with st.expander(f"📋 Xem danh sách {len(partial_matches)} sản phẩm liên quan"):
                        st.dataframe(partial_matches.head(50)[available_cols], use_container_width=True, height=300, column_config=col_config)
                else:
                    st.info("⚠️ Không tìm thấy sản phẩm tương tự trong hệ thống.")
        
        elif predict_btn and st.session_state.model is None:
            st.warning("⚠️ Vui lòng huấn luyện model trước khi dự đoán (nút 🤖 ở sidebar)")
        elif not predict_btn:
            pass  # User hasn't clicked predict yet
    
    # Tab 2: Data Analysis
    with tab2:
        st.subheader("📊 Phân tích dữ liệu giá")
        
        df = st.session_state.data.copy()
        
        # Clean data: remove products with price 0, missing, or negative
        df_clean = df[df['sales_price'].notna() & (df['sales_price'] > 0)]
        
        # Show data quality info
        total_products = len(df)
        valid_products = len(df_clean)
        excluded_products = total_products - valid_products
        
        if excluded_products > 0:
            st.info(f"📊 Đã loại bỏ {excluded_products:,} sản phẩm có giá = 0, âm hoặc thiếu. Phân tích với {valid_products:,} / {total_products:,} sản phẩm.")
        
        # Summary metrics using sales_price (clean data)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📦 Sản phẩm hợp lệ", f"{valid_products:,}")
        with col2:
            st.metric("💰 Giá TB (Sales Price)", f"${df_clean['sales_price'].mean():,.2f}")
        with col3:
            st.metric("📈 Giá cao nhất", f"${df_clean['sales_price'].max():,.2f}")
        with col4:
            st.metric("📉 Giá thấp nhất", f"${df_clean['sales_price'].min():,.2f}")
        
        st.divider()
        
        # Charts
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            # Price distribution by segment (using clean data)
            segment_counts = df_clean['segment'].value_counts()
            fig_segment = px.pie(
                values=segment_counts.values,
                names=segment_counts.index,
                title="Phân bố theo phân khúc",
                color=segment_counts.index,
                color_discrete_map={
                    'Super premium': '#9e7cc1',
                    'Premium': '#ff6b6b',
                    'Common': '#ffd93d',
                    'Economy': '#6bcb77'
                }
            )
            st.plotly_chart(fig_segment, use_container_width=True)
        
        with chart_col2:
            # Average sales_price by family (using clean data)
            avg_by_family = df_clean.groupby('family')['sales_price'].mean().sort_values(ascending=True)
            fig_family = px.bar(
                x=avg_by_family.values,
                y=avg_by_family.index,
                orientation='h',
                title="Giá bán trung bình theo loại sản phẩm",
                labels={'x': 'Sales Price (USD)', 'y': 'Loại sản phẩm'}
            )
            fig_family.update_traces(marker_color='#667eea')
            st.plotly_chart(fig_family, use_container_width=True)
        
        # Price by stone type (using clean data)
        st.markdown("#### 💎 Giá bán theo loại đá")
        fig_stone = px.box(
            df_clean,
            x='stone_color_type',
            y='sales_price',
            color='stone_color_type',
            title="Phân bố giá bán theo màu đá",
            labels={'sales_price': 'Sales Price (USD)', 'stone_color_type': 'Stone Color Type'}
        )
        st.plotly_chart(fig_stone, use_container_width=True)
        
        # Price vs dimensions (using clean data)
        st.markdown("#### 📐 Giá bán theo kích thước")
        fig_scatter = px.scatter(
            df_clean,
            x='volume_m3',
            y='sales_price',
            color='segment',
            size='height_cm',
            hover_data=['contract_product_name', 'family'],
            title="Sales Price vs Thể tích",
            labels={'sales_price': 'Sales Price (USD)', 'volume_m3': 'Volume (m³)'},
            color_discrete_map={
                'Super premium': '#9e7cc1',
                'Premium': '#ff6b6b',
                'Common': '#ffd93d',
                'Economy': '#6bcb77'
            }
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Tab 3: Similar Products
    with tab3:
        st.subheader("🔍 Tìm sản phẩm tương tự")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### Tiêu chí tìm kiếm")
            search_family = st.selectbox("Loại sản phẩm", [''] + PRODUCT_FAMILIES, key='search_family')
            search_stone = st.selectbox("Màu đá", [''] + STONE_COLOR_TYPES, key='search_stone')
            
            # Processing code dropdown
            search_processing = st.selectbox(
                "Main Processing",
                options=[code for code, name in PROCESSING_CODES],
                format_func=lambda x: f"{x} - {dict(PROCESSING_CODES).get(x, 'All')}" if x else "All",
                key='search_processing'
            )
            
            search_col1, search_col2, search_col3 = st.columns(3)
            with search_col1:
                search_length = st.number_input("Dài (cm)", min_value=0, value=30, key='search_l')
            with search_col2:
                search_width = st.number_input("Rộng (cm)", min_value=0, value=30, key='search_w')
            with search_col3:
                search_height = st.number_input("Dày (cm)", min_value=0.0, value=3.0, key='search_h')
            
            st.divider()
            
            # Show related checkbox and slider
            show_related = st.checkbox("📋 Hiển thị sản phẩm liên quan", value=False, 
                                       help="Hiển thị các sản phẩm có đặc điểm tương tự nếu không tìm thấy kết quả chính xác")
            
            if show_related:
                related_count = st.slider("Số sản phẩm liên quan", 5, 50, 20)
            
            search_btn = st.button("🔍 Tìm kiếm", type="primary", use_container_width=True)
        
        with col2:
            if search_btn:
                df = st.session_state.data.copy()
                
                # Clean data for searching
                df_clean = df[df['sales_price'].notna() & (df['sales_price'] > 0)].copy()
                df_clean = df_clean.reset_index(drop=True)  # Reset index to avoid alignment issues
                
                # Step 1: Find EXACT matches
                exact_mask = pd.Series([True] * len(df_clean), index=df_clean.index)
                
                if search_family:
                    exact_mask &= df_clean['family'] == search_family
                if search_stone:
                    exact_mask &= df_clean['stone_color_type'] == search_stone
                if search_processing and 'processing_code' in df_clean.columns:
                    exact_mask &= df_clean['processing_code'] == search_processing
                if search_length > 0:
                    exact_mask &= df_clean['length_cm'] == search_length
                if search_width > 0:
                    exact_mask &= df_clean['width_cm'] == search_width
                if search_height > 0:
                    exact_mask &= df_clean['height_cm'] == search_height
                
                exact_matches = df_clean[exact_mask]
                
                # Include processing columns in display - SKU and processing codes together
                display_cols = ['contract_product_name', 'family', 'stone_color_type', 
                                'sku', 'processing_code', 'processing_name',
                                'length_cm', 'width_cm', 'height_cm', 'charge_unit', 'sales_price', 'price_m3', 'segment']
                available_cols = [col for col in display_cols if col in df_clean.columns]
                
                # Column config for English headers
                col_config = {
                    'sku': st.column_config.TextColumn('SKU'),
                    'processing_code': st.column_config.TextColumn('Main Processing Code'),
                    'processing_name': st.column_config.TextColumn('Main Processing'),
                }
                
                # Display exact matches
                if len(exact_matches) > 0:
                    st.markdown(f"#### ✅ Tìm thấy {len(exact_matches)} sản phẩm khớp chính xác")
                    st.dataframe(exact_matches[available_cols], use_container_width=True, height=300, column_config=col_config)
                    
                    # Statistics for exact matches
                    valid_prices = exact_matches['sales_price']
                    if len(valid_prices) > 0:
                        st.markdown("##### 📊 Thống kê giá (khớp chính xác)")
                        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                        with stat_col1:
                            st.metric("Thấp nhất", f"${valid_prices.min():,.2f}")
                        with stat_col2:
                            st.metric("Cao nhất", f"${valid_prices.max():,.2f}")
                        with stat_col3:
                            st.metric("Trung bình", f"${valid_prices.mean():,.2f}")
                        with stat_col4:
                            st.metric("Trung vị", f"${valid_prices.median():,.2f}")
                else:
                    st.warning("⚠️ Không tìm thấy sản phẩm khớp chính xác với tiêu chí.")
                
                # Step 2: Show related products if checkbox is checked
                if show_related:
                    st.divider()
                    st.markdown(f"#### 🔗 Sản phẩm liên quan (top {related_count})")
                    
                    # Find related products based on partial criteria
                    related_mask = pd.Series([False] * len(df_clean), index=df_clean.index)
                    
                    if search_family:
                        related_mask |= df_clean['family'] == search_family
                    if search_stone:
                        related_mask |= df_clean['stone_color_type'] == search_stone
                    if search_processing and 'processing_code' in df_clean.columns:
                        related_mask |= df_clean['processing_code'] == search_processing
                    
                    # Exclude exact matches
                    related_mask &= ~exact_mask
                    
                    related_products = df_clean[related_mask].copy()
                    
                    # Sort by dimension similarity if dimensions provided
                    if search_length > 0 and search_width > 0 and search_height > 0:
                        related_products['dim_diff'] = (
                            abs(related_products['length_cm'] - search_length) +
                            abs(related_products['width_cm'] - search_width) +
                            abs(related_products['height_cm'] - search_height)
                        )
                        related_products = related_products.nsmallest(related_count, 'dim_diff')
                    else:
                        related_products = related_products.head(related_count)
                    
                    if len(related_products) > 0:
                        st.dataframe(related_products[available_cols], use_container_width=True, height=300, column_config=col_config)
                        
                        # Statistics for related products
                        valid_prices = related_products['sales_price']
                        if len(valid_prices) > 0:
                            st.markdown("##### 📊 Thống kê giá (sản phẩm liên quan)")
                            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                            with stat_col1:
                                st.metric("Thấp nhất", f"${valid_prices.min():,.2f}")
                            with stat_col2:
                                st.metric("Cao nhất", f"${valid_prices.max():,.2f}")
                            with stat_col3:
                                st.metric("Trung bình", f"${valid_prices.mean():,.2f}")
                            with stat_col4:
                                st.metric("Trung vị", f"${valid_prices.median():,.2f}")
                            
                            # Summary
                            price_range = valid_prices.max() - valid_prices.min()
                            st.caption(f"Khoảng giá: ${price_range:,.2f} | Độ lệch chuẩn: ${valid_prices.std():,.2f}")
                    else:
                        st.info("Không tìm thấy sản phẩm liên quan.")
    
    # Tab 4: Model Performance
    with tab4:
        st.subheader("📈 Hiệu suất Model ML")
        
        if st.session_state.model_metrics is not None:
            metrics = st.session_state.model_metrics
            
            # Show training info with data cleaning details
            removed = metrics.get('removed_samples', 0)
            n_est = metrics.get('n_estimators_used', 'N/A')
            st.info(f"📊 Model huấn luyện với **{metrics.get('train_samples', 'N/A'):,}** mẫu hợp lệ ({removed:,} mẫu bị loại bỏ). Dự đoán: **{metrics.get('target_col', 'sales_price')}**")
            
            # Main metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("MAE", f"${metrics['mae']:,.2f}", help="Mean Absolute Error - Sai số tuyệt đối trung bình")
            with col2:
                st.metric("R² Score", f"{metrics['r2']:.3f}", help="Coefficient of Determination - Càng gần 1 càng tốt")
            with col3:
                st.metric("CV MAE Mean", f"${metrics['cv_mae_mean']:,.2f}", help="Cross-Validation MAE")
            with col4:
                cv_r2 = metrics.get('cv_r2_mean', metrics['r2'])
                st.metric("CV R² Mean", f"{cv_r2:.3f}", help="Cross-Validation R² - Đánh giá tổng quát")
            
            st.divider()
            
            # Feature importance
            st.markdown("#### 🎯 Độ quan trọng của các đặc trưng")
            importance = st.session_state.model.get_feature_importance()
            
            fig_importance = px.bar(
                importance,
                x='importance',
                y='feature',
                orientation='h',
                title="Feature Importance"
            )
            fig_importance.update_traces(marker_color='#764ba2')
            st.plotly_chart(fig_importance, use_container_width=True)
            
            # Model info
            with st.expander("ℹ️ Thông tin Model"):
                st.markdown(f"""
                **Mục tiêu dự đoán:** `sales_price` (Giá bán theo đơn vị charge_unit)
                
                **Thuật toán:** Gradient Boosting Regressor (Early Stopping)
                
                **Hyperparameters (Optimized):**
                - n_estimators: 200 (thực tế dùng: {n_est})
                - learning_rate: 0.05
                - max_depth: 4
                - min_samples_split: 10
                - min_samples_leaf: 5
                - subsample: 0.8 (Stochastic GB)
                - max_features: sqrt
                - early stopping: 10 iterations
                
                **Features:**
                - Categorical: family, stone_color_type, charge_unit, **processing_code** (Main Processing)
                - Numerical: length_cm, width_cm, height_cm, volume_m3, area_m2
                
                > ⚠️ **Data Leakage Prevention:** `segment` đã được loại bỏ khỏi features vì nó được 
                > tính từ giá (price_m3). Việc dùng segment làm feature sẽ gây ra data leakage.
                
                **Data Cleaning:**
                - Loại bỏ giá = 0, âm, hoặc missing
                - Loại bỏ outliers (ngoài 1st-99th percentile)
                - Loại bỏ rows có missing values trong features
                - `processing_code` rỗng/Unknown được chuyển thành 'OTHER'
                """)
        else:
            st.info("👈 Huấn luyện model để xem hiệu suất (nút 🤖 ở sidebar)")
    
    # Tab 5: Detailed Data
    with tab5:
        st.subheader("📋 Dữ liệu chi tiết")
        
        # Filters
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        with filter_col1:
            filter_family = st.multiselect("Loại sản phẩm", PRODUCT_FAMILIES)
        with filter_col2:
            filter_segment = st.multiselect("Phân khúc", ['Economy', 'Common', 'Premium', 'Super premium'])
        with filter_col3:
            price_range = st.slider("Khoảng giá (USD/m³)", 0, 2000, (0, 2000))
        
        # Apply filters
        filtered_df = st.session_state.data.copy()
        if filter_family:
            filtered_df = filtered_df[filtered_df['family'].isin(filter_family)]
        if filter_segment:
            filtered_df = filtered_df[filtered_df['segment'].isin(filter_segment)]
        filtered_df = filtered_df[
            (filtered_df['price_m3'] >= price_range[0]) & 
            (filtered_df['price_m3'] <= price_range[1])
        ]
        
        st.markdown(f"**Hiển thị {len(filtered_df):,} / {len(st.session_state.data):,} sản phẩm**")
        
        # Define all columns from the contract query in logical order
        # These match the fields from contract_query.txt and salesforce_loader.py
        all_contract_columns = [
            'contract_product_name',   # Name
            'contract_name',           # Contract__r.Name
            'account_code',            # Account_Code_C__c
            'stone_color_type',        # Product__r.STONE_Color_Type__c
            'sku',                     # Product__r.StockKeepingUnit (SKU)
            'processing_code',         # Main processing code (from SKU)
            'processing_name',         # Main processing name (English)
            'family',                  # Product__r.Family
            'segment',                 # Segment__c
            'created_date',            # Created_Date__c
            'fy_year',                 # Fiscal Year (calculated)
            'product_description',     # Product_Discription__c
            'product_description_vn',  # Product__r.Product_description_in_Vietnamese__c
            'length_cm',               # Length__c
            'width_cm',                # Width__c
            'height_cm',               # Height__c
            'quantity',                # Quantity__c
            'crates',                  # Crates__c
            'm2',                      # m2__c
            'm3',                      # m3__c
            'ml',                      # ml__c
            'tons',                    # Tons__c
            'sales_price',             # Sales_Price__c
            'charge_unit',             # Charge_Unit__c
            'total_price_usd',         # Total_Price_USD__c
            'price_m3',                # Calculated price per m3
        ]
        
        # Filter to only columns that exist in the dataframe
        available_columns = [col for col in all_contract_columns if col in filtered_df.columns]
        
        # Add any remaining columns not in the predefined list
        remaining_columns = [col for col in filtered_df.columns if col not in available_columns]
        display_columns = available_columns + remaining_columns
        
        # Column configuration for English headers on specific columns
        column_config = {
            'sku': st.column_config.TextColumn('SKU', help='Product Stock Keeping Unit'),
            'processing_code': st.column_config.TextColumn('Main Processing Code', help='Ký hiệu gia công chính'),
            'processing_name': st.column_config.TextColumn('Main Processing', help='Nhóm mã gia công chính'),
        }
        
        # Display data with all columns
        st.dataframe(
            filtered_df[display_columns],
            use_container_width=True,
            height=500,
            column_config=column_config
        )
        
        # Download button
        csv = filtered_df[display_columns].to_csv(index=False)
        st.download_button(
            "📥 Tải xuống CSV",
            csv,
            "stone_price_data.csv",
            "text/csv",
            use_container_width=True
        )


if __name__ == "__main__":
    main()
