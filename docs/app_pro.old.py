"""
Stone Price Predictor - Enhanced Version with Salesforce Integration
Dự đoán giá sản phẩm đá tự nhiên với ML và Salesforce API
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
from salesforce_loader import SalesforceDataLoader, fetch_salesforce_data_for_prediction

# ============ Configuration ============
st.set_page_config(
    page_title="Stone Price Predictor Pro",
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
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .segment-badge {
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .segment-super { background: #9e7cc1; color: white; }
    .segment-premium { background: #ff6b6b; color: white; }
    .segment-common { background: #ffd93d; color: #333; }
    .segment-economy { background: #6bcb77; color: white; }
    .price-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    .customer-card {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 15px;
        margin: 5px 0;
    }
    .customer-card:hover {
        border-color: #667eea;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
    }
</style>
""", unsafe_allow_html=True)


# ============ Constants ============
SEGMENT_THRESHOLDS = {
    'Super premium': 1500,
    'Premium': 800,
    'Common': 400,
    'Economy': 0
}

CUSTOMER_PRICING_RULES = {
    'A': {
        'name': 'Khách thân thiết đặc biệt',
        'years': '>10 năm', 
        'volume': '50-150 containers',
        'discount_range': (-0.03, -0.015),
        'adjustment': 'Bớt 1.5-3% so với giá B',
        'examples': ['X19', 'X39']
    },
    'B': {
        'name': 'Khách lớn, chuyên nghiệp',
        'years': '3-10 năm',
        'volume': '20-50 containers, >100.000 USD/năm',
        'discount_range': (-0.04, -0.02),
        'adjustment': 'Thấp hơn C: 10-30 USD/m³',
        'examples': ['X21.2', 'X36', 'X49', 'X26', 'X27']
    },
    'C': {
        'name': 'Khách hàng phổ thông',
        'years': '1-5 năm',
        'volume': '5-20 containers',
        'discount_range': (0, 0),
        'adjustment': 'Giá sản phẩm phổ thông chuẩn',
        'examples': ['X17', 'X69', 'X59', 'X44', 'X45']
    },
    'D': {
        'name': 'Khách mới, size nhỏ',
        'years': '1 năm',
        'volume': '1-10 containers',
        'discount_range': (0.03, 0.06),
        'adjustment': 'Cao hơn C: 15-45 USD/m³',
        'examples': ['X91', 'X77', 'X46', 'X23.1']
    },
    'E': {
        'name': 'Sản phẩm mới, cao cấp',
        'years': '1 năm',
        'volume': '1-10 containers',
        'discount_range': (0.05, 0.10),
        'adjustment': 'Giá sản phẩm mới, cao cấp',
        'examples': ['X66', 'X11.2']
    },
    'F': {
        'name': 'Khách hàng dự án',
        'years': '1-5 năm',
        'volume': '1-50 containers',
        'discount_range': (-0.02, 0.02),
        'adjustment': 'Hàng lẻ, dự án, gia công phức tạp',
        'examples': ['X79', 'X87']
    }
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


# ============ ML Model ============
class StonePricePredictor:
    def __init__(self):
        self.model = None
        self.encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.categorical_columns = ['family', 'stone_class', 'stone_color_type', 'charge_unit']
        self.numerical_columns = ['length_cm', 'width_cm', 'height_cm', 'volume_m3', 'area_m2']
        
    def prepare_features(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        features = df.copy()
        
        for col in self.categorical_columns:
            if col in features.columns:
                if fit:
                    self.encoders[col] = LabelEncoder()
                    features[f'{col}_encoded'] = self.encoders[col].fit_transform(
                        features[col].fillna('Unknown').astype(str)
                    )
                else:
                    features[f'{col}_encoded'] = features[col].fillna('Unknown').apply(
                        lambda x: self.encoders[col].transform([str(x)])[0] 
                        if str(x) in self.encoders[col].classes_ else -1
                    )
        
        encoded_cols = [f'{col}_encoded' for col in self.categorical_columns]
        self.feature_columns = self.numerical_columns + encoded_cols
        
        X = features[self.feature_columns].fillna(0).values
        
        if fit:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)
            
        return X
    
    def train(self, df: pd.DataFrame, target_col: str = 'price_m3') -> Dict[str, float]:
        # Filter valid data
        valid_df = df.dropna(subset=[target_col])
        valid_df = valid_df[valid_df[target_col] > 0]
        
        if len(valid_df) < 50:
            raise ValueError(f"Not enough valid data. Got {len(valid_df)} records, need at least 50.")
        
        X = self.prepare_features(valid_df, fit=True)
        y = valid_df[target_col].values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='neg_mean_absolute_error')
        
        return {
            'mae': mae,
            'r2': r2,
            'cv_mae_mean': -cv_scores.mean(),
            'cv_mae_std': cv_scores.std(),
            'n_samples': len(valid_df)
        }
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained")
        X = self.prepare_features(df, fit=False)
        return self.model.predict(X)
    
    def get_feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            return pd.DataFrame()
        return pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)


# ============ Helper Functions ============
@st.cache_data(ttl=3600)
def generate_demo_data(n_samples: int = 500) -> pd.DataFrame:
    """Generate demo data for testing."""
    np.random.seed(42)
    
    data = []
    for i in range(n_samples):
        family = np.random.choice(PRODUCT_FAMILIES)
        stone_class = np.random.choice(STONE_CLASSES)
        stone_color = np.random.choice(STONE_COLOR_TYPES)
        
        length = np.random.choice([10, 15, 20, 30, 40, 50, 60, 80, 100, 120])
        width = np.random.choice([5, 8, 10, 15, 20, 30, 40, 60])
        height = np.random.choice([2, 2.5, 3, 5, 6, 7, 8, 10, 12, 15, 20])
        
        volume_m3 = (length * width * height) / 1000000
        area_m2 = (length * width) / 10000
        
        base_price_m3 = 350 + np.random.normal(0, 50)
        
        if family in ['STAIR', 'ART', 'High-Class']:
            base_price_m3 *= 2.5
        elif family in ['Interior_Tiles', 'SLAB']:
            base_price_m3 *= 1.8
        elif family == 'Exterior_Tiles':
            base_price_m3 *= 1.2
            
        if stone_color in ['ABSOLUTE BASALT', 'WHITE MARBLE']:
            base_price_m3 *= 1.5
        elif stone_color in ['YELLOW MARBLE', 'RED GRANITE']:
            base_price_m3 *= 1.3
            
        if length <= 15 and width <= 15:
            base_price_m3 *= 1.4
        elif length >= 60 or width >= 60:
            base_price_m3 *= 0.9
            
        if height <= 2:
            base_price_m3 *= 2.0
        elif height >= 10:
            base_price_m3 *= 0.85
            
        price_m3 = max(200, base_price_m3 + np.random.normal(0, 80))
        
        if price_m3 >= 1500:
            segment = 'Super premium'
        elif price_m3 >= 800:
            segment = 'Premium'
        elif price_m3 >= 400:
            segment = 'Common'
        else:
            segment = 'Economy'
            
        if family in ['PALISADE', 'STAIR']:
            charge_unit = 'USD/ML'
        elif height <= 3:
            charge_unit = 'USD/M2'
        elif length <= 20 and width <= 20:
            charge_unit = 'USD/PC'
        else:
            charge_unit = np.random.choice(['USD/M3', 'USD/TON'])
            
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
            'list_price': round(price_m3 * volume_m3 if charge_unit == 'USD/PC' else price_m3 * height/100 if charge_unit == 'USD/M2' else price_m3, 2),
            'price_m3': round(price_m3, 2),
            'segment': segment,
            'data_source': 'demo'
        })
    
    return pd.DataFrame(data)


def classify_segment(price_m3: float) -> str:
    if price_m3 >= 1500:
        return 'Super premium'
    elif price_m3 >= 800:
        return 'Premium'
    elif price_m3 >= 400:
        return 'Common'
    else:
        return 'Economy'


def get_segment_style(segment: str) -> Tuple[str, str]:
    styles = {
        'Super premium': ('#9e7cc1', 'white'),
        'Premium': ('#ff6b6b', 'white'),
        'Common': ('#ffd93d', '#333'),
        'Economy': ('#6bcb77', 'white')
    }
    return styles.get(segment, ('#808080', 'white'))


def calculate_customer_prices(base_price: float, segment: str) -> Dict[str, Dict]:
    """Calculate prices for all customer types."""
    results = {}
    
    # Define adjustment ranges based on segment
    segment_adjustments = {
        'Economy': {'tolerance': 10.0},
        'Common': {'tolerance': 15.0},
        'Premium': {'tolerance': 20.0},
        'Super premium': {'tolerance': 30.0}
    }
    
    for customer_type, info in CUSTOMER_PRICING_RULES.items():
        discount_range = info['discount_range']
        min_discount, max_discount = discount_range
        
        min_price = round(base_price * (1 + min_discount), 2)
        max_price = round(base_price * (1 + max_discount), 2)
        avg_price = round((min_price + max_price) / 2, 2)
        
        results[customer_type] = {
            'name': info['name'],
            'min_price': min_price,
            'max_price': max_price,
            'avg_price': avg_price,
            'adjustment': info['adjustment'],
            'examples': info['examples']
        }
    
    return results


def convert_price_to_unit(price_m3: float, charge_unit: str, 
                          length: float, width: float, height: float,
                          stone_class: str = 'BASALT') -> float:
    """Convert price/m³ to specified unit."""
    volume_m3 = (length * width * height) / 1000000
    specific_gravity = 2.8 if stone_class == 'BASALT' else 2.65
    
    if charge_unit == 'USD/M2':
        return price_m3 * height / 100
    elif charge_unit == 'USD/PC':
        return price_m3 * volume_m3
    elif charge_unit == 'USD/TON':
        return price_m3 / (specific_gravity * 1.1)
    elif charge_unit == 'USD/ML':
        return price_m3 * width * height / 10000
    else:
        return price_m3


def find_similar_products(df: pd.DataFrame, query: Dict, top_n: int = 10) -> pd.DataFrame:
    """Find products similar to the query."""
    mask = pd.Series([True] * len(df))
    
    if query.get('stone_class'):
        mask &= df['stone_class'] == query['stone_class']
    if query.get('family'):
        mask &= df['family'] == query['family']
    
    filtered_df = df[mask].copy()
    
    if len(filtered_df) == 0:
        return pd.DataFrame()
    
    # Calculate similarity score
    if all(k in query for k in ['length_cm', 'width_cm', 'height_cm']):
        filtered_df['similarity_score'] = (
            abs(filtered_df['length_cm'] - query['length_cm']) * 0.3 +
            abs(filtered_df['width_cm'] - query['width_cm']) * 0.3 +
            abs(filtered_df['height_cm'] - query['height_cm']) * 0.4
        )
        filtered_df = filtered_df.nsmallest(top_n, 'similarity_score')
    else:
        filtered_df = filtered_df.head(top_n)
    
    return filtered_df


# ============ Main App ============
def main():
    # Header
    st.markdown('<h1 class="main-header">💎 Stone Price Predictor Pro</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; margin-bottom: 2rem;">Dự đoán giá sản phẩm đá tự nhiên với AI & Salesforce</p>', unsafe_allow_html=True)
    
    # Session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'model_metrics' not in st.session_state:
        st.session_state.model_metrics = None
    if 'data_source' not in st.session_state:
        st.session_state.data_source = None
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/gemstone.png", width=80)
        st.title("⚙️ Cấu hình")
        
        # Data source
        st.markdown("### 📊 Nguồn dữ liệu")
        data_source = st.radio(
            "Chọn nguồn dữ liệu:",
            ["Demo (Dữ liệu mẫu)", "Salesforce (Dữ liệu thực)"],
            help="Demo: Sử dụng dữ liệu mẫu để test\nSalesforce: Kết nối Salesforce CRM thực"
        )
        
        if st.button("🔄 Tải dữ liệu", use_container_width=True, type="primary"):
            with st.spinner("Đang tải dữ liệu..."):
                try:
                    if "Demo" in data_source:
                        st.session_state.data = generate_demo_data(500)
                        st.session_state.data_source = "demo"
                        st.success(f"✅ Đã tải {len(st.session_state.data)} sản phẩm (Demo)")
                    else:
                        result = fetch_salesforce_data_for_prediction()
                        if result.get("success"):
                            st.session_state.data = result.get("combined_data")
                            st.session_state.data_source = "salesforce"
                            st.success(f"✅ Đã tải {len(st.session_state.data)} sản phẩm từ Salesforce")
                        else:
                            st.error(f"❌ Lỗi: {result.get('error')}")
                except Exception as e:
                    st.error(f"❌ Lỗi: {str(e)}")
                    st.info("💡 Sử dụng Demo để test")
        
        if st.session_state.data is not None:
            st.divider()
            st.markdown("### 🤖 Machine Learning")
            
            if st.button("🎯 Huấn luyện Model", use_container_width=True):
                with st.spinner("Đang huấn luyện..."):
                    try:
                        predictor = StonePricePredictor()
                        metrics = predictor.train(st.session_state.data)
                        st.session_state.model = predictor
                        st.session_state.model_metrics = metrics
                        st.success(f"✅ Model sẵn sàng!\nMAE: ${metrics['mae']:.2f}")
                    except Exception as e:
                        st.error(f"❌ Lỗi: {str(e)}")
            
            if st.session_state.model_metrics:
                metrics = st.session_state.model_metrics
                st.metric("MAE", f"${metrics['mae']:.0f}")
                st.metric("R² Score", f"{metrics['r2']:.2f}")
        
        st.divider()
        
        # Info
        with st.expander("📋 Phân khúc giá"):
            st.markdown("""
            | Phân khúc | USD/m³ |
            |-----------|--------|
            | 🟣 Super Premium | ≥ 1,500 |
            | 🔴 Premium | ≥ 800 |
            | 🟡 Common | ≥ 400 |
            | 🟢 Economy | < 400 |
            """)
    
    # Main content
    if st.session_state.data is None:
        st.info("👈 Vui lòng tải dữ liệu từ sidebar để bắt đầu")
        
        # Show pricing matrix
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📊 Ma trận giá theo loại sản phẩm")
            st.dataframe(pd.DataFrame({
                'Loại SP': ['Đá lát mỏng 1-1.5cm', 'Đá nội ngoại thất 2-5cm', 'Đá bậc thang', 'Đá cây palisade', 'Đá mỹ nghệ'],
                'Economy': ['Đá mẻ', 'Cơ bản', '-', 'Cưa lột', 'Cơ bản'],
                'Common': ['1 mặt đốt', 'Thông dụng', 'Nguyên khối', 'Đốt chải', 'Trung bình'],
                'Premium': ['Đặc biệt', 'Cao cấp', 'Ốp bậc', 'Nhiều mặt', 'Cao cấp'],
                'Super': ['Siêu mỏng', 'Hồ bơi', 'Đặc biệt', 'Đặc biệt', 'Đặc biệt']
            }), use_container_width=True)
        
        with col2:
            st.markdown("### 👥 Phân loại khách hàng")
            for code, info in CUSTOMER_PRICING_RULES.items():
                st.markdown(f"**{code}** - {info['name']}")
                st.caption(f"{info['adjustment']}")
        return
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔮 Dự đoán giá",
        "📊 Phân tích",
        "🔍 Tìm SP tương tự",
        "📋 Dữ liệu"
    ])
    
    # Tab 1: Price Prediction
    with tab1:
        st.markdown("## 🔮 Dự đoán giá sản phẩm")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📝 Thông tin sản phẩm")
            
            family = st.selectbox("Loại sản phẩm", PRODUCT_FAMILIES)
            stone_class = st.selectbox("Loại đá", STONE_CLASSES)
            stone_color = st.selectbox("Màu đá", STONE_COLOR_TYPES)
            
            dim_cols = st.columns(3)
            with dim_cols[0]:
                length = st.number_input("Dài (cm)", 1, 300, 30)
            with dim_cols[1]:
                width = st.number_input("Rộng (cm)", 1, 300, 30)
            with dim_cols[2]:
                height = st.number_input("Dày (cm)", 0.5, 50.0, 3.0, step=0.5)
            
            charge_unit = st.selectbox("Đơn vị tính giá", CHARGE_UNITS)
            
            predict_btn = st.button("🎯 DỰ ĐOÁN GIÁ", type="primary", use_container_width=True)
        
        with col2:
            if predict_btn:
                if st.session_state.model is None:
                    st.warning("⚠️ Vui lòng huấn luyện model trước!")
                else:
                    # Prepare input
                    volume_m3 = (length * width * height) / 1000000
                    area_m2 = (length * width) / 10000
                    
                    input_df = pd.DataFrame([{
                        'family': family,
                        'stone_class': stone_class,
                        'stone_color_type': stone_color,
                        'length_cm': length,
                        'width_cm': width,
                        'height_cm': height,
                        'volume_m3': volume_m3,
                        'area_m2': area_m2,
                        'charge_unit': charge_unit
                    }])
                    
                    # Predict
                    predicted_price_m3 = st.session_state.model.predict(input_df)[0]
                    segment = classify_segment(predicted_price_m3)
                    bg_color, text_color = get_segment_style(segment)
                    
                    unit_price = convert_price_to_unit(
                        predicted_price_m3, charge_unit, 
                        length, width, height, stone_class
                    )
                    
                    # Display results
                    st.markdown("### 📊 Kết quả dự đoán")
                    
                    # Segment badge
                    st.markdown(f"""
                    <div style="background: {bg_color}; color: {text_color}; 
                                padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
                        <span style="font-size: 1.5rem;">Phân khúc: <b>{segment}</b></span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Price metrics
                    price_cols = st.columns(2)
                    with price_cols[0]:
                        st.markdown(f"""
                        <div class="price-card">
                            <div class="metric-label">💰 Giá m³</div>
                            <div class="metric-value">${predicted_price_m3:,.0f}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with price_cols[1]:
                        st.markdown(f"""
                        <div class="price-card">
                            <div class="metric-label">📦 Giá/{charge_unit.split('/')[1]}</div>
                            <div class="metric-value">${unit_price:,.2f}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Customer prices
                    st.markdown("### 👥 Giá theo loại khách hàng")
                    customer_prices = calculate_customer_prices(unit_price, segment)
                    
                    for code in ['A', 'B', 'C', 'D', 'E', 'F']:
                        info = customer_prices[code]
                        with st.container():
                            cust_cols = st.columns([1, 2, 2, 2])
                            with cust_cols[0]:
                                st.markdown(f"**{code}**")
                            with cust_cols[1]:
                                st.markdown(f"{info['name']}")
                            with cust_cols[2]:
                                st.markdown(f"${info['min_price']:,.2f} - ${info['max_price']:,.2f}")
                            with cust_cols[3]:
                                st.caption(f"{info['adjustment']}")
                    
                    # Gauge chart
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=predicted_price_m3,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "USD/m³", 'font': {'size': 16}},
                        delta={'reference': 600, 'increasing': {'color': "red"}},
                        gauge={
                            'axis': {'range': [0, 2000], 'tickwidth': 1},
                            'bar': {'color': bg_color},
                            'steps': [
                                {'range': [0, 400], 'color': '#e8f5e9'},
                                {'range': [400, 800], 'color': '#fff9c4'},
                                {'range': [800, 1500], 'color': '#ffcdd2'},
                                {'range': [1500, 2000], 'color': '#e1bee7'}
                            ],
                            'threshold': {
                                'line': {'color': "black", 'width': 4},
                                'thickness': 0.75,
                                'value': predicted_price_m3
                            }
                        }
                    ))
                    fig.update_layout(height=250, margin={'t': 40, 'b': 0, 'l': 30, 'r': 30})
                    st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Analysis
    with tab2:
        st.markdown("## 📊 Phân tích dữ liệu")
        
        df = st.session_state.data
        
        # Metrics
        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.metric("📦 Sản phẩm", f"{len(df):,}")
        with metric_cols[1]:
            st.metric("💰 Giá TB", f"${df['price_m3'].mean():,.0f}/m³")
        with metric_cols[2]:
            st.metric("📈 Cao nhất", f"${df['price_m3'].max():,.0f}/m³")
        with metric_cols[3]:
            st.metric("📉 Thấp nhất", f"${df['price_m3'].min():,.0f}/m³")
        
        st.divider()
        
        chart_cols = st.columns(2)
        with chart_cols[0]:
            segment_counts = df['segment'].value_counts()
            fig = px.pie(
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
            st.plotly_chart(fig, use_container_width=True)
        
        with chart_cols[1]:
            avg_price = df.groupby('family')['price_m3'].mean().sort_values()
            fig = px.bar(
                x=avg_price.values,
                y=avg_price.index,
                orientation='h',
                title="Giá TB theo loại SP",
                labels={'x': 'USD/m³', 'y': ''}
            )
            fig.update_traces(marker_color='#667eea')
            st.plotly_chart(fig, use_container_width=True)
        
        # Scatter
        fig = px.scatter(
            df,
            x='height_cm',
            y='price_m3',
            color='segment',
            size='volume_m3',
            hover_data=['product_name', 'family'],
            title="Giá vs Độ dày",
            color_discrete_map={
                'Super premium': '#9e7cc1',
                'Premium': '#ff6b6b',
                'Common': '#ffd93d',
                'Economy': '#6bcb77'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: Similar Products
    with tab3:
        st.markdown("## 🔍 Tìm sản phẩm tương tự")
        
        search_cols = st.columns([1, 2])
        with search_cols[0]:
            s_family = st.selectbox("Loại SP", [''] + PRODUCT_FAMILIES, key='s_family')
            s_stone = st.selectbox("Loại đá", [''] + STONE_CLASSES, key='s_stone')
            
            s_cols = st.columns(3)
            with s_cols[0]:
                s_length = st.number_input("Dài", 0, 300, 30, key='s_l')
            with s_cols[1]:
                s_width = st.number_input("Rộng", 0, 300, 30, key='s_w')
            with s_cols[2]:
                s_height = st.number_input("Dày", 0.0, 50.0, 3.0, key='s_h')
            
            top_n = st.slider("Số kết quả", 3, 20, 10)
            search_btn = st.button("🔍 Tìm kiếm", type="primary")
        
        with search_cols[1]:
            if search_btn:
                query = {
                    'family': s_family if s_family else None,
                    'stone_class': s_stone if s_stone else None,
                    'length_cm': s_length,
                    'width_cm': s_width,
                    'height_cm': s_height
                }
                
                similar = find_similar_products(st.session_state.data, query, top_n)
                
                if len(similar) > 0:
                    st.markdown(f"### Tìm thấy {len(similar)} sản phẩm")
                    
                    display_cols = ['product_name', 'family', 'length_cm', 'width_cm', 
                                    'height_cm', 'price_m3', 'segment']
                    available_cols = [c for c in display_cols if c in similar.columns]
                    
                    st.dataframe(
                        similar[available_cols],
                        use_container_width=True
                    )
                    
                    # Chart
                    fig = px.bar(
                        similar.head(10),
                        x='product_name',
                        y='price_m3',
                        color='segment',
                        title="So sánh giá",
                        color_discrete_map={
                            'Super premium': '#9e7cc1',
                            'Premium': '#ff6b6b',
                            'Common': '#ffd93d',
                            'Economy': '#6bcb77'
                        }
                    )
                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Không tìm thấy sản phẩm phù hợp")
    
    # Tab 4: Data
    with tab4:
        st.markdown("## 📋 Dữ liệu chi tiết")
        
        filter_cols = st.columns(3)
        with filter_cols[0]:
            f_family = st.multiselect("Loại SP", PRODUCT_FAMILIES, key='f_family')
        with filter_cols[1]:
            f_segment = st.multiselect("Phân khúc", ['Economy', 'Common', 'Premium', 'Super premium'], key='f_segment')
        with filter_cols[2]:
            f_price = st.slider("Khoảng giá", 0, 2000, (0, 2000), key='f_price')
        
        filtered = st.session_state.data.copy()
        if f_family:
            filtered = filtered[filtered['family'].isin(f_family)]
        if f_segment:
            filtered = filtered[filtered['segment'].isin(f_segment)]
        filtered = filtered[(filtered['price_m3'] >= f_price[0]) & (filtered['price_m3'] <= f_price[1])]
        
        st.markdown(f"**{len(filtered):,} / {len(st.session_state.data):,} sản phẩm**")
        
        st.dataframe(
            filtered,
            use_container_width=True,
            height=500
        )
        
        st.download_button(
            "📥 Tải CSV",
            filtered.to_csv(index=False),
            "stone_prices.csv",
            "text/csv",
            use_container_width=True
        )


if __name__ == "__main__":
    main()
