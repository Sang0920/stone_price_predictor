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
    """Machine Learning model for stone price prediction."""
    
    def __init__(self):
        self.model = None
        self.encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.categorical_columns = ['family', 'stone_class', 'stone_color_type', 'charge_unit']
        self.numerical_columns = ['length_cm', 'width_cm', 'height_cm', 'volume_m3', 'area_m2']
        
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
        encoded_cols = [f'{col}_encoded' for col in self.categorical_columns]
        self.feature_columns = self.numerical_columns + encoded_cols
        
        X = features[self.feature_columns].values
        
        if fit:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)
            
        return X
    
    def train(self, df: pd.DataFrame, target_col: str = 'price_m3') -> Dict[str, float]:
        """Train the price prediction model."""
        # Prepare features
        X = self.prepare_features(df, fit=True)
        y = df[target_col].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Gradient Boosting model
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='neg_mean_absolute_error')
        
        return {
            'mae': mae,
            'r2': r2,
            'cv_mae_mean': -cv_scores.mean(),
            'cv_mae_std': cv_scores.std()
        }
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict prices for new data."""
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
    
    if query.get('stone_class'):
        mask &= df['stone_class'] == query['stone_class']
    
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
        st.image("https://img.icons8.com/fluency/96/gemstone.png", width=80)
        st.title("⚙️ Cấu hình")
        
        # Data source selection
        data_source = st.selectbox(
            "Nguồn dữ liệu",
            ["Dữ liệu mẫu (Demo)", "Salesforce API (Coming Soon)"]
        )
        
        if st.button("🔄 Tải / Làm mới dữ liệu", use_container_width=True):
            with st.spinner("Đang tải dữ liệu..."):
                st.session_state.data = generate_sample_data(500)
                st.success(f"✅ Đã tải {len(st.session_state.data)} sản phẩm!")
        
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
                # Prepare input data
                volume_m3 = (length * width * height) / 1000000
                area_m2 = (length * width) / 10000
                
                input_data = pd.DataFrame([{
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
                predicted_price_m3 = st.session_state.model.predict(input_data)[0]
                segment = classify_segment(predicted_price_m3)
                
                # Calculate unit price
                if charge_unit == 'USD/M2':
                    unit_price = predicted_price_m3 * height / 100
                elif charge_unit == 'USD/PC':
                    unit_price = predicted_price_m3 * volume_m3
                elif charge_unit == 'USD/TON':
                    specific_gravity = 2.8 if stone_class == 'BASALT' else 2.65
                    unit_price = predicted_price_m3 / (specific_gravity * 1.1)
                elif charge_unit == 'USD/ML':
                    unit_price = predicted_price_m3 * width * height / 10000
                else:
                    unit_price = predicted_price_m3
                
                # Customer price adjustment
                price_info = calculate_customer_price(unit_price, customer_type)
                
                # Display results
                st.markdown("#### 📊 Kết quả dự đoán")
                
                # Segment indicator
                segment_color = get_segment_color(segment)
                st.markdown(f"""
                <div style="background-color: {segment_color}; padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
                    <h3 style="color: white; margin: 0;">Phân khúc: {segment}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Metrics
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric("💰 Giá m³ dự đoán", f"${predicted_price_m3:,.2f}")
                with metric_col2:
                    st.metric(f"📦 Giá/{charge_unit.split('/')[1]}", f"${unit_price:,.2f}")
                
                st.divider()
                
                st.markdown(f"**👤 Giá theo khách hàng loại {customer_type}:**")
                st.markdown(f"- Giá cơ sở: **${price_info['base_price']:,.2f}**")
                st.markdown(f"- Khoảng giá: **${price_info['min_price']:,.2f}** - **${price_info['max_price']:,.2f}**")
                st.markdown(f"- Điều chỉnh: {price_info['adjustment_label']}")
                
                # Price gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=predicted_price_m3,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "USD/m³"},
                    gauge={
                        'axis': {'range': [0, 2000]},
                        'bar': {'color': segment_color},
                        'steps': [
                            {'range': [0, 400], 'color': '#6bcb77'},
                            {'range': [400, 800], 'color': '#ffd93d'},
                            {'range': [800, 1500], 'color': '#ff6b6b'},
                            {'range': [1500, 2000], 'color': '#9e7cc1'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': predicted_price_m3
                        }
                    }
                ))
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)
                
            elif predict_btn and st.session_state.model is None:
                st.warning("⚠️ Vui lòng huấn luyện model trước khi dự đoán (nút 🤖 ở sidebar)")
            else:
                st.info("👈 Nhập thông tin sản phẩm và nhấn 'Dự đoán giá'")
    
    # Tab 2: Data Analysis
    with tab2:
        st.subheader("📊 Phân tích dữ liệu giá")
        
        df = st.session_state.data
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📦 Tổng sản phẩm", f"{len(df):,}")
        with col2:
            st.metric("💰 Giá TB (m³)", f"${df['price_m3'].mean():,.0f}")
        with col3:
            st.metric("📈 Giá cao nhất", f"${df['price_m3'].max():,.0f}")
        with col4:
            st.metric("📉 Giá thấp nhất", f"${df['price_m3'].min():,.0f}")
        
        st.divider()
        
        # Charts
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            # Price distribution by segment
            segment_counts = df['segment'].value_counts()
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
            # Average price by family
            avg_by_family = df.groupby('family')['price_m3'].mean().sort_values(ascending=True)
            fig_family = px.bar(
                x=avg_by_family.values,
                y=avg_by_family.index,
                orientation='h',
                title="Giá trung bình theo loại sản phẩm",
                labels={'x': 'USD/m³', 'y': 'Loại sản phẩm'}
            )
            fig_family.update_traces(marker_color='#667eea')
            st.plotly_chart(fig_family, use_container_width=True)
        
        # Price by stone type
        st.markdown("#### 💎 Giá theo loại đá")
        fig_stone = px.box(
            df,
            x='stone_class',
            y='price_m3',
            color='stone_class',
            title="Phân bố giá theo loại đá"
        )
        st.plotly_chart(fig_stone, use_container_width=True)
        
        # Price vs dimensions
        st.markdown("#### 📐 Giá theo kích thước")
        fig_scatter = px.scatter(
            df,
            x='volume_m3',
            y='price_m3',
            color='segment',
            size='height_cm',
            hover_data=['product_name', 'family'],
            title="Giá m³ vs Thể tích",
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
            search_stone = st.selectbox("Loại đá", [''] + STONE_CLASSES, key='search_stone')
            
            search_col1, search_col2, search_col3 = st.columns(3)
            with search_col1:
                search_length = st.number_input("Dài (cm)", min_value=0, value=30, key='search_l')
            with search_col2:
                search_width = st.number_input("Rộng (cm)", min_value=0, value=30, key='search_w')
            with search_col3:
                search_height = st.number_input("Dày (cm)", min_value=0.0, value=3.0, key='search_h')
            
            top_n = st.slider("Số kết quả", 3, 20, 5)
            
            search_btn = st.button("🔍 Tìm kiếm", type="primary")
        
        with col2:
            if search_btn:
                query = {
                    'family': search_family if search_family else None,
                    'stone_class': search_stone if search_stone else None,
                    'length_cm': search_length,
                    'width_cm': search_width,
                    'height_cm': search_height
                }
                
                similar = find_similar_products(st.session_state.data, query, top_n)
                
                if len(similar) > 0:
                    st.markdown(f"#### Tìm thấy {len(similar)} sản phẩm tương tự")
                    
                    # Display results
                    display_cols = ['product_name', 'family', 'stone_class', 'length_cm', 'width_cm', 
                                    'height_cm', 'charge_unit', 'list_price', 'price_m3', 'segment']
                    st.dataframe(
                        similar[display_cols],
                        use_container_width=True
                    )
                    
                    # Price comparison chart
                    fig = px.bar(
                        similar,
                        x='product_name',
                        y='price_m3',
                        color='segment',
                        title="So sánh giá sản phẩm tương tự",
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
    
    # Tab 4: Model Performance
    with tab4:
        st.subheader("📈 Hiệu suất Model ML")
        
        if st.session_state.model_metrics is not None:
            metrics = st.session_state.model_metrics
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("MAE", f"${metrics['mae']:,.2f}", help="Mean Absolute Error")
            with col2:
                st.metric("R² Score", f"{metrics['r2']:.3f}", help="Coefficient of Determination")
            with col3:
                st.metric("CV MAE Mean", f"${metrics['cv_mae_mean']:,.2f}", help="Cross-Validation MAE")
            with col4:
                st.metric("CV MAE Std", f"${metrics['cv_mae_std']:,.2f}", help="Cross-Validation Std")
            
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
                st.markdown("""
                **Thuật toán:** Gradient Boosting Regressor
                
                **Hyperparameters:**
                - n_estimators: 100
                - learning_rate: 0.1
                - max_depth: 5
                - min_samples_split: 5
                - min_samples_leaf: 2
                
                **Features:**
                - Categorical: family, stone_class, stone_color_type, charge_unit
                - Numerical: length_cm, width_cm, height_cm, volume_m3, area_m2
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
        
        # Display data
        st.dataframe(
            filtered_df,
            use_container_width=True,
            height=500
        )
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            "📥 Tải xuống CSV",
            csv,
            "stone_price_data.csv",
            "text/csv",
            use_container_width=True
        )


if __name__ == "__main__":
    main()
