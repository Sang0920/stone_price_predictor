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

# TLR (Trọng Lượng Riêng - Specific Weight) per TÍNH TOÁN & BÁO GIÁ documentation
TLR_CONSTANTS = {
    # Đá đen khu vực Đak Nông
    'ABSOLUTE BASALT': 2.95,
    'DAK_NONG_BASALT': 2.95,
    # Đá khu vực Phước Hòa và Qui Nhơn
    'BLACK BASALT': 2.65,  # Chẻ tay: 2.65, cắt máy: 2.7
    'BLACK BASALT_SAWN': 2.70,
    'HIVE BASALT': 2.20,  # Đá tổ ong
    # Granite
    'GREY GRANITE': 2.70,
    'DARK GREY GRANITE': 2.90,
    'WHITE GRANITE': 2.70,
    'YELLOW GRANITE': 2.70,
    'RED GRANITE': 2.70,
    'PINK GRANITE': 2.70,
    # Bluestone
    'BLUESTONE': 2.70,
    # Marble
    'WHITE MARBLE': 2.70,
    'YELLOW MARBLE': 2.70,
    # Default
    'DEFAULT': 2.70,
}

# HS (Hệ Số Ốp Đáy - Coating Factor) per TÍNH TOÁN & BÁO GIÁ documentation
HS_FACTORS = {
    # Đá lát 6cm mặt đốt, cạnh sộ (ốp đáy giảm 3%)
    'FLAMED_TILE_6CM': 0.97,
    # Đá cubic chẻ tay
    'CUBE_5X5X5': 1.00,
    'CUBE_8X8X8': 0.95,
    'CUBE_10X10X8': 0.875,
    'CUBE_20X10X8': 0.875,
    'CUBE_15X15X12': 0.85,
    # Đá cubic mặt đốt, cạnh chẻ tay
    'CUBE_FLAMED_10X10X8': 0.95,
    'CUBE_FLAMED_20X10X8': 0.95,
    # Đá cây cưa lột (thêm 5% do dày 10.5cm thực tế)
    'PALISADE_SAWN': 1.05,
    # Default
    'DEFAULT': 1.00,
}

# Customer Pricing Rules (A-F) per NGUYÊN TẮC ÁP DỤNG BẢNG GIÁ documentation
# Segment-aware adjustments
CUSTOMER_PRICING_RULES = {
    'A': {
        'description': 'Khách thân thiết đặc biệt (>10 năm, 50-150 cont)',
        'base_adjustment': {'min': -0.03, 'max': -0.015},  # -1.5% to -3% vs B
        'label': 'Bớt 1.5-3% so với B',
        'years': '>10',
        'volume': '50-150 cont',
        'authority': 'Thảo luận chiến lược'
    },
    'B': {
        'description': 'Khách lớn, chuyên nghiệp (3-10 năm, 20-50 cont)',
        'base_adjustment': {'min': -0.04, 'max': -0.02},  # -2% to -4% vs C
        'usd_adjustment': {'min': -30, 'max': -10},  # -10 to -30 USD/m³ vs C
        'label': 'Thấp hơn C: 2-4% (10-30 USD/m³)',
        'years': '3-10',
        'volume': '20-50 cont',
        'authority': 'Thảo luận chiến lược'
    },
    'C': {
        'description': 'Khách hàng phổ thông (1-5 năm, 5-20 cont)',
        'base_adjustment': {'min': 0, 'max': 0},  # Base price
        'label': 'Giá chuẩn',
        'years': '1-5',
        'volume': '5-20 cont',
        'authority': {
            'Economy': 10,      # ±10 USD/m³
            'Common': 15,       # ±15 USD/m³
            'Premium': 20,      # ±20 USD/m³ or ±0.5 USD/m²
        }
    },
    'D': {
        'description': 'Khách mới, khu vực chi trả cao, size nhỏ (1 năm, 1-10 cont)',
        'base_adjustment': {'min': 0.03, 'max': 0.06},  # +3% to +6%
        'usd_adjustment': {'min': 15, 'max': 45},  # +15 to +45 USD/m³
        'label': 'Cao hơn C: 3-6% (15-45 USD/m³)',
        'years': '1',
        'volume': '1-10 cont',
        'authority': {
            'Premium': 30,       # ±30 USD/m³ or ±1.0 USD/m²
            'Super premium': 40, # ±40 USD/m³ or ±1.5 USD/m²
        }
    },
    'E': {
        'description': 'Sản phẩm mới, sáng tạo, cao cấp (1 năm, 1-10 cont)',
        'base_adjustment': {'min': 0.08, 'max': 0.15},  # ×1.08 to ×1.15
        'label': 'Giá cao cấp: ×1.08-1.15 (+5-10%)',
        'years': '1',
        'volume': '1-10 cont',
        'authority': {
            'Premium': 30,       # ±30 USD/m³ or ±1.0 USD/m²
            'Super premium': 40, # ±40 USD/m³ or ±1.5 USD/m²
        }
    },
    'F': {
        'description': 'Khách hàng dự án, cao cấp (1-5 năm, 1-50 cont)',
        'base_adjustment': {'min': 0.08, 'max': 0.15},  # ×1.08 to ×1.15
        'label': 'Dự án: ×1.08-1.15',
        'years': '1-5',
        'volume': '1-50 cont',
        'authority': {
            'Premium': 30,       # ±30 USD/m³ or ±1.0 USD/m²
            'Super premium': 40, # ±40 USD/m³ or ±1.5 USD/m²
        }
    },
}

PRODUCT_FAMILIES = [
    'Exterior_Tiles', 'Interior_Tiles', 'WALLSTONE', 'PALISADE', 
    'STAIR', 'ART', 'High-Class', 'SKIRTING', 'SLAB'
]

# Application codes (SKU positions 3-4) with English names
# Per "Copy of Code Rule AND Product list" and "Application Mapping" docs
APPLICATION_CODES = [
    ('1.1', 'Cubes / Cobbles'),
    ('1.3', 'Paving stone / Paving slab/ Crazy Paving'),
    ('2.1', 'Wall stone / Wall brick'),
    ('2.2', 'Wall covering / Wall top'),
    ('2.3', 'Rockface Walling'),
    ('3.1', 'Palisades'),
    ('3.2', 'Border / Kerbs'),
    ('3.3', 'Corner'),
    ('4.1', 'Stair / Step (Block)'),
    ('4.2', 'Step (Cladding)'),
    ('5.1', 'Block'),
    ('6.1', 'Pool surrounding'),
    ('6.2', 'Window sill'),
    ('7.2', 'Tile / Paver'),
    ('8.1', 'Skirtings'),
    ('9.1', 'Slab'),
]

# Application codes for search (includes 'All' option)
APPLICATION_CODES_SEARCH = [('', 'All')] + APPLICATION_CODES

# Stone Color Types and their family groupings
STONE_COLOR_TYPES = [
    'BLACK BASALT', 'BLUESTONE', 'GREY GRANITE', 'ABSOLUTE BASALT',
    'WHITE GRANITE', 'YELLOW GRANITE', 'RED GRANITE', 'PINK GRANITE',
    'WHITE MARBLE', 'YELLOW MARBLE', 'HIVE BASALT'
]

# Stone family mapping (for Priority 2 matching - same family)
STONE_FAMILY_MAP = {
    'BLACK BASALT': 'BASALT',
    'ABSOLUTE BASALT': 'BASALT',
    'HIVE BASALT': 'BASALT',
    'GREY GRANITE': 'GRANITE',
    'WHITE GRANITE': 'GRANITE',
    'YELLOW GRANITE': 'GRANITE',
    'RED GRANITE': 'GRANITE',
    'PINK GRANITE': 'GRANITE',
    'BLUESTONE': 'BLUESTONE',
    'WHITE MARBLE': 'MARBLE',
    'YELLOW MARBLE': 'MARBLE',
}

# Dimension tolerance levels per notes.md
DIMENSION_PRIORITY_LEVELS = {
    'Ưu tiên 1 - Đúng kích thước': {'height': 0, 'width': 0, 'length': 0},
    'Ưu tiên 2 - Sai lệch nhỏ': {'height': 1, 'width': 5, 'length': 10},
    'Ưu tiên 3 - Sai lệch lớn': {'height': 5, 'width': 15, 'length': 30},
}

CHARGE_UNITS = ['USD/PC', 'USD/M2', 'USD/TON', 'USD/ML', 'USD/M3']

# Customer Regional Groups (Nhóm Khu vực KH)
CUSTOMER_REGIONAL_GROUPS = [
    ('', 'All'),
    ('Nhóm đầu 0', 'Nhóm đầu 0'),
    ('Nhóm đầu 1', 'Nhóm đầu 1'),
    ('Nhóm đầu 2', 'Nhóm đầu 2'),
    ('Nhóm đầu 3', 'Nhóm đầu 3'),
    ('Nhóm đầu 4', 'Nhóm đầu 4'),
    ('Nhóm đầu 5', 'Nhóm đầu 5'),
    ('Nhóm đầu 6', 'Nhóm đầu 6'),
    ('Nhóm đầu 7', 'Nhóm đầu 7'),
    ('Nhóm đầu 8', 'Nhóm đầu 8'),
    ('Nhóm đầu 9', 'Nhóm đầu 9'),
]

# Processing codes with English and Vietnamese names
# Format: (code, English, Vietnamese)
PROCESSING_CODES = [
    ('CUA', 'Sawn', 'Cưa'),
    ('DOT', 'Flamed', 'Đốt'),
    ('DOC', 'Flamed Brush', 'Đốt Chải'),
    ('DOX', 'Flamed Water', 'Đốt Xịt Nước'),
    ('HON', 'Honed', 'Hon/Mài Mịn'),
    ('CTA', 'Split Handmade', 'Chẻ Tay'),
    ('CLO', 'Sawn then Cleaved', 'Cưa Lột'),
    ('TDE', 'Chiseled', 'Tước Đẽo'),
    ('GCR', 'Vibrated Honed Tumbled', 'Gọt Cạnh Rung'),
    ('GCT', 'Old Imitation', 'Giả Cổ Tay'),
    ('MGI', 'Scraped', 'Mài Giấy'),
    ('PCA', 'Sandblasted', 'Phun Cát'),
    ('QME', 'Tumbled', 'Quay Mẻ'),
    ('TLO', 'Cleaved', 'Tự Nhiên Lồi'),
    ('BON', 'Polished', 'Bóng'),
    ('BAM', 'Bush Hammered', 'Băm'),
    ('CHA', 'Brush', 'Chải'),
]

# Processing codes for search (includes 'All' option)
PROCESSING_CODES_SEARCH = [('', 'All', 'Tất cả')] + PROCESSING_CODES


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
        # customer_regional_group is the customer's regional group (Nhóm đầu 0-9) as per notes.md
        self.categorical_columns = ['family', 'stone_color_type', 'charge_unit', 'processing_code', 'customer_regional_group']
        self.numerical_columns = ['length_cm', 'width_cm', 'height_cm', 'volume_m3', 'area_m2']
        # Recency weight decay factor (prices decay by half every 365 days)
        self.recency_half_life_days = 365
        
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
        
        # Clean customer_regional_group: replace empty/None with 'Unknown'
        if 'customer_regional_group' in df_clean.columns:
            df_clean['customer_regional_group'] = df_clean['customer_regional_group'].fillna('Unknown')
            df_clean['customer_regional_group'] = df_clean['customer_regional_group'].replace('', 'Unknown')
        
        # Remove rows with missing critical features (excluding columns handled above)
        handled_cols = ['processing_code', 'customer_regional_group']
        for col in self.categorical_columns:
            if col in df_clean.columns and col not in handled_cols:
                df_clean = df_clean[df_clean[col].notna()]
        
        for col in self.numerical_columns:
            if col in df_clean.columns:
                df_clean = df_clean[df_clean[col].notna() & (df_clean[col] >= 0)]
        
        # Remove extreme outliers using IQR method for target variable
        Q1 = df_clean[target_col].quantile(0.01)
        Q3 = df_clean[target_col].quantile(0.99)
        df_clean = df_clean[(df_clean[target_col] >= Q1) & (df_clean[target_col] <= Q3)]
        
        return df_clean
    
    def calculate_recency_weights(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calculate sample weights based on recency (more recent prices have higher weight).
        Uses exponential time decay with configurable half-life.
        
        This helps improve accuracy for new products by prioritizing recent price data,
        accounting for annual cost increases in raw materials and labor.
        """
        if 'created_date' not in df.columns:
            return np.ones(len(df))
        
        # Convert created_date to datetime
        dates = pd.to_datetime(df['created_date'], errors='coerce', utc=True)
        
        # Calculate days since each transaction
        reference_date = pd.Timestamp.now(tz='UTC')
        days_ago = (reference_date - dates).dt.total_seconds() / (24 * 3600)
        
        # Handle NaT values - fill with a large number (oldest date equivalent)
        max_days = days_ago.max()
        if pd.isna(max_days):
            max_days = 365 * 5  # Default to 5 years if all dates are NaT
        days_ago = days_ago.fillna(max_days)
        
        # Exponential time decay: weight = 2^(-days_ago / half_life)
        # Recent prices (days_ago=0) get weight=1, prices from 1 year ago get weight=0.5
        weights = np.power(2, -days_ago / self.recency_half_life_days)
        
        # Normalize weights to have mean of 1 (preserves sample count influence)
        weights = weights / weights.mean()
        
        return weights.values
        
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
        """Train the sales price prediction model with proper data cleaning and recency weighting."""
        # Clean data: remove invalid, missing, and outlier data
        df_clean = self.clean_data(df, target_col)
        
        if len(df_clean) < 50:
            raise ValueError(f"Không đủ dữ liệu hợp lệ để huấn luyện model (chỉ có {len(df_clean)} mẫu, cần ít nhất 50)")
        
        # Calculate recency weights (recent prices have higher weight)
        sample_weights = self.calculate_recency_weights(df_clean)
        
        # Prepare features
        X = self.prepare_features(df_clean, fit=True)
        y = df_clean[target_col].values
        
        # Split data with stratification based on charge_unit if possible
        # Also split sample weights to use during training
        X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
            X, y, sample_weights, test_size=0.2, random_state=42
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
        # Use sample weights during training to prioritize recent prices
        self.model.fit(X_train, y_train, sample_weight=weights_train)
        
        # Evaluate on test set (weighted by recency)
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred, sample_weight=weights_test)
        r2 = r2_score(y_test, y_pred, sample_weight=weights_test)
        
        # Cross-validation for more robust metrics (unweighted for comparison)
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
            'n_estimators_used': self.model.n_estimators_,
            'recency_weighted': True
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
def get_tlr(stone_color_type: str, processing_code: str = None) -> float:
    """
    Get TLR (Specific Weight) for stone type.
    Per TÍNH TOÁN & BÁO GIÁ documentation.
    """
    # Check for sawn processing (cắt máy = higher TLR)
    if processing_code in ['CUA', 'HON', 'BON'] and 'BASALT' in stone_color_type.upper():
        return TLR_CONSTANTS.get(stone_color_type + '_SAWN', TLR_CONSTANTS.get(stone_color_type, 2.70))
    return TLR_CONSTANTS.get(stone_color_type, TLR_CONSTANTS['DEFAULT'])


def get_hs_factor(dimensions: tuple = None, processing_code: str = None, family: str = None) -> float:
    """
    Get HS (Coating Factor) for product dimensions/type.
    Per TÍNH TOÁN & BÁO GIÁ documentation.
    """
    if dimensions:
        l, w, h = dimensions
        # Check for cube dimensions
        if l == 5 and w == 5 and h == 5:
            return HS_FACTORS['CUBE_5X5X5']
        if l == 8 and w == 8 and h == 8:
            return HS_FACTORS['CUBE_8X8X8']
        if (l == 10 and w == 10 and h == 8) or (l == 20 and w == 10 and h == 8):
            if processing_code in ['DOT', 'DOC', 'DOX']:  # Flamed
                return HS_FACTORS['CUBE_FLAMED_10X10X8']
            return HS_FACTORS['CUBE_10X10X8']
        if l == 15 and w == 15 and h == 12:
            return HS_FACTORS['CUBE_15X15X12']
        # Flamed tile 6cm
        if h == 6 and processing_code in ['DOT', 'DOC', 'DOX']:
            return HS_FACTORS['FLAMED_TILE_6CM']
    
    # Check for palisade
    if family == 'PALISADE' and processing_code in ['CLO', 'CUA']:
        return HS_FACTORS['PALISADE_SAWN']
    
    return HS_FACTORS['DEFAULT']


def calculate_volume_m3(length_cm: float, width_cm: float, height_cm: float, quantity: int = 1) -> float:
    """Calculate volume in m³. Formula: (L×W×H)/1,000,000 × qty"""
    return (length_cm * width_cm * height_cm) / 1_000_000 * quantity


def calculate_area_m2(length_cm: float, width_cm: float, quantity: int = 1) -> float:
    """Calculate area in m². Formula: (L×W)/10,000 × qty"""
    return (length_cm * width_cm) / 10_000 * quantity


def calculate_weight_tons(volume_m3: float, stone_color_type: str, processing_code: str = None,
                          dimensions: tuple = None, family: str = None) -> float:
    """
    Calculate weight in tons.
    Formula: m³ × TLR × HS
    """
    tlr = get_tlr(stone_color_type, processing_code)
    hs = get_hs_factor(dimensions, processing_code, family)
    return volume_m3 * tlr * hs


def convert_price(price: float, from_unit: str, to_unit: str, 
                  height_cm: float = None, tlr: float = 2.70, hs: float = 1.0,
                  length_cm: float = None, width_cm: float = None) -> float:
    """
    Convert price between units (USD/PC, USD/M2, USD/M3, USD/TON).
    Per TÍNH TOÁN & BÁO GIÁ documentation.
    """
    height_m = (height_cm / 100) if height_cm else 0.03
    
    # First convert to price per m³
    if from_unit == 'USD/M3':
        price_m3 = price
    elif from_unit == 'USD/M2':
        price_m3 = price / height_m
    elif from_unit == 'USD/TON':
        price_m3 = price * tlr * hs
    elif from_unit == 'USD/PC':
        if length_cm and width_cm and height_cm:
            vol = (length_cm * width_cm * height_cm) / 1_000_000
            price_m3 = price / vol if vol > 0 else price
        else:
            price_m3 = price * 100  # Rough estimate
    else:
        price_m3 = price
    
    # Then convert from m³ to target unit
    if to_unit == 'USD/M3':
        return price_m3
    elif to_unit == 'USD/M2':
        return price_m3 * height_m
    elif to_unit == 'USD/TON':
        return price_m3 / tlr / hs if tlr > 0 else price_m3
    elif to_unit == 'USD/PC':
        if length_cm and width_cm and height_cm:
            vol = (length_cm * width_cm * height_cm) / 1_000_000
            return price_m3 * vol
        else:
            return price_m3 / 100  # Rough estimate
    else:
        return price_m3


def classify_segment(price_m3: float, height_cm: float = None, family: str = None, 
                     processing_code: str = None) -> str:
    """
    Classify price into segment.
    Per PHÂN KHÚC DỰA TRÊN GIÁ VÀ SẢN PHẨM documentation.
    
    Considers both price AND product characteristics:
    - Super premium: ≥$1500/m³ OR thin paving (1-1.5cm), wall/pool covering, decorative
    - Premium: ≥$800/m³ OR tiles (2-5cm), slabs, steps
    - Common: ≥$400/m³ OR palisades, cubes, tumbled
    - Economy: <$400/m³ OR natural split, thick pavers
    """
    # Check product-based rules first
    if height_cm is not None:
        # Thin paving (1.0-1.5cm) = Super premium
        if height_cm <= 1.5 and family in ['Exterior_Tiles', 'Interior_Tiles']:
            return 'Super premium'
        # Tiles 2-5cm with quality processing = Premium
        if 2.0 <= height_cm <= 5.0 and processing_code in ['DOT', 'DOC', 'DOX', 'HON', 'BON']:
            if price_m3 >= 600:  # Slightly lower threshold for processed tiles
                return 'Premium'
        # Thick natural split (≥6cm) = Economy
        if height_cm >= 6 and processing_code in ['CTA', 'TLO']:
            return 'Economy'
    
    # Check family-based rules
    if family:
        if family in ['ART', 'High-Class']:
            return 'Super premium'
        if family in ['SLAB', 'STAIR']:
            return 'Premium' if price_m3 >= 600 else 'Common'
        if family == 'PALISADE' and processing_code in ['CLO', 'CUA']:
            return 'Common'
    
    # Fall back to price-based classification
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

def calculate_customer_price(base_price: float, customer_type: str, 
                             segment: str = None, charge_unit: str = 'USD/M3') -> Dict[str, Any]:
    """
    Calculate price adjustments for different customer types.
    Per NGUYÊN TẮC ÁP DỤNG BẢNG GIÁ ABCDEF documentation.
    
    Args:
        base_price: The reference price
        customer_type: Customer classification (A-F)
        segment: Product segment for authority range
        charge_unit: Price unit for displaying USD adjustments
    """
    rules = CUSTOMER_PRICING_RULES.get(customer_type, CUSTOMER_PRICING_RULES['C'])
    adj = rules.get('base_adjustment', {'min': 0, 'max': 0})
    
    min_price = round(base_price * (1 + adj['min']), 2)
    max_price = round(base_price * (1 + adj['max']), 2)
    
    # Get authority range based on segment
    authority_range = None
    authority = rules.get('authority')
    if isinstance(authority, dict) and segment:
        authority_range = authority.get(segment)
    
    # Format authority display
    if authority_range:
        if charge_unit == 'USD/M2':
            auth_display = f"±{authority_range * 0.05:.1f} USD/m²"  # Approximate m² conversion
        else:
            auth_display = f"±{authority_range} USD/m³"
    elif isinstance(authority, str):
        auth_display = authority
    else:
        auth_display = rules.get('label', 'N/A')
    
    return {
        'base_price': base_price,
        'min_price': min_price,
        'max_price': max_price,
        'adjustment_label': rules.get('label', 'N/A'),
        'customer_description': rules.get('description', ''),
        'authority_range': auth_display,
        'volume': rules.get('volume', ''),
        'years': rules.get('years', ''),
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


# ============ Similarity-Based Price Predictor ============
class SimilarityPricePredictor:
    """
    Price estimation based on similarity search with priority levels.
    Matches products using criteria from notes.md.
    """
    
    def __init__(self):
        self.data = None
        self.recency_half_life_days = 365
        
    def load_data(self, df: pd.DataFrame):
        """Load and prepare data for similarity search."""
        self.data = df[df['sales_price'].notna() & (df['sales_price'] > 0)].copy()
        # Add stone family for priority 2 matching
        if 'stone_color_type' in self.data.columns:
            self.data['stone_family'] = self.data['stone_color_type'].map(STONE_FAMILY_MAP).fillna('OTHER')
        return len(self.data)
    
    def find_matching_products(
        self, 
        stone_color_type: str,
        processing_code: str,
        length_cm: float,
        width_cm: float,
        height_cm: float,
        application_codes: list,  # List of application codes (empty = all)
        customer_regional_group: str,
        charge_unit: str,
        stone_priority: str = 'Ưu tiên 1',  # Exact, Same Family, All
        processing_priority: str = 'Ưu tiên 1',  # Exact, All
        dimension_priority: str = 'Ưu tiên 1 - Đúng kích thước',
        region_priority: str = 'Ưu tiên 1',  # Exact, All
    ) -> pd.DataFrame:
        """
        Find matching products based on priority criteria from notes.md.
        
        Priority Levels:
        - Ưu tiên 1: Exact match
        - Ưu tiên 2: Same family / small tolerance
        - Ưu tiên 3: All / large tolerance
        """
        if self.data is None or len(self.data) == 0:
            return pd.DataFrame()
        
        df = self.data.copy()
        mask = pd.Series([True] * len(df), index=df.index)
        
        # 1. Stone Type Filter
        query_family = STONE_FAMILY_MAP.get(stone_color_type, 'OTHER')
        if stone_priority == 'Ưu tiên 1':
            mask &= df['stone_color_type'] == stone_color_type
        elif stone_priority == 'Ưu tiên 2':
            mask &= df['stone_family'] == query_family
        # Ưu tiên 3: No filter (All stones)
        
        # 2. Processing Filter
        if processing_priority == 'Ưu tiên 1' and processing_code:
            mask &= df['processing_code'] == processing_code
        # Ưu tiên 2+: No filter (All processing types)
        
        # 3. Application Filter (extracted from SKU positions 3-4)
        # If application_codes is not empty, filter by those codes
        if application_codes and len(application_codes) > 0 and 'application_code' in df.columns:
            mask &= df['application_code'].isin(application_codes)
        
        # 4. Charge Unit Filter
        if charge_unit:
            mask &= df['charge_unit'] == charge_unit
        
        # 5. Regional Group Filter
        if 'customer_regional_group' in df.columns:
            if region_priority == 'Ưu tiên 1' and customer_regional_group:
                mask &= df['customer_regional_group'] == customer_regional_group
            # Ưu tiên 2+: No filter (All regions)
        
        # Apply initial filters
        df_filtered = df[mask].copy()
        
        if len(df_filtered) == 0:
            return pd.DataFrame()
        
        # 6. Dimension Filter with tolerances
        tolerances = DIMENSION_PRIORITY_LEVELS.get(dimension_priority, {'height': 0, 'width': 0, 'length': 0})
        
        dim_mask = (
            (abs(df_filtered['height_cm'] - height_cm) <= tolerances['height']) &
            (abs(df_filtered['width_cm'] - width_cm) <= tolerances['width']) &
            (abs(df_filtered['length_cm'] - length_cm) <= tolerances['length'])
        )
        
        df_matches = df_filtered[dim_mask].copy()
        
        return df_matches
    
    def get_match_diagnostics(
        self,
        stone_color_type: str,
        processing_code: str,
        length_cm: float,
        width_cm: float,
        height_cm: float,
        application_codes: list,
        customer_regional_group: str,
        charge_unit: str,
        stone_priority: str = 'Ưu tiên 1',
        processing_priority: str = 'Ưu tiên 1',
        dimension_priority: str = 'Ưu tiên 1 - Đúng kích thước',
        region_priority: str = 'Ưu tiên 1',
    ) -> Dict[str, Any]:
        """
        Analyze why no matches were found and return diagnostic information.
        Returns closest available dimensions and filter breakdown.
        """
        if self.data is None or len(self.data) == 0:
            return {'reason': 'Không có dữ liệu', 'suggestions': []}
        
        df = self.data.copy()
        diagnostics = {
            'reason': '',
            'suggestions': [],
            'closest_height': None,
            'closest_width': None,
            'closest_length': None,
            'filter_counts': {}
        }
        
        # Track filter stages
        mask = pd.Series([True] * len(df), index=df.index)
        diagnostics['filter_counts']['total'] = len(df)
        
        # 1. Stone type
        query_family = STONE_FAMILY_MAP.get(stone_color_type, 'OTHER')
        if stone_priority == 'Ưu tiên 1':
            stone_mask = df['stone_color_type'] == stone_color_type
        elif stone_priority == 'Ưu tiên 2':
            stone_mask = df['stone_family'] == query_family
        else:
            stone_mask = pd.Series([True] * len(df), index=df.index)
        mask &= stone_mask
        diagnostics['filter_counts']['after_stone'] = mask.sum()
        
        # 2. Processing
        if processing_priority == 'Ưu tiên 1' and processing_code:
            proc_mask = df['processing_code'] == processing_code
            mask &= proc_mask
        diagnostics['filter_counts']['after_processing'] = mask.sum()
        
        # 3. Application
        if application_codes and len(application_codes) > 0 and 'application_code' in df.columns:
            mask &= df['application_code'].isin(application_codes)
        diagnostics['filter_counts']['after_application'] = mask.sum()
        
        # 4. Charge unit
        if charge_unit:
            mask &= df['charge_unit'] == charge_unit
        diagnostics['filter_counts']['after_charge_unit'] = mask.sum()
        
        # 5. Region
        if 'customer_regional_group' in df.columns and region_priority == 'Ưu tiên 1' and customer_regional_group:
            mask &= df['customer_regional_group'] == customer_regional_group
        diagnostics['filter_counts']['after_region'] = mask.sum()
        
        df_filtered = df[mask].copy()
        
        if len(df_filtered) == 0:
            # Find which filter caused the problem
            if diagnostics['filter_counts']['after_stone'] == 0:
                diagnostics['reason'] = f"Không tìm thấy sản phẩm loại đá '{stone_color_type}'"
                diagnostics['suggestions'].append("Thử chọn Ưu tiên 2 hoặc 3 cho Loại đá")
            elif diagnostics['filter_counts']['after_processing'] == 0:
                diagnostics['reason'] = f"Không tìm thấy gia công '{processing_code}' cho loại đá này"
                diagnostics['suggestions'].append("Thử chọn Ưu tiên 2 cho Gia công")
            elif diagnostics['filter_counts']['after_application'] == 0:
                app_names = ', '.join(application_codes) if application_codes else ''
                diagnostics['reason'] = f"Không tìm thấy ứng dụng '{app_names}' cho các tiêu chí đã chọn"
                diagnostics['suggestions'].append("Thử bỏ chọn ứng dụng cụ thể")
            elif diagnostics['filter_counts']['after_charge_unit'] == 0:
                diagnostics['reason'] = f"Không tìm thấy đơn vị tính '{charge_unit}'"
                diagnostics['suggestions'].append("Thử đổi đơn vị tính giá")
            else:
                diagnostics['reason'] = "Không tìm thấy sản phẩm với các tiêu chí đã chọn"
            return diagnostics
        
        # 6. Check dimensions
        tolerances = DIMENSION_PRIORITY_LEVELS.get(dimension_priority, {'height': 0, 'width': 0, 'length': 0})
        
        # Find closest dimensions in filtered data
        closest_height = df_filtered.loc[(df_filtered['height_cm'] - height_cm).abs().idxmin(), 'height_cm']
        closest_width = df_filtered.loc[(df_filtered['width_cm'] - width_cm).abs().idxmin(), 'width_cm']
        closest_length = df_filtered.loc[(df_filtered['length_cm'] - length_cm).abs().idxmin(), 'length_cm']
        
        diagnostics['closest_height'] = closest_height
        diagnostics['closest_width'] = closest_width
        diagnostics['closest_length'] = closest_length
        
        height_diff = abs(closest_height - height_cm)
        width_diff = abs(closest_width - width_cm)
        length_diff = abs(closest_length - length_cm)
        
        # Check which dimension is blocking
        dim_issues = []
        if height_diff > tolerances['height']:
            dim_issues.append(f"Cao {height_cm}cm (gần nhất: {closest_height}cm, sai lệch: {height_diff:.0f}cm > ±{tolerances['height']}cm)")
        if width_diff > tolerances['width']:
            dim_issues.append(f"Rộng {width_cm}cm (gần nhất: {closest_width}cm, sai lệch: {width_diff:.0f}cm > ±{tolerances['width']}cm)")
        if length_diff > tolerances['length']:
            dim_issues.append(f"Dài {length_cm}cm (gần nhất: {closest_length}cm, sai lệch: {length_diff:.0f}cm > ±{tolerances['length']}cm)")
        
        if dim_issues:
            diagnostics['reason'] = "Không tìm thấy kích thước phù hợp:\n• " + "\n• ".join(dim_issues)
            diagnostics['suggestions'].append("Thử chọn Ưu tiên 3 cho Kích thước (sai lệch lớn)")
        
        diagnostics['filter_counts']['after_dimensions'] = len(self.find_matching_products(
            stone_color_type, processing_code, length_cm, width_cm, height_cm,
            application_codes, customer_regional_group, charge_unit,
            stone_priority, processing_priority, dimension_priority, region_priority
        ))
        
        return diagnostics
    
    def calculate_recency_weights(self, df: pd.DataFrame) -> pd.Series:
        """Calculate recency weights for price averaging."""
        if 'created_date' not in df.columns or len(df) == 0:
            return pd.Series([1.0] * len(df), index=df.index)
        
        dates = pd.to_datetime(df['created_date'], errors='coerce', utc=True)
        reference_date = pd.Timestamp.now(tz='UTC')
        days_ago = (reference_date - dates).dt.total_seconds() / (24 * 3600)
        
        max_days = days_ago.max()
        if pd.isna(max_days):
            max_days = 365 * 5
        days_ago = days_ago.fillna(max_days)
        
        weights = np.power(2, -days_ago / self.recency_half_life_days)
        return weights
    
    def estimate_price(self, matches: pd.DataFrame, use_recent_only: bool = True, recent_count: int = 10,
                        query_length_cm: float = None, query_width_cm: float = None, query_height_cm: float = None,
                        target_charge_unit: str = 'USD/M3', stone_color_type: str = None, processing_code: str = None) -> Dict[str, Any]:
        """
        Estimate price from matching products.
        Uses recency-weighted average, optionally filtering to most recent products.
        
        IMPORTANT: Normalizes all prices to USD/M3 before averaging to account for 
        different product sizes. Then converts back to target_charge_unit using 
        query dimensions. This ensures that larger products are priced proportionally 
        higher than smaller similar products.
        
        Args:
            matches: DataFrame of matching products
            use_recent_only: If True, filter to only the most recent products
            recent_count: Number of most recent products to use (if use_recent_only=True)
            query_length_cm: Length of the product being quoted (for unit conversion)
            query_width_cm: Width of the product being quoted (for unit conversion)
            query_height_cm: Height of the product being quoted (for unit conversion)
            target_charge_unit: The unit to return the price in (USD/PC, USD/M2, USD/M3, USD/TON)
            stone_color_type: Stone type for TLR calculation
            processing_code: Processing code for TLR/HS calculation
        """
        if len(matches) == 0:
            return {
                'estimated_price': None,
                'min_price': None,
                'max_price': None,
                'median_price': None,
                'match_count': 0,
                'total_matches': 0,
                'confidence': 'none',
                'years_used': '',
                'price_m3': None
            }
        
        total_matches = len(matches)
        
        # Filter to most recent products based on fy_year and created_date
        if use_recent_only and len(matches) > recent_count:
            # Sort by fy_year (desc) then created_date (desc)
            sorted_matches = matches.copy()
            if 'fy_year' in sorted_matches.columns:
                # Convert fy_year to numeric for proper sorting
                sorted_matches['_fy_year_numeric'] = pd.to_numeric(sorted_matches['fy_year'], errors='coerce')
                sorted_matches = sorted_matches.sort_values(
                    by=['_fy_year_numeric', 'created_date'], 
                    ascending=[False, False],
                    na_position='last'
                )
                sorted_matches = sorted_matches.drop(columns=['_fy_year_numeric'])
            elif 'created_date' in sorted_matches.columns:
                sorted_matches = sorted_matches.sort_values(
                    by=['created_date'], 
                    ascending=[False],
                    na_position='last'
                )
            # Take only the top N most recent
            matches = sorted_matches.head(recent_count)
        
        # Get years used for display
        years_used = ''
        if 'fy_year' in matches.columns:
            unique_years = matches['fy_year'].dropna().unique()
            unique_years = sorted([int(y) for y in unique_years if pd.notna(y)], reverse=True)
            if len(unique_years) > 0:
                years_used = ', '.join(str(y) for y in unique_years[:3])
        
        # Calculate weights
        weights = self.calculate_recency_weights(matches)
        
        # Normalize all prices to USD/M3 before averaging
        # This ensures fair comparison across different product sizes
        prices_m3 = []
        for idx, row in matches.iterrows():
            price = row['sales_price']
            unit = row.get('charge_unit', 'USD/M3')
            match_length = row.get('length_cm', 10)
            match_width = row.get('width_cm', 10)
            match_height = row.get('height_cm', 3)
            match_stone = row.get('stone_color_type', stone_color_type or 'ABSOLUTE BASALT')
            match_proc = row.get('processing_code', processing_code)
            
            # Get TLR and HS for this product
            tlr = get_tlr(match_stone, match_proc)
            hs = get_hs_factor((match_length, match_width, match_height), match_proc)
            
            # Convert to USD/M3
            price_m3 = convert_price(
                price, unit, 'USD/M3',
                height_cm=match_height,
                length_cm=match_length,
                width_cm=match_width,
                tlr=tlr,
                hs=hs
            )
            prices_m3.append(price_m3)
        
        prices_m3 = pd.Series(prices_m3, index=matches.index)
        
        # Weighted average in USD/M3 (the normalized unit)
        weighted_price_m3 = np.average(prices_m3, weights=weights)
        
        # Convert from USD/M3 to target unit using QUERY dimensions
        # This is the key: we use the NEW product's dimensions, not the matched products'
        if query_length_cm is not None and query_width_cm is not None and query_height_cm is not None:
            query_tlr = get_tlr(stone_color_type or 'ABSOLUTE BASALT', processing_code)
            query_hs = get_hs_factor((query_length_cm, query_width_cm, query_height_cm), processing_code)
            
            estimated_price = convert_price(
                weighted_price_m3, 'USD/M3', target_charge_unit,
                height_cm=query_height_cm,
                length_cm=query_length_cm,
                width_cm=query_width_cm,
                tlr=query_tlr,
                hs=query_hs
            )
            
            # Also convert min/max/median to target unit
            min_price = convert_price(
                prices_m3.min(), 'USD/M3', target_charge_unit,
                height_cm=query_height_cm, length_cm=query_length_cm, width_cm=query_width_cm,
                tlr=query_tlr, hs=query_hs
            )
            max_price = convert_price(
                prices_m3.max(), 'USD/M3', target_charge_unit,
                height_cm=query_height_cm, length_cm=query_length_cm, width_cm=query_width_cm,
                tlr=query_tlr, hs=query_hs
            )
            median_price = convert_price(
                prices_m3.median(), 'USD/M3', target_charge_unit,
                height_cm=query_height_cm, length_cm=query_length_cm, width_cm=query_width_cm,
                tlr=query_tlr, hs=query_hs
            )
        else:
            # Fallback: use original method (direct averaging) if no query dimensions
            prices = matches['sales_price']
            estimated_price = np.average(prices, weights=weights)
            min_price = prices.min()
            max_price = prices.max()
            median_price = prices.median()
        
        # Confidence based on match count
        if len(matches) >= 10:
            confidence = 'high'
        elif len(matches) >= 5:
            confidence = 'medium'
        elif len(matches) >= 2:
            confidence = 'low'
        else:
            confidence = 'very_low'
        
        # Calculate price trend based on fy_year
        price_trend = None
        trend_pct = None
        if 'fy_year' in matches.columns and len(matches) >= 3:
            # Group by year and calculate average price_m3
            yearly_data = pd.DataFrame({
                'fy_year': matches['fy_year'],
                'price_m3': prices_m3
            })
            yearly_avg = yearly_data.groupby('fy_year')['price_m3'].mean()
            if len(yearly_avg) >= 2:
                sorted_years = sorted(yearly_avg.index, reverse=True)
                if len(sorted_years) >= 2:
                    this_year_price = yearly_avg[sorted_years[0]]
                    last_year_price = yearly_avg[sorted_years[1]]
                    if last_year_price > 0:
                        trend_pct = ((this_year_price - last_year_price) / last_year_price) * 100
                        if trend_pct > 0:
                            price_trend = 'up'
                        elif trend_pct < 0:
                            price_trend = 'down'
                        else:
                            price_trend = 'stable'
        
        return {
            'estimated_price': round(estimated_price, 2),
            'min_price': round(min_price, 2),
            'max_price': round(max_price, 2),
            'median_price': round(median_price, 2),
            'match_count': len(matches),
            'total_matches': total_matches,
            'confidence': confidence,
            'years_used': years_used,
            'price_m3': round(weighted_price_m3, 2),
            'price_trend': price_trend,
            'trend_pct': round(trend_pct, 1) if trend_pct is not None else None
        }
    
    def predict_with_escalation(
        self,
        stone_color_type: str,
        processing_code: str,
        length_cm: float,
        width_cm: float,
        height_cm: float,
        application_codes: list,  # List of application codes (empty = all)
        customer_regional_group: str,
        charge_unit: str,
    ) -> Tuple[Dict[str, Any], pd.DataFrame, str]:
        """
        Try to find matches with automatic priority escalation.
        Starts with Ưu tiên 1 and escalates if no matches found.
        
        Returns:
            - Price estimation dict
            - Matching products DataFrame
            - Priority level used
        """
        priority_levels = [
            ('Ưu tiên 1', 'Ưu tiên 1', 'Ưu tiên 1 - Đúng kích thước', 'Ưu tiên 1'),
            ('Ưu tiên 1', 'Ưu tiên 1', 'Ưu tiên 2 - Sai lệch nhỏ', 'Ưu tiên 1'),
            ('Ưu tiên 1', 'Ưu tiên 2', 'Ưu tiên 2 - Sai lệch nhỏ', 'Ưu tiên 2'),
            ('Ưu tiên 2', 'Ưu tiên 2', 'Ưu tiên 2 - Sai lệch nhỏ', 'Ưu tiên 2'),
            ('Ưu tiên 2', 'Ưu tiên 2', 'Ưu tiên 3 - Sai lệch lớn', 'Ưu tiên 2'),
            ('Ưu tiên 3', 'Ưu tiên 2', 'Ưu tiên 3 - Sai lệch lớn', 'Ưu tiên 2'),
        ]
        
        for stone_p, proc_p, dim_p, region_p in priority_levels:
            matches = self.find_matching_products(
                stone_color_type=stone_color_type,
                processing_code=processing_code,
                length_cm=length_cm,
                width_cm=width_cm,
                height_cm=height_cm,
                application_codes=application_codes,
                customer_regional_group=customer_regional_group,
                charge_unit=charge_unit,
                stone_priority=stone_p,
                processing_priority=proc_p,
                dimension_priority=dim_p,
                region_priority=region_p,
            )
            
            if len(matches) > 0:
                estimation = self.estimate_price(matches)
                priority_used = f"Đá: {stone_p}, Gia công: {proc_p}, Kích thước: {dim_p}, Khu vực: {region_p}"
                return estimation, matches, priority_used
        
        return self.estimate_price(pd.DataFrame()), pd.DataFrame(), "Không tìm thấy"


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
            placeholder="e.g., X09",
            help="Lọc theo Account_Code_C__c"
        )
        
        if st.button("🔄 Tải / Làm mới dữ liệu từ Salesforce", use_container_width=True):
            with st.spinner("Đang tải và xử lý dữ liệu..."):
                if SALESFORCE_AVAILABLE:
                    try:
                        # Step 1: Load data from Salesforce
                        loader = SalesforceDataLoader()
                        df = loader.get_contract_products(account_code=account_filter if account_filter else None)
                        if len(df) > 0:
                            st.session_state.data = df
                            
                            # Step 2: Auto-preprocess data
                            predictor = SimilarityPricePredictor()
                            count = predictor.load_data(df)
                            st.session_state.model = predictor
                            st.session_state.model_metrics = {'loaded_samples': count}
                            
                            st.success(f"✅ Đã tải {len(df):,} sản phẩm, sẵn sàng với {count:,} sản phẩm có giá!")
                        else:
                            st.warning("⚠️ Không tìm thấy dữ liệu từ Salesforce.")
                    except Exception as e:
                        st.error(f"❌ Lỗi kết nối Salesforce: {str(e)}")
                else:
                    st.error("❌ Salesforce chưa được cấu hình. Vui lòng kiểm tra file .env")
        
        # Show status
        if st.session_state.data is not None:
            count = len(st.session_state.data)
            ready_count = st.session_state.model_metrics.get('loaded_samples', 0) if st.session_state.model_metrics else 0
            st.success(f"✅ Đã sẵn sàng với {count:,} sản phẩm ({ready_count:,} sản phẩm có giá)")
        
        st.divider()
    
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
        "📐 Bảng tra cứu",
        "📋 Dữ liệu chi tiết"
    ])
    
    # Tab 1: Price Prediction
    with tab1:
        st.subheader("🔮 Ước tính giá sản phẩm (Similarity-Based)")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Thông tin sản phẩm")
            
            # Application multiselect - allows selecting multiple applications (e.g., 4.1 + 4.2 for "Step")
            # Empty selection = All (no filter)
            application_lookup = {code: name for code, name in APPLICATION_CODES}
            selected_applications = st.multiselect(
                "Ứng dụng sản phẩm (Application)",
                options=[code for code, name in APPLICATION_CODES],
                format_func=lambda x: f"{x} - {application_lookup.get(x, 'Unknown')}",
                default=[],
                help="Chọn một hoặc nhiều ứng dụng. Để trống = Tất cả"
            )
            stone_color = st.selectbox("Màu đá (Stone Color)", STONE_COLOR_TYPES)
            
            # Main Processing dropdown with English and Vietnamese names
            # Create a dict for lookup: code -> (english, vietnamese)
            processing_lookup = {code: (eng, vn) for code, eng, vn in PROCESSING_CODES}
            processing_code = st.selectbox(
                "Gia công chính (Main Processing)",
                options=[code for code, eng, vn in PROCESSING_CODES],
                format_func=lambda x: f"{x} - {processing_lookup.get(x, ('Other', 'Khác'))[0]} ({processing_lookup.get(x, ('Other', 'Khác'))[1]})",
                index=0
            )
            
            col_dim1, col_dim2, col_dim3 = st.columns(3)
            with col_dim1:
                length = st.number_input("Dài (cm)", min_value=1, max_value=300, value=30)
            with col_dim2:
                width = st.number_input("Rộng (cm)", min_value=1, max_value=300, value=30)
            with col_dim3:
                height = st.number_input("Dày (cm)", min_value=0.5, max_value=50.0, value=3.0, step=0.5)
            
            charge_unit = st.selectbox("Đơn vị tính giá", CHARGE_UNITS)
            
            # Customer Regional Group (Nhóm Khu vực KH)
            customer_regional_group = st.selectbox(
                "Nhóm Khu vực KH (Regional Group)",
                options=[code for code, name in CUSTOMER_REGIONAL_GROUPS if code],
                format_func=lambda x: x,
                index=0,
                help="Nhóm đầu 0-9 theo khu vực khách hàng"
            )
            
            st.divider()
            
            # Rules and Formulas expanders
            with st.expander("📋 Quy tắc định giá"):
                st.markdown("""
**Phân khúc giá (USD/m³):**
| Phân khúc | Giá | Sản phẩm |
|-----------|-----|----------|
| 🟣 Super Premium | ≥ $1,500 | Đá mỏng 1-1.5cm, nắp tường, mỹ nghệ |
| 🔴 Premium | ≥ $800 | Đá lát 2-5cm, slab, bậc thang |
| 🟡 Common | ≥ $400 | Đá cây, cubic đốt, quay mẻ |
| 🟢 Economy | < $400 | Đá gõ tay, cubic chẻ tay |
                """)
            
            with st.expander("👥 Phân loại khách hàng"):
                st.markdown("""
| Loại | Mô tả | Điều chỉnh |
|------|-------|------------|
| **A** | Thân thiết >10 năm | -1.5% đến -3% |
| **B** | Lớn 3-10 năm | -2% đến -4% |
| **C** | Phổ thông | Giá chuẩn |
| **D** | Mới, nhỏ | +3% đến +6% |
| **E** | Sản phẩm mới | ×1.08-1.15 |
| **F** | Dự án | ×1.08-1.15 |
                """)
            
            with st.expander("📐 Công thức tính toán"):
                st.markdown("""
**Thể tích:** `m³ = (D×R×C) / 1.000.000 × SL`

**Diện tích:** `m² = (D×R) / 10.000 × SL`

**Trọng lượng:** `Tấn = m³ × TLR × HS`

**Quy đổi giá:**
- `Giá/m² = Giá/m³ × Cao(m)`
- `Giá/Tấn = Giá/m³ ÷ TLR ÷ HS`

**TLR tham khảo:**
- Absolute Basalt: 2.95
- Black Basalt: 2.65-2.70
- Granite thường: 2.70
- Dark Grey Granite: 2.90
                """)
            
            with st.expander("🎯 Tiêu chí tìm kiếm", expanded=True):
                st.markdown("""
| Tiêu chí | Ưu tiên 1 | Ưu tiên 2 | Ưu tiên 3 |
|----------|-----------|-----------|-----------|
| **Loại đá** | Đúng màu đá | Cùng chủng loại | Tất cả loại đá |
| **Gia công** | Đúng loại gia công | Tất cả gia công | - |
| **Cao (cm)** | ±0 | ±1 | ±5 |
| **Rộng (cm)** | ±0 | ±5 | ±15 |
| **Dài (cm)** | ±0 | ±10 | ±30 |
| **Khu vực** | Đúng khu vực | Tất cả khu vực | - |
                """)
            
            with st.expander("📦 Quy tắc ứng dụng sản phẩm (Application Mapping)"):
                st.markdown("""
**Mã ứng dụng** trích xuất từ vị trí 3-4 trong SKU (chuyển đổi sang định dạng X.Y).

| Mã | English | Tiếng Việt |
|----|---------|------------|
| 1.1 | Cubes / Cobbles | Cubic (Đá vuông) |
| 1.3 | Paving stone / Paving slab / Crazy paving | Đá lát ngoài trời |
| 2.1 | Wall stone / Wall brick | Đá xây tường rào |
| 2.2 | Wall covering / Wall top | Đá ốp tường rào |
| 2.3 | Rockface Walling | Đá mặt lỗi ốp tường |
| 3.1 | Palisades | Đá cây |
| 3.2 | Border / Kerbs | Đá bó vỉa hè loại thẳng |
| 3.3 | Corner | Đá bó vỉa hè, loại góc hoặc cong |
| 4.1 | Stair / Step (Block) | Đá bậc thang nguyên khối |
| 4.2 | Step (Cladding) | Đá ốp bậc thang |
| 5.1 | Block | Đá khối |
| 6.1 | Pool surrounding | Đá ghép hồ bơi |
| 6.2 | Window sill | Đá bệ cửa sổ, gờ tường |
| 7.2 | Tile / Paver | Đá lát, cắt quy cách |
| 8.1 | Skirtings | Đá len chân tường |
| 9.1 | Slab | Đá slab kích thước khổ lớn |

**Ví dụ SKU:** `BD01DOT2-06004060`
- `BD` = Vật liệu (Basalt Đen)
- `01` → `0.1` = **Ứng dụng** (Cubes)
- `DOT` = Gia công (Đốt/Flamed)
                """)
            
            customer_type = st.selectbox(
                "Phân loại khách hàng",
                ['C', 'A', 'B', 'D', 'E', 'F'],
                format_func=lambda x: f"{x} - {CUSTOMER_PRICING_RULES[x]['description']}"
            )
            
            st.divider()
            st.markdown("#### 🎚️ Mức độ ưu tiên tìm kiếm")
            
            # Priority level selectors per notes.md
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                stone_priority = st.selectbox(
                    "Loại đá",
                    options=['Ưu tiên 1', 'Ưu tiên 2', 'Ưu tiên 3'],
                    format_func=lambda x: {
                        'Ưu tiên 1': '1 - Đúng màu đá',
                        'Ưu tiên 2': '2 - Cùng chủng loại',
                        'Ưu tiên 3': '3 - Tất cả loại đá',
                    }[x],
                    index=0  # Default: Ưu tiên 1 (Đúng màu đá)
                )
                processing_priority = st.selectbox(
                    "Gia công",
                    options=['Ưu tiên 1', 'Ưu tiên 2'],
                    format_func=lambda x: {
                        'Ưu tiên 1': '1 - Đúng loại gia công',
                        'Ưu tiên 2': '2 - Tất cả gia công',
                    }[x],
                    index=1  # Default: Ưu tiên 2 (Tất cả gia công)
                )
            with col_p2:
                dimension_priority = st.selectbox(
                    "Kích thước",
                    options=list(DIMENSION_PRIORITY_LEVELS.keys()),
                    index=0  # Default: Ưu tiên 1 (Đúng kích thước)
                )
                region_priority = st.selectbox(
                    "Khu vực KH",
                    options=['Ưu tiên 1', 'Ưu tiên 2'],
                    format_func=lambda x: {
                        'Ưu tiên 1': '1 - Đúng khu vực',
                        'Ưu tiên 2': '2 - Tất cả khu vực',
                    }[x],
                    index=1  # Default: Ưu tiên 2 (Tất cả khu vực)
                )
            
            st.divider()
            st.markdown("#### 📅 Cài đặt tính toán giá")
            use_recent_only = st.checkbox(
                "Chỉ sử dụng dữ liệu gần nhất",
                value=True,
                help="Chỉ sử dụng N sản phẩm gần nhất (theo năm tài chính) để ước tính giá chính xác hơn"
            )
            recent_count = st.slider(
                "Số lượng sản phẩm tham khảo",
                min_value=5,
                max_value=35,
                value=15,
                step=5,
                help="Số lượng sản phẩm gần nhất sử dụng để ước tính giá",
                disabled=not use_recent_only
            )
            
            predict_btn = st.button("🔍 Tìm kiếm & Ước tính giá", type="primary", use_container_width=True)
        
        with col2:
            if predict_btn and st.session_state.model is not None:
                # Use similarity-based predictor
                predictor = st.session_state.model
                
                matches = predictor.find_matching_products(
                    stone_color_type=stone_color,
                    processing_code=processing_code,
                    length_cm=length,
                    width_cm=width,
                    height_cm=height,
                    application_codes=selected_applications,
                    customer_regional_group=customer_regional_group,
                    charge_unit=charge_unit,
                    stone_priority=stone_priority,
                    processing_priority=processing_priority,
                    dimension_priority=dimension_priority,
                    region_priority=region_priority,
                )
                
                estimation = predictor.estimate_price(
                    matches, 
                    use_recent_only=use_recent_only, 
                    recent_count=recent_count,
                    query_length_cm=length,
                    query_width_cm=width,
                    query_height_cm=height,
                    target_charge_unit=charge_unit,
                    stone_color_type=stone_color,
                    processing_code=processing_code
                )
                
                st.markdown("#### 📊 Kết quả ước tính")
                
                if estimation['estimated_price'] is not None:
                    # Confidence indicator
                    confidence_colors = {
                        'high': '#6bcb77',
                        'medium': '#ffd93d',
                        'low': '#ff6b6b',
                        'very_low': '#9e7cc1',
                    }
                    confidence_labels = {
                        'high': 'Cao (≥10 mẫu)',
                        'medium': 'Trung bình (5-9 mẫu)',
                        'low': 'Thấp (2-4 mẫu)',
                        'very_low': 'Rất thấp (1 mẫu)',
                    }
                    conf_color = confidence_colors.get(estimation['confidence'], '#808080')
                    conf_label = confidence_labels.get(estimation['confidence'], 'N/A')
                    
                    st.markdown(f"""
                    <div style="background-color: {conf_color}; padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
                        <h3 style="color: white; margin: 0;">Độ tin cậy: {conf_label}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Main estimated price
                    st.metric(f"💰 Giá ước tính ({charge_unit})", f"${estimation['estimated_price']:,.2f}")
                    
                    # Price range
                    st.markdown(f"Khoảng giá thực tế: **\\${estimation['min_price']:,.2f}** – **\\${estimation['max_price']:,.2f}**")
                    st.markdown(f"**Giá trung vị:** ${estimation['median_price']:,.2f}")
                    
                    # Show match count info with years if using recent only
                    if use_recent_only and estimation.get('total_matches', 0) > estimation['match_count']:
                        st.markdown(f"**Số mẫu khớp:** {estimation['match_count']} / {estimation['total_matches']} (sử dụng {estimation['match_count']} mẫu gần nhất)")
                        if estimation.get('years_used'):
                            st.markdown(f"**Năm tham khảo:** {estimation['years_used']}")
                    else:
                        st.markdown(f"**Số mẫu khớp:** {estimation['match_count']}")
                    
                    # Show price trend if available
                    if estimation.get('price_trend') and estimation.get('trend_pct') is not None:
                        trend_pct = estimation['trend_pct']
                        if estimation['price_trend'] == 'up':
                            st.markdown(f"📈 **Xu hướng giá:** Tăng **+{abs(trend_pct):.1f}%** so với năm trước")
                        elif estimation['price_trend'] == 'down':
                            st.markdown(f"📉 **Xu hướng giá:** Giảm **-{abs(trend_pct):.1f}%** so với năm trước")
                        else:
                            st.markdown(f"➡️ **Xu hướng giá:** Ổn định")
                    
                    st.divider()
                    
                    # Calculate segment for pricing (use first selected application or empty for classify_segment)
                    first_app = selected_applications[0] if selected_applications else ''
                    est_price_m3 = convert_price(
                        estimation['estimated_price'], charge_unit, 'USD/M3',
                        height_cm=height, length_cm=length, width_cm=width,
                        tlr=get_tlr(stone_color, processing_code)
                    )
                    segment = classify_segment(est_price_m3, height_cm=height, family=first_app, processing_code=processing_code)
                    
                    # Customer price adjustment with segment awareness
                    price_info = calculate_customer_price(
                        estimation['estimated_price'], customer_type, 
                        segment=segment, charge_unit=charge_unit
                    )
                    st.markdown(f"**👤 Giá theo khách hàng loại {customer_type}:**")
                    st.markdown(f"- {price_info['customer_description']}")
                    st.markdown(f"- Khoảng giá: **\\${price_info['min_price']:,.2f}** – **\\${price_info['max_price']:,.2f}**")
                    st.markdown(f"- Điều chỉnh: {price_info['adjustment_label']}")
                    st.markdown(f"- Quyền tự quyết: {price_info['authority_range']}")
                    
                else:
                    # Get detailed diagnostics for why no matches found
                    diagnostics = predictor.get_match_diagnostics(
                        stone_color_type=stone_color,
                        processing_code=processing_code,
                        length_cm=length,
                        width_cm=width,
                        height_cm=height,
                        application_codes=selected_applications,
                        customer_regional_group=customer_regional_group,
                        charge_unit=charge_unit,
                        stone_priority=stone_priority,
                        processing_priority=processing_priority,
                        dimension_priority=dimension_priority,
                        region_priority=region_priority,
                    )
                    
                    st.warning(f"⚠️ Không tìm thấy sản phẩm phù hợp")
                    
                    if diagnostics.get('reason'):
                        st.error(f"**Lý do:** {diagnostics['reason']}")
                    
                    if diagnostics.get('suggestions'):
                        st.info("**💡 Gợi ý:**\n" + "\n".join([f"• {s}" for s in diagnostics['suggestions']]))
                
                st.divider()
                
                # Product info summary with weight calculation
                st.markdown("**📦 Thông tin sản phẩm:**")
                volume_m3 = calculate_volume_m3(length, width, height)
                area_m2 = calculate_area_m2(length, width)
                tlr = get_tlr(stone_color, processing_code)
                first_app = selected_applications[0] if selected_applications else ''
                hs = get_hs_factor((length, width, height), processing_code, first_app)
                weight_tons = calculate_weight_tons(volume_m3, stone_color, processing_code, (length, width, height), first_app)
                
                col_info1, col_info2 = st.columns(2)
                with col_info1:
                    st.markdown(f"- Kích thước: {length} x {width} x {height} cm")
                    st.markdown(f"- Thể tích: {volume_m3:.6f} m³")
                    st.markdown(f"- Diện tích: {area_m2:.4f} m²")
                with col_info2:
                    st.markdown(f"- TLR: {tlr} tấn/m³")
                    st.markdown(f"- HS: {hs}")
                    st.markdown(f"- Khối lượng: **{weight_tons:.4f} tấn**")
                
        # ============ MATCHING PRODUCTS (Full Width) ============
        if predict_btn and st.session_state.model is not None:
            st.divider()
            st.markdown("#### 📋 Sản phẩm trong hệ thống khớp tiêu chí")
            
            if len(matches) > 0:
                st.success(f"✅ Tìm thấy **{len(matches)}** sản phẩm khớp tiêu chí!")
                
                # Statistics
                match_prices = matches['sales_price']
                stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                with stat_col1:
                    st.metric("Thấp nhất", f"${match_prices.min():,.2f}")
                with stat_col2:
                    st.metric("Cao nhất", f"${match_prices.max():,.2f}")
                with stat_col3:
                    st.metric("Trung bình", f"${match_prices.mean():,.2f}")
                with stat_col4:
                    st.metric("Trung vị", f"${match_prices.median():,.2f}")
                
                # Show table of ALL matches with Regional Group included
                display_cols = [
                    'contract_product_name', 'contract_name', 'account_code',
                    'customer_regional_group',  # Regional Group now visible
                    'sku', 'application_code', 'application',
                    'processing_code', 'processing_name',
                    'stone_color_type', 'segment',
                    'length_cm', 'width_cm', 'height_cm',
                    'charge_unit', 'sales_price', 'price_m3',
                    'created_date', 'fy_year',
                ]
                available_cols = [col for col in display_cols if col in matches.columns]
                
                # Column config for headers
                col_config = {
                    'sku': st.column_config.TextColumn('SKU'),
                    'application_code': st.column_config.TextColumn('App Code'),
                    'application': st.column_config.TextColumn('Application'),
                    'processing_code': st.column_config.TextColumn('Main Processing Code'),
                    'processing_name': st.column_config.TextColumn('Main Processing'),
                    'customer_regional_group': st.column_config.TextColumn('Regional Group'),
                }
                
                with st.expander(f"📋 Xem danh sách {len(matches)} sản phẩm khớp", expanded=True):
                    st.dataframe(matches[available_cols], use_container_width=True, height=300, column_config=col_config)
            else:
                st.info("⚠️ Không tìm thấy sản phẩm phù hợp. Thử mở rộng tiêu chí (Ưu tiên 2 hoặc 3).")
        
        elif predict_btn and st.session_state.model is None:
            st.warning("⚠️ Vui lòng chuẩn bị tìm kiếm trước (nút 🔍 ở sidebar)")
        elif not predict_btn:
            pass  # User hasn't clicked yet
    
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
        
        st.divider()
        
        # New charts section
        st.markdown("### 📊 Phân tích nâng cao")
        
        chart_col3, chart_col4 = st.columns(2)
        
        with chart_col3:
            # Price by Application (new column)
            if 'application' in df_clean.columns:
                app_prices = df_clean.groupby('application').agg({
                    'sales_price': ['mean', 'count']
                }).round(2)
                app_prices.columns = ['Giá TB', 'Số lượng']
                app_prices = app_prices.sort_values('Giá TB', ascending=True)
                
                fig_app = px.bar(
                    x=app_prices['Giá TB'].values,
                    y=app_prices.index,
                    orientation='h',
                    title="💰 Giá trung bình theo Ứng dụng (Application)",
                    labels={'x': 'Giá TB (USD)', 'y': 'Application'},
                    text=app_prices['Số lượng'].values
                )
                fig_app.update_traces(marker_color='#48bb78', texttemplate='n=%{text}', textposition='inside')
                st.plotly_chart(fig_app, use_container_width=True)
        
        with chart_col4:
            # Price by Processing type
            if 'processing_name' in df_clean.columns:
                proc_prices = df_clean.groupby('processing_name').agg({
                    'sales_price': ['mean', 'count']
                }).round(2)
                proc_prices.columns = ['Giá TB', 'Số lượng']
                proc_prices = proc_prices.sort_values('Giá TB', ascending=True)
                
                fig_proc = px.bar(
                    x=proc_prices['Giá TB'].values,
                    y=proc_prices.index,
                    orientation='h',
                    title="🔧 Giá trung bình theo Gia công (Processing)",
                    labels={'x': 'Giá TB (USD)', 'y': 'Processing'},
                    text=proc_prices['Số lượng'].values
                )
                fig_proc.update_traces(marker_color='#ed8936', texttemplate='n=%{text}', textposition='inside')
                st.plotly_chart(fig_proc, use_container_width=True)
        
        chart_col5, chart_col6 = st.columns(2)
        
        with chart_col5:
            # Price trend by year
            if 'fy_year' in df_clean.columns:
                yearly_data = df_clean.groupby('fy_year').agg({
                    'sales_price': ['mean', 'median', 'count'],
                    'price_m3': 'mean'
                }).round(2)
                yearly_data.columns = ['Giá TB', 'Giá Trung vị', 'Số đơn hàng', 'Giá/m³ TB']
                yearly_data = yearly_data.reset_index()
                yearly_data = yearly_data[yearly_data['fy_year'].notna()]
                
                fig_trend = px.line(
                    yearly_data,
                    x='fy_year',
                    y=['Giá TB', 'Giá Trung vị'],
                    title="📈 Xu hướng giá theo năm",
                    labels={'value': 'Giá (USD)', 'fy_year': 'Năm', 'variable': 'Loại giá'},
                    markers=True
                )
                fig_trend.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02))
                st.plotly_chart(fig_trend, use_container_width=True)
        
        with chart_col6:
            # Regional Group analysis
            if 'customer_regional_group' in df_clean.columns:
                region_data = df_clean[df_clean['customer_regional_group'].notna()]
                if len(region_data) > 0:
                    region_prices = region_data.groupby('customer_regional_group').agg({
                        'sales_price': ['mean', 'count'],
                        'price_m3': 'mean'
                    }).round(2)
                    region_prices.columns = ['Giá TB', 'Số đơn hàng', 'Giá/m³ TB']
                    region_prices = region_prices.sort_values('Giá TB', ascending=True).reset_index()
                    
                    fig_region = px.bar(
                        region_prices,
                        x='customer_regional_group',
                        y='Giá TB',
                        color='Giá/m³ TB',
                        title="🌍 Giá trung bình theo Khu vực khách hàng",
                        labels={'customer_regional_group': 'Nhóm Khu vực', 'Giá TB': 'Giá TB (USD)'},
                        text='Số đơn hàng',
                        color_continuous_scale='Blues'
                    )
                    fig_region.update_traces(texttemplate='n=%{text}', textposition='outside')
                    st.plotly_chart(fig_region, use_container_width=True)
        
        # Correlation heatmap for numeric columns
        st.markdown("#### 🔗 Tương quan giữa các yếu tố")
        numeric_cols = ['length_cm', 'width_cm', 'height_cm', 'volume_m3', 'area_m2', 'sales_price', 'price_m3']
        available_numeric = [col for col in numeric_cols if col in df_clean.columns]
        if len(available_numeric) >= 3:
            corr_matrix = df_clean[available_numeric].corr().round(2)
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=True,
                title="Ma trận tương quan (Correlation Matrix)",
                color_continuous_scale='RdBu_r',
                aspect='auto'
            )
            st.plotly_chart(fig_corr, use_container_width=True)
    
    # Tab 3: Similar Products
    with tab3:
        st.subheader("🔍 Tìm sản phẩm tương tự")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### Tiêu chí tìm kiếm")
            search_family = st.selectbox("Loại sản phẩm", [''] + PRODUCT_FAMILIES, key='search_family')
            search_stone = st.selectbox("Màu đá", [''] + STONE_COLOR_TYPES, key='search_stone')
            
            # Processing code dropdown with Vietnamese
            search_processing_lookup = {code: (eng, vn) for code, eng, vn in PROCESSING_CODES_SEARCH}
            search_processing = st.selectbox(
                "Gia công chính (Main Processing)",
                options=[code for code, eng, vn in PROCESSING_CODES_SEARCH],
                format_func=lambda x: f"{x} - {search_processing_lookup.get(x, ('All', 'Tất cả'))[0]} ({search_processing_lookup.get(x, ('All', 'Tất cả'))[1]})" if x else "All (Tất cả)",
                key='search_processing'
            )
            
            # Customer Regional Group filter
            search_regional_group = st.selectbox(
                "Nhóm Khu vực KH (Regional Group)",
                options=[code for code, name in CUSTOMER_REGIONAL_GROUPS],
                format_func=lambda x: x if x else "All",
                key='search_regional_group',
                help="Lọc theo nhóm khu vực khách hàng"
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
                if search_regional_group and 'customer_regional_group' in df_clean.columns:
                    exact_mask &= df_clean['customer_regional_group'] == search_regional_group
                if search_length > 0:
                    exact_mask &= df_clean['length_cm'] == search_length
                if search_width > 0:
                    exact_mask &= df_clean['width_cm'] == search_width
                if search_height > 0:
                    exact_mask &= df_clean['height_cm'] == search_height
                
                exact_matches = df_clean[exact_mask]
                
                # Include application and processing columns in display
                display_cols = ['contract_product_name', 'stone_color_type', 
                                'sku', 'application_code', 'application',
                                'processing_code', 'processing_name',
                                'customer_regional_group',
                                'length_cm', 'width_cm', 'height_cm', 'charge_unit', 'sales_price', 'price_m3', 'segment']
                available_cols = [col for col in display_cols if col in df_clean.columns]
                
                # Column config for English headers
                col_config = {
                    'sku': st.column_config.TextColumn('SKU'),
                    'application_code': st.column_config.TextColumn('App Code'),
                    'application': st.column_config.TextColumn('Application'),
                    'processing_code': st.column_config.TextColumn('Main Processing Code'),
                    'processing_name': st.column_config.TextColumn('Main Processing'),
                    'customer_regional_group': st.column_config.TextColumn('Regional Group'),
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
                    if search_regional_group and 'customer_regional_group' in df_clean.columns:
                        related_mask |= df_clean['customer_regional_group'] == search_regional_group
                    
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
    
    # Tab 4: Weight & Conversion Reference
    with tab4:
        st.subheader("📐 Bảng tra cứu TLR & Hệ số")
        
        if st.session_state.model_metrics is not None:
            metrics = st.session_state.model_metrics
            loaded = metrics.get('loaded_samples', 0)
            st.success(f"✅ Đã tải **{loaded:,}** sản phẩm có giá")
        
        st.divider()
        
        # TLR Reference Table
        st.markdown("#### ⚖️ Trọng Lượng Riêng (TLR)")
        tlr_data = {
            'Sản phẩm': [
                'Đá đen Đak Nông (Absolute Basalt)',
                'Đá Phước Hòa/Qui Nhơn (cưa cắt máy)',
                'Đá Phước Hòa/Qui Nhơn (chẻ tay)',
                'Dark Grey Granite',
                'Granite thường',
                'Bluestone (Thanh Hóa)',
                'Đá tổ ong'
            ],
            'TLR (tấn/m³)': ['2.95', '2.70', '2.65', '2.90', '2.70', '2.70', '2.20'],
            'Ghi chú': [
                'Hàng Dak Nông mỗi cont 9.3-9.6 m³',
                '',
                '',
                '',
                '',
                '',
                ''
            ]
        }
        st.dataframe(pd.DataFrame(tlr_data), use_container_width=True, hide_index=True)
        
        st.divider()
        
        # HS Factors Table
        st.markdown("#### 📊 Hệ Số Ốp Đáy (HS)")
        hs_data = {
            'Sản phẩm': [
                'Đá lát 6cm mặt đốt, cạnh sộ',
                'Đá cubic chẻ tay 5×5×5cm',
                'Đá cubic chẻ tay 8×8×8cm',
                'Đá cubic chẻ tay 10×10×8cm, 20×10×8cm',
                'Đá cubic chẻ tay 15×15×12cm',
                'Đá cubic mặt đốt, cạnh chẻ tay',
                'Đá cây cưa lột'
            ],
            'HS': ['0.97', '1.00', '0.95', '0.875', '0.85', '0.95', '1.05'],
            'Ghi chú': [
                'Ốp đáy giảm 3%',
                '',
                '',
                '',
                '',
                '',
                'Dày thực tế 10.5cm, +5%'
            ]
        }
        st.dataframe(pd.DataFrame(hs_data), use_container_width=True, hide_index=True)
        
        st.divider()
        
        # Formulas
        st.markdown("#### 📝 Công thức tính toán")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
**Tính m³ (Thể tích):**
```
m³ = (Dài × Rộng × Cao) / 1.000.000 × Số viên
```

**Tính m² (Diện tích):**
```
m² = (Dài × Rộng) / 10.000 × Số viên
```

**Tính Tấn (Trọng lượng):**
```
Tấn = m³ × TLR × HS
```
            """)
        with col2:
            st.markdown("""
**Quy đổi giá từ Viên:**
- `Giá/m² = Giá Viên ÷ D(m) ÷ R(m)`
- `Giá/m³ = Giá Viên ÷ D(m) ÷ R(m) ÷ C(m)`
- `Giá/Tấn = Giá Viên ÷ D ÷ R ÷ C ÷ TLR ÷ HS`

**Quy đổi giữa đơn vị:**
- `Giá/m² = Giá/m³ × Cao(m)`
- `Giá/m³ = Giá/Tấn × TLR × HS`
            """)
        
        st.divider()
        
        # Container weight reference
        st.markdown("#### 🚢 Quy chuẩn trọng lượng Container")
        container_data = {
            'Thị trường': ['Mỹ', 'Châu Âu', 'Úc', 'Nhật'],
            'Trọng lượng (tấn)': ['20-21', '27-28', '24-26', '27.5-28']
        }
        st.dataframe(pd.DataFrame(container_data), use_container_width=True, hide_index=True)
    
    # Tab 5: Detailed Data
    with tab5:
        st.subheader("📋 Dữ liệu chi tiết")
        
        # Filters
        filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
        with filter_col1:
            filter_family = st.multiselect("Loại sản phẩm", PRODUCT_FAMILIES)
        with filter_col2:
            filter_segment = st.multiselect("Phân khúc", ['Economy', 'Common', 'Premium', 'Super premium'])
        with filter_col3:
            filter_regional_group = st.multiselect(
                "Nhóm Khu vực KH", 
                [code for code, name in CUSTOMER_REGIONAL_GROUPS if code]
            )
        with filter_col4:
            price_range = st.slider("Khoảng giá (USD/m³)", 0, 2000, (0, 2000))
        
        # Apply filters
        filtered_df = st.session_state.data.copy()
        if filter_family:
            filtered_df = filtered_df[filtered_df['family'].isin(filter_family)]
        if filter_segment:
            filtered_df = filtered_df[filtered_df['segment'].isin(filter_segment)]
        if filter_regional_group and 'customer_regional_group' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['customer_regional_group'].isin(filter_regional_group)]
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
            'customer_regional_group', # Contract__r.Account__r.Nhom_Khu_vuc_KH__c
            'stone_color_type',        # Product__r.STONE_Color_Type__c
            'sku',                     # Product__r.StockKeepingUnit (SKU)
            'application_code',        # Application code (from SKU positions 3-5)
            'application',             # Application name (English)
            'application_vn',          # Application name (Vietnamese)
            'processing_code',         # Main processing code (from SKU)
            'processing_name',         # Main processing name (English)
            'family',                  # Product__r.Family (legacy)
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
            'volume_m3',               # Calculated volume
            'area_m2',                 # Calculated area
        ]
        
        # Filter to only columns that exist in the dataframe
        available_columns = [col for col in all_contract_columns if col in filtered_df.columns]
        
        # Add any remaining columns not in the predefined list
        remaining_columns = [col for col in filtered_df.columns if col not in available_columns]
        display_columns = available_columns + remaining_columns
        
        # Column configuration for English headers on specific columns
        column_config = {
            'sku': st.column_config.TextColumn('SKU', help='Product Stock Keeping Unit'),
            'application_code': st.column_config.TextColumn('App Code', help='Application code from SKU'),
            'application': st.column_config.TextColumn('Application', help='Application name (English)'),
            'application_vn': st.column_config.TextColumn('Application (VN)', help='Application name (Vietnamese)'),
            'processing_code': st.column_config.TextColumn('Main Processing Code', help='Ký hiệu gia công chính'),
            'processing_name': st.column_config.TextColumn('Main Processing', help='Nhóm mã gia công chính'),
            'customer_regional_group': st.column_config.TextColumn('Regional Group', help='Nhóm Khu vực KH'),
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
