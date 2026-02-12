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

# Application codes (SKU positions 3-4) with application names
# Per "Application Mapping - Application Mapping.pdf" and LaTeX docs
# Format: (code_value, display_name) where display shows "APP_NAME - Code(s)"
APPLICATION_CODES = [
    ('1.1', 'CUBE - 1.1'),                     # Cubes / Cobbles
    ('1.3', 'PAVING - 1.3'),                   # Paving stone / Paving slab
    ('1.4', 'CRAZY - 1.4'),                    # Crazy Paving
    ('2.1', 'WALL_STONE - 2.1'),               # Wall stone / Wall brick
    ('2.2', 'WALL_COVERING - 2.2'),            # Wall covering / Wall top
    ('2.3', 'ROCKFACE_WALLING - 2.3'),         # Rockface Walling
    ('3.1', 'PALISADE - 3.1'),                 # Palisades
    ('3.2', 'KERB - 3.2'),                     # Border / Kerbs
    ('3.3', 'CORNER - 3.3'),                   # Corner
    ('4.1,4.2', 'STEP - 4.1 & 4.2'),           # Step (Solid + Cladding)
    ('5.1', 'BLOCK - 5.1'),                    # Block
    ('6.1', 'POOL_SURROUNDING - 6.1'),         # Pool surrounding
    ('6.2', 'WINDOW_SILL - 6.2'),              # Window sill
    ('7.1,7.2,7.3', 'TILE - 7.1 & 7.2 & 7.3'), # Tile / Paver
    ('8.1', 'SKIRTINGS - 8.1'),                # Skirtings
    ('9.1', 'SLAB - 9.1'),                     # Slab
]

# Application codes for search (includes 'All' option)
APPLICATION_CODES_SEARCH = [('', 'All')] + APPLICATION_CODES

# Special product shapes (SKU position 13) per sku.tex
# These are non-standard shapes that require special handling
# Format: (code, Vietnamese name, English name)
SPECIAL_SHAPES = [
    ('R', 'Xẻ rãnh thoát nước', 'Drain Groove'),
    ('L', 'Cắt chữ L', 'L-Cut'),
    ('U', 'Cắt chữ U', 'U-Cut / U-Profile'),
    ('G', 'Cắt góc vuông', 'Corner Cut'),
    ('C', 'Cắt vòng cung', 'Arc Cut'),
    ('K', 'Lỗ khoan', 'Drill Hole'),
    ('T', 'Hình trụ', 'Cylinder'),
    ('B', 'Đá bộ', 'Set Stone'),
    ('V', 'Đá vành', 'Ring Stone'),
]

# Stone Color Types and their family groupings
# Based on sku.tex - Nguyên vật liệu (Vị trí 1, 2)
# Format: (internal_value, display_label)
# Stone classes for categorization
STONE_CLASSES = ['BASALT', 'GRANITE', 'BLUE STONE']

# Stone Color Types and their family groupings
# Based on sku.tex - Nguyên vật liệu (Vị trí 1, 2)
# Format: (internal_value, display_label)
STONE_COLOR_TYPES = [
    ('BD', 'BD - Basalt Black'),
    ('BX', 'BX - Basalt Grey'),
    ('BT', 'BT - Basalt Hive'),
    ('GX', 'GX - Granite Grey'),
    ('GT', 'GT - Granite White'),
    ('GV', 'GV - Granite Yellow'),
    ('GD', 'GD - Granite Red'),
    ('GH', 'GH - Granite Pink'),
    ('MB', 'MB - Marble Bluestone'),
    ('MT', 'MT - Marble White'),
    ('MV', 'MV - Marble Yellow'),
]

# Lookup for display labels
STONE_COLOR_LOOKUP = {code: label for code, label in STONE_COLOR_TYPES}

# Stone family mapping (for Priority 2 matching - same family)
STONE_FAMILY_MAP = {
    'BD': 'BASALT',
    'BX': 'BASALT',
    'BT': 'BASALT',
    'GX': 'GRANITE',
    'GT': 'GRANITE',
    'GV': 'GRANITE',
    'GD': 'GRANITE',
    'GH': 'GRANITE',
    'MB': 'MARBLE',
    'MT': 'MARBLE',
    'MV': 'MARBLE',
}

# Dimension tolerance levels per notes.md
DIMENSION_PRIORITY_LEVELS = {
    'Ưu tiên 1 - Đúng kích thước': {'height': 0, 'width': 0, 'length': 0},
    'Ưu tiên 2 - Sai lệch nhỏ': {'height': 1, 'width': 5, 'length': 10},
    'Ưu tiên 3 - Sai lệch lớn': {'height': 5, 'width': 20, 'length': 30},
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

# Processing Groups for Priority 2 matching (per Notes on Modifying the Pricing Tool.tex)
# Group: GIA CÔNG TAY (Hand Processing)
# Group: GIA CÔNG MÁY + TAY (Machine + Hand)
# Group: GIA CÔNG MÁY (Machine Processing)
# Group: GIA CÔNG MÁY CAO CẤP (High-end Machine)
PROCESSING_GROUPS = {
    'GIA_CONG_TAY': ['CTA', 'TLO', 'TDE'],  # Chẻ tay, Tự nhiên lồi, Tước đẽo
    'GIA_CONG_MAY_TAY': ['CUA', 'CLO', 'QME', 'GCT'],  # Cưa, Cưa lột, Quay mẻ, Giả cổ tay
    'GIA_CONG_MAY': ['DOT', 'DOC', 'DOX', 'GCR', 'MGI', 'PCA', 'BAM'],  # Đốt, Đốt chải, Đốt xịt, etc.
    'GIA_CONG_MAY_CAO_CAP': ['HON', 'BON', 'CHA'],  # Hone, Bóng, Chải
}

# Reverse mapping: code -> group name
PROCESSING_CODE_TO_GROUP = {}
for group_name, codes in PROCESSING_GROUPS.items():
    for code in codes:
        PROCESSING_CODE_TO_GROUP[code] = group_name

# Human-readable group names
PROCESSING_GROUP_NAMES = {
    'GIA_CONG_TAY': 'Gia công Tay',
    'GIA_CONG_MAY_TAY': 'Gia công Máy + Tay',
    'GIA_CONG_MAY': 'Gia công Máy',
    'GIA_CONG_MAY_CAO_CAP': 'Gia công Máy Cao cấp',
}

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


def calculate_special_shape_volume_m3(
    shape_code: str,
    length_cm: float,
    width_cm: float,
    height_cm: float,
    wall_thickness_cm: float = None,
    cut_leg_a_cm: float = None,
    cut_leg_b_cm: float = None,
    arc_radius_cm: float = None,
    arc_angle_degrees: float = 90.0,
    hole_count: int = None,
    hole_diameter_cm: float = None,
    hole_depth_cm: float = None,
    outer_radius_cm: float = None,
    inner_radius_cm: float = None,
) -> float:
    """
    Calculate volume in m³ for special shapes.
    Based on formulas from special_products_report.tex.
    
    Args:
        shape_code: One of L, U, G, C, K, T, B, V
        length_cm, width_cm, height_cm: Base dimensions in cm
        wall_thickness_cm: Wall thickness for L-shape and U-profile
        cut_leg_a_cm, cut_leg_b_cm: Cut dimensions for G (angle cut)
        arc_radius_cm: Radius for C (arc cut) and T (cylinder)
        arc_angle_degrees: Arc angle for C and V shapes (default 90°)
        hole_count, hole_diameter_cm, hole_depth_cm: Parameters for K (drilled hole)
        outer_radius_cm, inner_radius_cm: Radii for T (hollow cylinder) and V (ring)
    
    Returns:
        Volume in m³
    """
    import math
    
    # Convert cm to m
    L = length_cm / 100
    W = width_cm / 100
    H = height_cm / 100
    
    if shape_code == 'L':
        # L-profile: V = L × (t×W + t×H - t²)
        # Interpretation: L-profile cross-section extruded along length
        t = (wall_thickness_cm or 3) / 100  # Default 3cm wall
        area = t * W + t * H - t * t
        return L * area
    
    elif shape_code == 'U':
        # U-profile: V = L × (W×H - (W-2t)(H-t))
        t = (wall_thickness_cm or 3) / 100  # Default 3cm wall
        w_in = W - 2 * t
        h_in = H - t
        if w_in > 0 and h_in > 0:
            area = W * H - w_in * h_in
        else:
            area = W * H  # Fallback to solid if wall too thick
        return L * area
    
    elif shape_code == 'G':
        # Angle cut: V = (L×W - ½ab) × H
        a = (cut_leg_a_cm or 0) / 100
        b = (cut_leg_b_cm or 0) / 100
        cut_area = 0.5 * a * b
        plan_area = L * W - cut_area
        return plan_area * H
    
    elif shape_code == 'C':
        # Arc cut: V = (L×W - πr²×(θ/360)) × H
        r = (arc_radius_cm or 0) / 100
        theta = arc_angle_degrees or 90
        cut_area = math.pi * r * r * (theta / 360)
        plan_area = L * W - cut_area
        return plan_area * H
    
    elif shape_code == 'K':
        # Drilled hole: V = L×W×H - n×π(d/2)²×h
        base_volume = L * W * H
        n = hole_count or 0
        d = (hole_diameter_cm or 0) / 100
        h = (hole_depth_cm or height_cm) / 100  # Default to full depth
        hole_volume = n * math.pi * (d / 2) ** 2 * h
        return base_volume - hole_volume
    
    elif shape_code == 'T':
        # Cylinder: V = π(d/2)²×H or π(Ro²-Ri²)×H for hollow
        if outer_radius_cm and inner_radius_cm:
            # Hollow cylinder
            Ro = outer_radius_cm / 100
            Ri = inner_radius_cm / 100
            return math.pi * (Ro ** 2 - Ri ** 2) * H
        else:
            # Solid cylinder - use width as diameter
            d = W  # Already in meters
            return math.pi * (d / 2) ** 2 * H
    
    elif shape_code == 'B':
        # Set/kit: Use standard volume as approximation
        # Real calculation requires bill of materials
        return L * W * H
    
    elif shape_code == 'V':
        # Ring: V = π(Ro²-Ri²)×H×(θ/360)
        Ro = (outer_radius_cm or width_cm / 2) / 100
        Ri = (inner_radius_cm or 0) / 100
        theta = arc_angle_degrees or 360  # Full ring by default
        return math.pi * (Ro ** 2 - Ri ** 2) * H * (theta / 360)
    
    else:
        # Unknown shape: use rectangular volume
        return L * W * H


# Shape-specific input configuration
SPECIAL_SHAPE_INPUTS = {
    'R': {
        'name_vn': 'Hình chữ nhật',
        'name_en': 'Rectangular (Standard)',
        'inputs': [],  # No additional inputs for standard rectangular
        'formula': 'V = L × W × H',
    },
    'L': {
        'name_vn': 'Cắt chữ L',
        'name_en': 'L-Shape',
        'inputs': [
            {'key': 'wall_thickness_cm', 'label': 'Độ dày thành (t)', 'unit': 'cm', 'default': 3.0, 'min': 0.5, 'max': 20.0},
        ],
        'formula': 'V = L × (t×W + t×H - t²)',
    },
    'U': {
        'name_vn': 'Cắt chữ U',
        'name_en': 'U-Profile',
        'inputs': [
            {'key': 'wall_thickness_cm', 'label': 'Độ dày thành (t)', 'unit': 'cm', 'default': 3.0, 'min': 0.5, 'max': 20.0},
        ],
        'formula': 'V = L × (W×H - (W-2t)(H-t))',
    },
    'G': {
        'name_vn': 'Cắt góc vuông',
        'name_en': 'Corner Cut',
        'inputs': [
            {'key': 'cut_leg_a_cm', 'label': 'Cạnh cắt A', 'unit': 'cm', 'default': 5.0, 'min': 0.1, 'max': 100.0},
            {'key': 'cut_leg_b_cm', 'label': 'Cạnh cắt B', 'unit': 'cm', 'default': 5.0, 'min': 0.1, 'max': 100.0},
        ],
        'formula': 'V = (L×W - ½ab) × H',
    },
    'C': {
        'name_vn': 'Cắt vòng cung',
        'name_en': 'Arc Cut',
        'inputs': [
            {'key': 'arc_radius_cm', 'label': 'Bán kính cung (r)', 'unit': 'cm', 'default': 5.0, 'min': 0.1, 'max': 100.0},
            {'key': 'arc_angle_degrees', 'label': 'Góc cung (θ)', 'unit': '°', 'default': 90.0, 'min': 1.0, 'max': 360.0},
        ],
        'formula': 'V = (L×W - πr²×θ/360) × H',
    },
    'K': {
        'name_vn': 'Lỗ khoan',
        'name_en': 'Drilled Hole',
        'inputs': [
            {'key': 'hole_count', 'label': 'Số lỗ (n)', 'unit': '', 'default': 1, 'min': 1, 'max': 100, 'step': 1},
            {'key': 'hole_diameter_cm', 'label': 'Đường kính lỗ (d)', 'unit': 'cm', 'default': 2.0, 'min': 0.1, 'max': 50.0},
            {'key': 'hole_depth_cm', 'label': 'Độ sâu lỗ (h)', 'unit': 'cm', 'default': None, 'min': 0.1, 'max': 100.0},
        ],
        'formula': 'V = L×W×H - n×π(d/2)²×h',
    },
    'T': {
        'name_vn': 'Hình trụ',
        'name_en': 'Cylinder',
        'inputs': [
            {'key': 'outer_radius_cm', 'label': 'Bán kính ngoài (Ro)', 'unit': 'cm', 'default': None, 'min': 0.1, 'max': 100.0},
            {'key': 'inner_radius_cm', 'label': 'Bán kính trong (Ri)', 'unit': 'cm', 'default': 0, 'min': 0, 'max': 100.0},
        ],
        'formula': 'V = π(Ro²-Ri²)×H',
    },
    'B': {
        'name_vn': 'Đá bộ',
        'name_en': 'Set/Kit',
        'inputs': [],  # No additional inputs - uses raw prices
        'formula': 'V = ΣVi×qi (sum of components)',
        'note': 'Sử dụng giá gốc, không chuẩn hóa thể tích',
    },
    'V': {
        'name_vn': 'Đá vành',
        'name_en': 'Ring Stone',
        'inputs': [
            {'key': 'outer_radius_cm', 'label': 'Bán kính ngoài (Ro)', 'unit': 'cm', 'default': None, 'min': 0.1, 'max': 200.0},
            {'key': 'inner_radius_cm', 'label': 'Bán kính trong (Ri)', 'unit': 'cm', 'default': 0, 'min': 0, 'max': 200.0},
            {'key': 'arc_angle_degrees', 'label': 'Góc cung (θ)', 'unit': '°', 'default': 360.0, 'min': 1.0, 'max': 360.0},
        ],
        'formula': 'V = π(Ro²-Ri²)×H×(θ/360)',
    },
}


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


# ============ 3D Visualization Helpers ============
def get_processing_color(processing_code: str) -> str:
    """
    Get color for a processing code for 3D visualization.
    Colors are distinct for different processing types.
    """
    processing_colors = {
        # Hand processing - Earthy tones
        'CTA': '#8B4513',  # Saddle Brown - Chẻ tay tự nhiên
        'TLO': '#A0522D',  # Sienna - Tự nhiên lồi
        'TDE': '#CD853F',  # Peru - Tước đẽo
        
        # Machine + Hand - Mixed tones
        'CUA': '#708090',  # Slate Gray - Cưa
        'CLO': '#778899',  # Light Slate Gray - Cưa lột
        'QME': '#696969',  # Dim Gray - Quay mẻ
        'GCT': '#808080',  # Gray - Giả cổ tay
        
        # Machine processing - Cool tones
        'DOT': '#FF6347',  # Tomato - Đốt (Flamed)
        'DOC': '#FF4500',  # Orange Red - Đốt chải
        'DOX': '#DC143C',  # Crimson - Đốt xịt nước
        'HON': '#4682B4',  # Steel Blue - Hon/Mài mịn
        'BON': '#1E90FF',  # Dodger Blue - Bóng
        'BAM': '#2F4F4F',  # Dark Slate Gray - Băm
        'GCR': '#556B2F',  # Dark Olive Green - Giả cổ rung
        
        # High-end machine - Premium tones
        'MGI': '#9370DB',  # Medium Purple - Mài giấy
        'PCA': '#DDA0DD',  # Plum - Phun cát
    }
    return processing_colors.get(processing_code, '#CCCCCC')


def generate_3d_cuboid_html(length_cm: float, width_cm: float, height_cm: float,
                             surface_processing: Dict[str, str]) -> str:
    """
    Generate pure CSS 3D cuboid visualization.
    Uses CSS 3D transforms for reliable rendering without external dependencies.
    """
    # Get colors for each face
    proc_lookup = {code: (eng, vn) for code, eng, vn in PROCESSING_CODES}
    
    face_labels_vn = {
        'top': 'Trên', 'bottom': 'Đáy', 'front': 'Trước',
        'back': 'Sau', 'left': 'Trái', 'right': 'Phải',
    }
    
    face_colors = {}
    face_info = {}
    for face in ['top', 'bottom', 'front', 'back', 'left', 'right']:
        proc = surface_processing.get(face, 'CUA')
        face_colors[face] = get_processing_color(proc)
        proc_name = proc_lookup.get(proc, ('Unknown', 'Không xác định'))
        face_info[face] = f"{face_labels_vn[face]}: {proc} - {proc_name[1]}"
    
    # Build legend HTML
    legend_items = []
    unique_procs = list(set(surface_processing.values()))
    for proc in unique_procs:
        color = get_processing_color(proc)
        proc_name = proc_lookup.get(proc, ('Unknown', 'Không xác định'))
        legend_items.append(f'''
            <div style="display:flex;align-items:center;margin:4px 0;">
                <span style="width:18px;height:18px;background:{color};display:inline-block;margin-right:8px;border:2px solid #333;border-radius:3px;"></span>
                <span style="font-size:13px;"><b>{proc}</b> - {proc_name[1]}</span>
            </div>
        ''')
    legend_html = ''.join(legend_items)
    
    # Scale dimensions for CSS (max size ~200px for display)
    max_dim = max(length_cm, width_cm, height_cm)
    scale = 180 / max_dim
    w = int(length_cm * scale)  # CSS width (length)
    h = int(height_cm * scale)  # CSS height (height)
    d = int(width_cm * scale)   # CSS depth (width)
    
    html = f'''
    <style>
    .scene {{
        width: 100%;
        height: 380px;
        perspective: 800px;
        display: flex;
        justify-content: center;
        align-items: center;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 12px;
        position: relative;
    }}
    .cube-container {{
        width: {w}px;
        height: {h}px;
        position: relative;
        transform-style: preserve-3d;
        transform: rotateX(-25deg) rotateY(-35deg);
        animation: rotate 20s infinite linear;
        animation-play-state: running;
    }}
    .cube-container:hover {{
        animation-play-state: paused;
    }}
    @keyframes rotate {{
        from {{ transform: rotateX(-25deg) rotateY(-35deg); }}
        to {{ transform: rotateX(-25deg) rotateY(325deg); }}
    }}
    .face {{
        position: absolute;
        display: flex;
        justify-content: center;
        align-items: center;
        font-size: 14px;
        font-weight: bold;
        color: white;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.8);
        border: 2px solid rgba(0,0,0,0.5);
        box-sizing: border-box;
        transition: all 0.3s;
        cursor: pointer;
    }}
    .face:hover {{
        filter: brightness(1.2);
        z-index: 100;
    }}
    .face-front {{
        width: {w}px; height: {h}px;
        background: {face_colors['front']};
        transform: translateZ({d//2}px);
    }}
    .face-back {{
        width: {w}px; height: {h}px;
        background: {face_colors['back']};
        transform: rotateY(180deg) translateZ({d//2}px);
    }}
    .face-right {{
        width: {d}px; height: {h}px;
        background: {face_colors['right']};
        transform: rotateY(90deg) translateZ({w//2}px);
    }}
    .face-left {{
        width: {d}px; height: {h}px;
        background: {face_colors['left']};
        transform: rotateY(-90deg) translateZ({w//2}px);
    }}
    .face-top {{
        width: {w}px; height: {d}px;
        background: {face_colors['top']};
        transform: rotateX(90deg) translateZ({h//2}px);
    }}
    .face-bottom {{
        width: {w}px; height: {d}px;
        background: {face_colors['bottom']};
        transform: rotateX(-90deg) translateZ({h//2}px);
    }}
    .info-panel {{
        position: absolute;
        top: 12px;
        left: 12px;
        background: rgba(255,255,255,0.95);
        padding: 12px 15px;
        border-radius: 10px;
        z-index: 10;
        max-width: 220px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }}
    .info-title {{
        font-weight: bold;
        margin-bottom: 6px;
        font-size: 14px;
        color: #333;
    }}
    .info-text {{
        font-size: 12px;
        color: #555;
    }}
    .hint {{
        position: absolute;
        bottom: 12px;
        right: 12px;
        background: rgba(255,255,255,0.85);
        padding: 6px 12px;
        border-radius: 6px;
        font-size: 11px;
        color: #666;
    }}
    </style>
    <div class="scene">
        <div class="info-panel">
            <div class="info-title">📦 Kích thước</div>
            <div class="info-text">Dài: {length_cm}cm × Rộng: {width_cm}cm × Cao: {height_cm}cm</div>
            <hr style="margin:10px 0;border:none;border-top:1px solid #ddd;">
            <div class="info-title">🎨 Gia công bề mặt</div>
            {legend_html}
        </div>
        <div class="cube-container">
            <div class="face face-front" title="{face_info['front']}">Trước<br>{surface_processing.get('front', 'CUA')}</div>
            <div class="face face-back" title="{face_info['back']}">Sau<br>{surface_processing.get('back', 'CUA')}</div>
            <div class="face face-right" title="{face_info['right']}">Phải<br>{surface_processing.get('right', 'CUA')}</div>
            <div class="face face-left" title="{face_info['left']}">Trái<br>{surface_processing.get('left', 'CUA')}</div>
            <div class="face face-top" title="{face_info['top']}">Trên<br>{surface_processing.get('top', 'DOT')}</div>
            <div class="face face-bottom" title="{face_info['bottom']}">Đáy<br>{surface_processing.get('bottom', 'CUA')}</div>
        </div>
        <div class="hint">🔄 Tự động xoay | Di chuột để dừng</div>
    </div>
    '''
    
    return html


def get_texture_for_processing(processing_code: str) -> str:
    """Map processing code to texture filename."""
    texture_map = {
        # Flamed textures
        'DOT': 'flamed.png',
        'DOX': 'flamed.png',
        'DOC': 'flamed.png',
        # Polished textures
        'BON': 'polished.png',
        'DAB': 'polished.png',
        # Sawn textures
        'CUA': 'sawn.png',
        'CLO': 'sawn.png',
        'CUL': 'sawn.png',
        # Brushed textures
        'CHA': 'brushed.png',
        'LEC': 'brushed.png',
        # Honed textures
        'MAI': 'honed.png',
        'TLO': 'honed.png',
        'CTA': 'honed.png',
    }
    return texture_map.get(processing_code, 'sawn.png')


def generate_3d_textured_cuboid(length_cm: float, width_cm: float, height_cm: float,
                                 surface_processing: Dict[str, str]) -> str:
    """
    Generate Three.js 3D viewer with per-face textures based on processing codes.
    Uses base64 data URLs for reliable texture loading in Streamlit.
    """
    import base64
    import os
    
    # Get texture base64 data for each face
    texture_dir = os.path.join(os.path.dirname(__file__), 'assets', 'textures')
    
    face_textures = {}
    for face in ['top', 'bottom', 'front', 'back', 'left', 'right']:
        proc_code = surface_processing.get(face, 'CUA')
        texture_file = get_texture_for_processing(proc_code)
        texture_path = os.path.join(texture_dir, texture_file)
        
        try:
            with open(texture_path, 'rb') as f:
                img_data = base64.b64encode(f.read()).decode('utf-8')
                face_textures[face] = f"data:image/png;base64,{img_data}"
        except Exception as e:
            # Fallback to color if texture not found
            face_textures[face] = None
            print(f"Warning: Could not load texture for {face}: {e}")
    
    # Get processing lookup for labels
    proc_lookup = {code: (eng, vn) for code, eng, vn in PROCESSING_CODES}
    face_labels_vn = {
        'top': 'Trên', 'bottom': 'Đáy', 'front': 'Trước',
        'back': 'Sau', 'left': 'Trái', 'right': 'Phải',
    }
    
    # Build legend HTML
    legend_items = []
    unique_procs = list(set(surface_processing.values()))
    for proc in unique_procs:
        color = get_processing_color(proc)
        proc_name = proc_lookup.get(proc, ('Unknown', 'Không xác định'))
        texture_name = get_texture_for_processing(proc).replace('.png', '').capitalize()
        legend_items.append(f'''
            <div style="display:flex;align-items:center;margin:4px 0;">
                <span style="width:18px;height:18px;background:{color};display:inline-block;margin-right:8px;border:2px solid #333;border-radius:3px;"></span>
                <span style="font-size:12px;"><b>{proc}</b> - {proc_name[1]} ({texture_name})</span>
            </div>
        ''')
    legend_html = ''.join(legend_items)
    
    # Three.js HTML with textures
    html = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{ margin: 0; padding: 0; overflow: hidden; }}
            #threejs-container {{ width: 100%; height: 450px; position: relative; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); }}
            .info-panel {{ position: absolute; top: 10px; left: 10px; background: rgba(255,255,255,0.95); padding: 10px; border-radius: 8px; z-index: 10; max-width: 200px; box-shadow: 0 4px 15px rgba(0,0,0,0.2); }}
            .info-title {{ font-weight: bold; margin-bottom: 5px; font-size: 13px; color: #333; }}
            .info-text {{ font-size: 12px; color: #555; }}
            .hint {{ position: absolute; bottom: 10px; right: 10px; background: rgba(255,255,255,0.8); padding: 5px 10px; border-radius: 5px; font-size: 11px; z-index: 10; }}
            #loading {{ position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: white; font-size: 16px; z-index: 5; }}
        </style>
    </head>
    <body>
        <div id="loading">⏳ Loading 3D model...</div>
        <div id="threejs-container"></div>
        <div class="info-panel">
            <div class="info-title">📦 Kích thước</div>
            <div class="info-text">Dài: {length_cm}cm × Rộng: {width_cm}cm × Cao: {height_cm}cm</div>
            <hr style="margin:8px 0;border:none;border-top:1px solid #ddd;">
            <div class="info-title">🎨 Gia công</div>
            {legend_html}
        </div>
        <div class="hint">🖱️ Kéo để xoay | Cuộn để zoom</div>
        
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script>
        (function() {{
            // Wait for Three.js to load
            if (typeof THREE === 'undefined') {{
                setTimeout(arguments.callee, 50);
                return;
            }}
            
            document.getElementById('loading').style.display = 'none';
        const container = document.getElementById('threejs-container');
        const width = container.clientWidth || 800;
        const height = 450;
        
        // Scene setup
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
        renderer.setSize(width, height);
        renderer.setClearColor(0x000000, 0);
        container.appendChild(renderer.domElement);
        
        // Dimensions
        const l = {length_cm};
        const w = {width_cm};
        const h = {height_cm};
        
        // Texture URLs
        const textureUrls = {{
            right: "{face_textures.get('right', '')}",
            left: "{face_textures.get('left', '')}",
            top: "{face_textures.get('top', '')}",
            bottom: "{face_textures.get('bottom', '')}",
            front: "{face_textures.get('front', '')}",
            back: "{face_textures.get('back', '')}"
        }};
        
        // Fallback colors
        const colors = {{
            right: "{get_processing_color(surface_processing.get('right', 'CUA'))}",
            left: "{get_processing_color(surface_processing.get('left', 'CUA'))}",
            top: "{get_processing_color(surface_processing.get('top', 'DOT'))}",
            bottom: "{get_processing_color(surface_processing.get('bottom', 'CUA'))}",
            front: "{get_processing_color(surface_processing.get('front', 'CUA'))}",
            back: "{get_processing_color(surface_processing.get('back', 'CUA'))}"
        }};
        
        // Create materials with textures
        const loader = new THREE.TextureLoader();
        const materials = [];
        const faces = ['right', 'left', 'top', 'bottom', 'front', 'back'];
        
        faces.forEach(face => {{
            if (textureUrls[face] && textureUrls[face].startsWith('data:')) {{
                const texture = loader.load(textureUrls[face]);
                texture.wrapS = THREE.RepeatWrapping;
                texture.wrapT = THREE.RepeatWrapping;
                materials.push(new THREE.MeshStandardMaterial({{
                    map: texture,
                    roughness: 0.7,
                    metalness: 0.1
                }}));
            }} else {{
                materials.push(new THREE.MeshStandardMaterial({{
                    color: colors[face],
                    roughness: 0.7,
                    metalness: 0.1
                }}));
            }}
        }});
        
        // Create box geometry
        const geometry = new THREE.BoxGeometry(l, h, w);
        const cube = new THREE.Mesh(geometry, materials);
        cube.position.set(0, 0, 0);
        scene.add(cube);
        
        // Add edges for visibility
        const edges = new THREE.EdgesGeometry(geometry);
        const line = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({{ color: 0x000000 }}));
        line.position.copy(cube.position);
        scene.add(line);
        
        // Lighting - even on all sides
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        scene.add(ambientLight);
        
        const lights = [
            {{ pos: [100, 100, 100], intensity: 0.4 }},
            {{ pos: [-100, 100, -100], intensity: 0.3 }},
            {{ pos: [100, -100, -100], intensity: 0.3 }},
            {{ pos: [-100, -100, 100], intensity: 0.3 }}
        ];
        
        lights.forEach(l => {{
            const light = new THREE.DirectionalLight(0xffffff, l.intensity);
            light.position.set(l.pos[0], l.pos[1], l.pos[2]);
            scene.add(light);
        }});
        
        // Camera position
        const maxDim = Math.max(l, w, h);
        const distance = maxDim * 2.5;
        camera.position.set(distance, distance * 0.8, distance);
        camera.lookAt(0, 0, 0);
        
        // Orbit controls
        let isDragging = false;
        let prevMouse = {{ x: 0, y: 0 }};
        let rotation = {{ x: -0.3, y: 0.5 }};
        let autoRotate = true;
        
        container.addEventListener('mousedown', (e) => {{
            isDragging = true;
            autoRotate = false;
            prevMouse = {{ x: e.clientX, y: e.clientY }};
        }});
        
        container.addEventListener('mousemove', (e) => {{
            if (isDragging) {{
                rotation.y += (e.clientX - prevMouse.x) * 0.01;
                rotation.x += (e.clientY - prevMouse.y) * 0.01;
                rotation.x = Math.max(-Math.PI/2, Math.min(Math.PI/2, rotation.x));
                prevMouse = {{ x: e.clientX, y: e.clientY }};
            }}
        }});
        
        container.addEventListener('mouseup', () => {{ isDragging = false; }});
        container.addEventListener('mouseleave', () => {{ isDragging = false; }});
        
        container.addEventListener('wheel', (e) => {{
            e.preventDefault();
            const zoom = 1 + e.deltaY * 0.001;
            camera.position.multiplyScalar(zoom);
        }});
        
        // Animation
        function animate() {{
            requestAnimationFrame(animate);
            
            if (autoRotate) {{
                rotation.y += 0.005;
            }}
            
            cube.rotation.x = rotation.x;
            cube.rotation.y = rotation.y;
            line.rotation.x = rotation.x;
            line.rotation.y = rotation.y;
            
            renderer.render(scene, camera);
        }}
        
        // Start animation
        animate();
        
        // Handle window resize
        window.addEventListener('resize', function() {{
            const newWidth = container.clientWidth;
            camera.aspect = newWidth / height;
            camera.updateProjectionMatrix();
            renderer.setSize(newWidth, height);
        }});
    }})();
    </script>
    </body>
    </html>
    '''
    
    return html


def generate_cuboid_stl(length_cm: float, width_cm: float, height_cm: float) -> bytes:
    """
    Generate ASCII STL content for a cuboid mesh.
    
    Args:
        length_cm: Length in cm (X-axis)
        width_cm: Width in cm (Y-axis)
        height_cm: Height in cm (Z-axis)
    
    Returns:
        bytes: STL file content
    """
    # Vertices of the cuboid
    l, w, h = length_cm, width_cm, height_cm
    
    # 12 triangles (2 per face, 6 faces)
    triangles = [
        # Bottom face (z=0) - normal (0,0,-1)
        ((0,0,0), (l,w,0), (l,0,0), (0,0,-1)),
        ((0,0,0), (0,w,0), (l,w,0), (0,0,-1)),
        # Top face (z=h) - normal (0,0,1)
        ((0,0,h), (l,0,h), (l,w,h), (0,0,1)),
        ((0,0,h), (l,w,h), (0,w,h), (0,0,1)),
        # Front face (y=0) - normal (0,-1,0)
        ((0,0,0), (l,0,0), (l,0,h), (0,-1,0)),
        ((0,0,0), (l,0,h), (0,0,h), (0,-1,0)),
        # Back face (y=w) - normal (0,1,0)
        ((0,w,0), (l,w,h), (l,w,0), (0,1,0)),
        ((0,w,0), (0,w,h), (l,w,h), (0,1,0)),
        # Left face (x=0) - normal (-1,0,0)
        ((0,0,0), (0,0,h), (0,w,h), (-1,0,0)),
        ((0,0,0), (0,w,h), (0,w,0), (-1,0,0)),
        # Right face (x=l) - normal (1,0,0)
        ((l,0,0), (l,w,0), (l,w,h), (1,0,0)),
        ((l,0,0), (l,w,h), (l,0,h), (1,0,0)),
    ]
    
    # Generate ASCII STL
    stl_lines = ["solid cuboid"]
    for v1, v2, v3, normal in triangles:
        stl_lines.append(f"  facet normal {normal[0]} {normal[1]} {normal[2]}")
        stl_lines.append("    outer loop")
        stl_lines.append(f"      vertex {v1[0]} {v1[1]} {v1[2]}")
        stl_lines.append(f"      vertex {v2[0]} {v2[1]} {v2[2]}")
        stl_lines.append(f"      vertex {v3[0]} {v3[1]} {v3[2]}")
        stl_lines.append("    endloop")
        stl_lines.append("  endfacet")
    stl_lines.append("endsolid cuboid")
    
    return "\n".join(stl_lines).encode('ascii')


def generate_cuboid_3mf(length_cm: float, width_cm: float, height_cm: float,
                         surface_processing: Dict[str, str]) -> bytes:
    """
    Generate 3MF file content for a cuboid with per-face colors.
    3MF is a ZIP-based format that embeds geometry and materials in one file.
    
    Args:
        length_cm, width_cm, height_cm: Dimensions in cm
        surface_processing: Dict with processing codes for each face
    
    Returns:
        bytes: 3MF file content (ZIP archive)
    """
    import zipfile
    import io
    
    l, w, h = length_cm, width_cm, height_cm
    
    # 8 vertices of the cuboid
    vertices = [
        (0, 0, 0),   # 0
        (l, 0, 0),   # 1
        (l, w, 0),   # 2
        (0, w, 0),   # 3
        (0, 0, h),   # 4
        (l, 0, h),   # 5
        (l, w, h),   # 6
        (0, w, h),   # 7
    ]
    
    # Triangles for each face (vertex indices) with their processing codes
    face_triangles = {
        'bottom': [(0, 2, 1), (0, 3, 2)],
        'top':    [(4, 5, 6), (4, 6, 7)],
        'front':  [(0, 1, 5), (0, 5, 4)],
        'back':   [(2, 3, 7), (2, 7, 6)],
        'left':   [(0, 4, 7), (0, 7, 3)],
        'right':  [(1, 2, 6), (1, 6, 5)],
    }
    
    # Get unique colors and create color index mapping
    colors = {}
    color_idx = 0
    for face in ['top', 'bottom', 'front', 'back', 'left', 'right']:
        proc = surface_processing.get(face, 'CUA')
        color = get_processing_color(proc)
        if color not in colors:
            colors[color] = color_idx
            color_idx += 1
    
    # Build vertices XML
    vertices_xml = "\n".join([
        f'          <vertex x="{v[0]}" y="{v[1]}" z="{v[2]}" />'
        for v in vertices
    ])
    
    # Build triangles XML with color properties
    triangles_xml_parts = []
    for face, tris in face_triangles.items():
        proc = surface_processing.get(face, 'CUA')
        color = get_processing_color(proc)
        pid = colors[color] + 1  # 1-based index
        for tri in tris:
            triangles_xml_parts.append(
                f'          <triangle v1="{tri[0]}" v2="{tri[1]}" v3="{tri[2]}" pid="1" p1="{pid}" />'
            )
    triangles_xml = "\n".join(triangles_xml_parts)
    
    # Build basematerials (colors) XML
    basematerials_xml_parts = []
    sorted_colors = sorted(colors.items(), key=lambda x: x[1])
    for color_hex, idx in sorted_colors:
        # Convert hex to 3MF format (sRGB hex without #)
        basematerials_xml_parts.append(
            f'        <base name="Color{idx+1}" displaycolor="{color_hex.upper()}" />'
        )
    basematerials_xml = "\n".join(basematerials_xml_parts)
    
    # 3D Model XML
    model_xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<model unit="millimeter" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
       xmlns:m="http://schemas.microsoft.com/3dmanufacturing/material/2015/02">
  <metadata name="Title">Stone Cuboid</metadata>
  <metadata name="Designer">Stone Price Predictor</metadata>
  <metadata name="Description">Dimensions: {l}cm x {w}cm x {h}cm</metadata>
  <resources>
    <basematerials id="1">
{basematerials_xml}
    </basematerials>
    <object id="2" type="model">
      <mesh>
        <vertices>
{vertices_xml}
        </vertices>
        <triangles>
{triangles_xml}
        </triangles>
      </mesh>
    </object>
  </resources>
  <build>
    <item objectid="2" />
  </build>
</model>'''
    
    # Content Types XML
    content_types_xml = '''<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml" />
  <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml" />
</Types>'''
    
    # Relationships XML
    rels_xml = '''<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel" />
</Relationships>'''
    
    # Create ZIP archive in memory
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('[Content_Types].xml', content_types_xml)
        zf.writestr('_rels/.rels', rels_xml)
        zf.writestr('3D/3dmodel.model', model_xml)
    
    return buffer.getvalue()


def generate_cuboid_obj(length_cm: float, width_cm: float, height_cm: float,
                         surface_processing: Dict[str, str]) -> tuple:
    """
    Generate OBJ and MTL content for a cuboid mesh with materials.
    
    Args:
        length_cm, width_cm, height_cm: Dimensions in cm
        surface_processing: Dict with processing codes for each face
    
    Returns:
        tuple: (obj_content: bytes, mtl_content: bytes)
    """
    l, w, h = length_cm, width_cm, height_cm
    
    # 8 vertices of the cuboid
    vertices = [
        (0, 0, 0),   # 1
        (l, 0, 0),   # 2
        (l, w, 0),   # 3
        (0, w, 0),   # 4
        (0, 0, h),   # 5
        (l, 0, h),   # 6
        (l, w, h),   # 7
        (0, w, h),   # 8
    ]
    
    # Face definitions (vertex indices, 1-based) and their processing
    # OBJ face order: v1 v2 v3 v4 (counter-clockwise when viewed from outside)
    faces = {
        'bottom': {'verts': [1, 4, 3, 2], 'proc': surface_processing.get('bottom', 'CUA')},
        'top':    {'verts': [5, 6, 7, 8], 'proc': surface_processing.get('top', 'DOT')},
        'front':  {'verts': [1, 2, 6, 5], 'proc': surface_processing.get('front', 'CUA')},
        'back':   {'verts': [3, 4, 8, 7], 'proc': surface_processing.get('back', 'CUA')},
        'left':   {'verts': [1, 5, 8, 4], 'proc': surface_processing.get('left', 'CUA')},
        'right':  {'verts': [2, 3, 7, 6], 'proc': surface_processing.get('right', 'CUA')},
    }
    
    # Generate OBJ content
    obj_lines = [
        "# Stone Cuboid Model",
        f"# Dimensions: {l}cm x {w}cm x {h}cm",
        "# Generated by Stone Price Predictor",
        "",
        "mtllib stone_cuboid.mtl",
        ""
    ]
    
    # Add vertices
    for v in vertices:
        obj_lines.append(f"v {v[0]} {v[1]} {v[2]}")
    
    obj_lines.append("")
    
    # Add texture coordinates (simple UV mapping for each face)
    obj_lines.extend(["vt 0 0", "vt 1 0", "vt 1 1", "vt 0 1"])
    obj_lines.append("")
    
    # Add normals
    normals = {
        'bottom': (0, 0, -1),
        'top':    (0, 0, 1),
        'front':  (0, -1, 0),
        'back':   (0, 1, 0),
        'left':   (-1, 0, 0),
        'right':  (1, 0, 0),
    }
    for face_name, normal in normals.items():
        obj_lines.append(f"vn {normal[0]} {normal[1]} {normal[2]}")
    
    obj_lines.append("")
    
    # Add faces with materials
    normal_idx = 1
    for face_name, face_data in faces.items():
        proc = face_data['proc']
        verts = face_data['verts']
        
        obj_lines.append(f"usemtl {proc}")
        obj_lines.append(f"# {face_name} face")
        # Two triangles for the quad
        obj_lines.append(f"f {verts[0]}/1/{normal_idx} {verts[1]}/2/{normal_idx} {verts[2]}/3/{normal_idx}")
        obj_lines.append(f"f {verts[0]}/1/{normal_idx} {verts[2]}/3/{normal_idx} {verts[3]}/4/{normal_idx}")
        normal_idx += 1
    
    obj_content = "\n".join(obj_lines).encode('utf-8')
    
    # Generate MTL content with colors for each processing type
    proc_lookup = {code: (eng, vn) for code, eng, vn in PROCESSING_CODES}
    unique_procs = set(surface_processing.values())
    
    mtl_lines = [
        "# Material Library for Stone Cuboid",
        "# Generated by Stone Price Predictor",
        ""
    ]
    
    for proc in unique_procs:
        color = get_processing_color(proc)
        # Convert hex color to RGB (0-1 range)
        r = int(color[1:3], 16) / 255
        g = int(color[3:5], 16) / 255
        b = int(color[5:7], 16) / 255
        
        proc_name = proc_lookup.get(proc, ('Unknown', 'Unknown'))
        texture_file = get_texture_for_processing(proc)
        
        mtl_lines.extend([
            f"newmtl {proc}",
            f"# {proc_name[1]} ({proc_name[0]})",
            f"Kd {r:.4f} {g:.4f} {b:.4f}",  # Diffuse color
            f"Ka {r*0.3:.4f} {g*0.3:.4f} {b*0.3:.4f}",  # Ambient color
            "Ks 0.2 0.2 0.2",  # Specular color
            "Ns 50",  # Shininess
            "d 1.0",  # Opacity
            f"map_Kd textures/{texture_file}",  # Texture map
            ""
        ])
    
    mtl_content = "\n".join(mtl_lines).encode('utf-8')
    
    return obj_content, mtl_content


def generate_3d_cuboid(length_cm: float, width_cm: float, height_cm: float,
                       surface_processing: Dict[str, str]) -> go.Figure:
    """
    Generate a 3D cuboid visualization using Plotly (fallback).
    For better results, use generate_3d_cuboid_html() with st.components.html()
    """
    import numpy as np
    
    l, w, h = length_cm, width_cm, height_cm
    proc_lookup = {code: (eng, vn) for code, eng, vn in PROCESSING_CODES}
    
    traces = []
    vertices = np.array([
        [0, 0, 0], [l, 0, 0], [l, w, 0], [0, w, 0],
        [0, 0, h], [l, 0, h], [l, w, h], [0, w, h],
    ])
    
    faces = {
        'bottom': [0, 1, 2, 3, 0], 'top': [4, 5, 6, 7, 4],
        'front': [0, 1, 5, 4, 0], 'back': [3, 2, 6, 7, 3],
        'left': [0, 3, 7, 4, 0], 'right': [1, 2, 6, 5, 1],
    }
    face_labels_vn = {'top': 'Trên', 'bottom': 'Đáy', 'front': 'Trước', 'back': 'Sau', 'left': 'Trái', 'right': 'Phải'}
    
    for face_name, vertex_indices in faces.items():
        proc_code = surface_processing.get(face_name, 'CUA')
        color = get_processing_color(proc_code)
        proc_name = proc_lookup.get(proc_code, ('Unknown', 'Không xác định'))
        face_verts = vertices[vertex_indices]
        
        traces.append(go.Mesh3d(
            x=face_verts[:4, 0], y=face_verts[:4, 1], z=face_verts[:4, 2],
            i=[0, 0], j=[1, 2], k=[2, 3],
            color=color, opacity=0.85,
            name=f"{face_labels_vn[face_name]}: {proc_code}",
            hovertemplate=f"<b>{face_labels_vn[face_name]}</b><br>Gia công: {proc_code}<br>{proc_name[1]}<extra></extra>",
            showlegend=True, flatshading=True
        ))
    
    # Edges
    edge_pairs = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    edges_x, edges_y, edges_z = [], [], []
    for v1, v2 in edge_pairs:
        edges_x.extend([vertices[v1, 0], vertices[v2, 0], None])
        edges_y.extend([vertices[v1, 1], vertices[v2, 1], None])
        edges_z.extend([vertices[v1, 2], vertices[v2, 2], None])
    
    traces.append(go.Scatter3d(x=edges_x, y=edges_y, z=edges_z, mode='lines',
                               line=dict(color='black', width=4), showlegend=False, hoverinfo='skip'))
    
    fig = go.Figure(data=traces)
    max_dim = max(l, w, h)
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='Dài (cm)', range=[-max_dim*0.1, l + max_dim*0.1]),
            yaxis=dict(title='Rộng (cm)', range=[-max_dim*0.1, w + max_dim*0.1]),
            zaxis=dict(title='Cao (cm)', range=[-max_dim*0.1, h + max_dim*0.1]),
            aspectmode='data',
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.2), up=dict(x=0, y=0, z=1)),
        ),
        title=dict(text=f'🧊 {length_cm}×{width_cm}×{height_cm} cm', x=0.5),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.9)"),
        margin=dict(l=0, r=0, t=50, b=0), height=450
    )
    return fig



def calculate_multi_surface_price(
    base_price_m3: float,
    surface_processing: Dict[str, str],
    length_cm: float,
    width_cm: float, 
    height_cm: float,
    stone_color_type: str,
    custom_volume_m3: float = None
) -> Dict[str, Any]:
    """
    Calculate price for an object with different processing on each surface.
    
    Args:
        base_price_m3: Base price per m³
        surface_processing: Dict with keys 'top', 'bottom', 'front', 'back', 'left', 'right'
        length_cm, width_cm, height_cm: Dimensions
        stone_color_type: Stone type
        custom_volume_m3: Optional custom volume (from STL or special shape calculation)
    
    Returns:
        Dict with price breakdown
    """
    # Calculate area of each surface in m²
    surface_areas = {
        'top': (length_cm * width_cm) / 10000,
        'bottom': (length_cm * width_cm) / 10000,
        'front': (length_cm * height_cm) / 10000,
        'back': (length_cm * height_cm) / 10000,
        'left': (width_cm * height_cm) / 10000,
        'right': (width_cm * height_cm) / 10000,
    }
    
    total_surface_area = sum(surface_areas.values())
    
    # Use custom volume if provided (from STL or special shape), otherwise calculate
    volume_m3 = custom_volume_m3 if custom_volume_m3 is not None else calculate_volume_m3(length_cm, width_cm, height_cm)
    
    # Processing complexity factors (higher = more expensive)
    processing_factors = {
        'CTA': 0.8,   # Hand split - basic
        'TLO': 0.85,  # Natural raised
        'TDE': 0.9,   # Trimmed
        'CUA': 1.0,   # Sawn - baseline
        'CLO': 1.05,  # Stripped sawn
        'QME': 0.95,  # Tumbled
        'GCT': 1.1,   # Hand antiqued
        'DOT': 1.15,  # Flamed
        'DOC': 1.2,   # Flamed brushed
        'DOX': 1.25,  # Flamed water jet
        'HON': 1.3,   # Honed
        'BON': 1.5,   # Polished
        'BAM': 1.1,   # Bush-hammered
        'GCR': 1.15,  # Machine antiqued
        'MGI': 1.35,  # Paper polished
        'PCA': 1.2,   # Sandblasted
    }
    
    # Calculate weighted processing factor
    weighted_factor = 0
    for surface, area in surface_areas.items():
        proc_code = surface_processing.get(surface, 'CUA')
        factor = processing_factors.get(proc_code, 1.0)
        weighted_factor += (area / total_surface_area) * factor
    
    # Calculate price
    adjusted_price_m3 = base_price_m3 * weighted_factor
    
    # Multi-surface premium (complexity in production)
    unique_processes = len(set(surface_processing.values()))
    if unique_processes > 1:
        complexity_premium = 1 + (unique_processes - 1) * 0.03  # 3% per additional process type
    else:
        complexity_premium = 1.0
    
    final_price_m3 = adjusted_price_m3 * complexity_premium
    
    # Calculate price per piece
    price_per_piece = final_price_m3 * volume_m3
    
    # Calculate per m²
    price_per_m2 = final_price_m3 * (height_cm / 100)
    
    return {
        'base_price_m3': round(base_price_m3, 2),
        'adjusted_price_m3': round(adjusted_price_m3, 2),
        'final_price_m3': round(final_price_m3, 2),
        'price_per_piece': round(price_per_piece, 2),
        'price_per_m2': round(price_per_m2, 2),
        'weighted_factor': round(weighted_factor, 3),
        'complexity_premium': round((complexity_premium - 1) * 100, 1),
        'unique_processes': unique_processes,
        'volume_m3': round(volume_m3, 6),
        'total_surface_area_m2': round(total_surface_area, 4),
        'surface_areas': {k: round(v, 4) for k, v in surface_areas.items()},
    }


def _get_priority_text(priority_value) -> str:
    """Map priority strictness value to readable text."""
    try:
        val = float(str(priority_value).replace('%', ''))
        if val >= 90:
            return "Exact Match (P1)"
        elif val >= 70:
            return "Strict (P1-2)"
        elif val >= 50:
            return "Mixed (P1-3)"
        elif val >= 30:
            return "Relaxed (P2-4)"
        else:
            return "Flexible (All)"
    except (ValueError, TypeError):
        return str(priority_value)


def display_estimation_result(
    estimation: Dict[str, Any],
    price_info: Dict[str, Any],
    charge_unit: str,
    customer_type: str,
    conf_color: str,
    text_color: str,
    conf_label: str,
    apply_yearly_adjustment: bool = False,
    yearly_increase_pct: float = 0,
    is_manual: bool = False,
    manual_count: int = 0,
):
    """
    DRY helper function to display estimation results consistently across all 3 sections:
    - Main search (predict_btn)
    - Cached results (last_estimation)
    - Manual recalculation (manual_estimation)
    
    Returns the yearly_adj_info dict for use in report generation.
    """
    from datetime import datetime
    
    # === Calculate final prices with adjustments ===
    final_price = (price_info['min_price'] + price_info['max_price']) / 2
    final_min = price_info['min_price']
    final_max = price_info['max_price']
    
    # Yearly adjustment calculation
    year_adjustment_note = ""
    yearly_adj_info = None
    if apply_yearly_adjustment and yearly_increase_pct > 0:
        current_year = datetime.now().year
        avg_fy_year = estimation.get('avg_fy_year', current_year)
        if avg_fy_year and avg_fy_year < current_year:
            years_diff = current_year - int(avg_fy_year)
            adjustment_factor = (1 + yearly_increase_pct / 100) ** years_diff
            final_price *= adjustment_factor
            final_min *= adjustment_factor
            final_max *= adjustment_factor
            year_adjustment_note = f" (+{yearly_increase_pct:.1f}%/năm × {years_diff} năm)"
            yearly_adj_info = {
                'applied': True,
                'rate': yearly_increase_pct,
                'avg_year': avg_fy_year,
                'years_diff': years_diff,
                'adjusted_price': final_price,
            }
    
    # === Display Price Card ===
    title_suffix = " (từ sản phẩm đã chọn)" if is_manual else ""
    st.markdown(f"#### 💰 Giá cuối cùng cho khách hàng loại {customer_type}{title_suffix}")
    
    manual_info_line = f'<p style="color: {text_color}; margin: 5px 0;">📦 Số mẫu: {manual_count} sản phẩm được chọn</p>' if is_manual else ""
    
    st.markdown(f"""
    <div style="background-color: {conf_color}; padding: 20px; border-radius: 10px; margin-bottom: 10px;">
        <p style="color: {text_color}; margin: 0; font-size: 1.1em; font-weight: bold;">💵 Giá đề xuất ({charge_unit}):</p>
        <h1 style="color: {text_color}; margin: 5px 0; font-size: 3.5em;">${final_price:,.2f}</h1>
        <p style="color: {text_color}; margin: 0; font-size: 0.9em;">Khoảng giá: <b>${final_min:,.2f}</b> – <b>${final_max:,.2f}</b></p>
        <hr style="margin: 10px 0; border-top: 1px solid rgba(0,0,0,0.2);">
        <p style="color: {text_color}; margin: 5px 0;">👤 {price_info['customer_description']}</p>
        <p style="color: {text_color}; margin: 5px 0;">📊 Điều chỉnh: {price_info['adjustment_label']}{year_adjustment_note}</p>
        <p style="color: {text_color}; margin: 5px 0;">🎯 Quyền tự quyết: {price_info['authority_range']}</p>
        <p style="color: {text_color}; margin: 5px 0;">📈 Độ tin cậy: {conf_label}</p>
        {manual_info_line}
    </div>
    """, unsafe_allow_html=True)
    
    # === Display Expander with Details ===
    expander_title = "📋 Chi tiết ước tính cơ bản" + (" (từ sản phẩm đã chọn)" if is_manual else "")
    with st.expander(expander_title, expanded=False):
        col_detail1, col_detail2 = st.columns(2)
        with col_detail1:
            st.markdown("**💰 Giá gốc (Base Price):**")
            
            # Step-by-step calculation if we have dimension info in estimation
            length = estimation.get('query_length_cm', 0)
            width = estimation.get('query_width_cm', 0) 
            height = estimation.get('query_height_cm', 0)
            
            if length > 0 and width > 0 and height > 0:
                volume_m3 = (length * width * height) / 1_000_000
                area_m2 = (length * width) / 10_000
                st.markdown(f"• **Bước 1 - Thể tích:** {length}×{width}×{height} cm = **{volume_m3:.4f} m³** ({area_m2:.4f} m²)")
                
                sample_count = manual_count if is_manual else estimation['match_count']
                st.markdown(f"• **Bước 2 - Chuyển đổi:** Quy đổi {sample_count} mẫu về USD/m³")
                
                price_m3 = estimation.get('estimated_price_m3', 0)
                if price_m3 > 0:
                    st.markdown(f"• **Bước 3 - Giá TB (USD/m³):** ≈ **${price_m3:,.2f}** / m³")
                    
                    if charge_unit == 'USD/M3':
                        display_price = price_m3
                        st.markdown(f"• **Bước 4 - Giá gốc ({charge_unit}):** ${display_price:,.2f}")
                    elif charge_unit == 'USD/M2':
                        # USD/m³ → USD/m² = price_m3 × height(m)
                        display_price = price_m3 * (height / 100)
                        st.markdown(f"• **Bước 4 - Giá gốc ({charge_unit}):** ${price_m3:,.2f} × {height/100:.3f}m = ${display_price:,.2f}")
                    elif charge_unit == 'USD/PC':
                        display_price = price_m3 * volume_m3
                        st.markdown(f"• **Bước 4 - Giá gốc ({charge_unit}):** ${price_m3:,.2f} × {volume_m3:.4f} m³ = ${display_price:,.2f}")
                    else:
                        display_price = estimation['estimated_price']
                        st.markdown(f"• **Bước 4 - Giá gốc ({charge_unit}):** ${display_price:,.2f}")
                    
                    st.markdown("  *(Step 5-6: Yearly & Customer adjustments applied above)*")
                else:
                    st.markdown(f"• Giá trung bình: **${estimation['estimated_price']:,.2f}** ({charge_unit})")
            else:
                st.markdown(f"• Giá trung bình: **${estimation['estimated_price']:,.2f}** ({charge_unit})")
            
            st.markdown("---")
            st.markdown(f"• Khoảng giá: \\${estimation['min_price']:,.2f} - \\${estimation['max_price']:,.2f}")
            st.markdown(f"• Giá trung vị: ${estimation.get('median_price', estimation['estimated_price']):,.2f}")
            sample_count = manual_count if is_manual else estimation['match_count']
            total_matches = estimation.get('total_matches', sample_count)
            st.markdown(f"• Số mẫu sử dụng: **{sample_count}** / {total_matches}")
            if estimation.get('years_used'):
                st.markdown(f"• Năm dữ liệu: {estimation['years_used']}")
            if estimation.get('price_trend'):
                trend_pct = estimation.get('trend_pct', 0)
                trend_emoji = '📈' if estimation['price_trend'] == 'up' else '📉'
                trend_sign = '+' if estimation['price_trend'] == 'up' else '-'
                # YoY label for clarity
                st.markdown(f"• {trend_emoji} Xu hướng (YoY): {trend_sign}{abs(trend_pct):.1f}%")
        
        with col_detail2:
            if estimation.get('confidence_breakdown'):
                st.markdown("**📊 Độ tin cậy:**")
                breakdown = estimation['confidence_breakdown']
                
                # Enhanced factor names with clearer formatting
                factor_config = {
                    'sample_count': {'name': 'Số mẫu', 'format': lambda v: f"{v} samples"},
                    'recency': {'name': 'Độ mới', 'format': lambda v: f"{v}"},
                    'dimensional': {'name': 'Kích thước', 'format': lambda v: f"{v}"},
                    'stone_match': {'name': 'Loại đá', 'format': lambda v: f"{v}"},
                    'processing_match': {'name': 'Gia công', 'format': lambda v: f"{v}"},
                    'application_match': {'name': 'Ứng dụng', 'format': lambda v: f"{v}"},
                    'charge_unit_match': {'name': 'Đơn vị', 'format': lambda v: f"{v}"},
                    'priority_strictness': {'name': 'Tiêu chí', 'format': lambda v: _get_priority_text(v)},
                }
                
                for key, info in breakdown.items():
                    config = factor_config.get(key, {'name': key, 'format': lambda v: str(v)})
                    name = config['name']
                    score = info.get('score', 0)
                    value = info.get('value', '')
                    
                    # Format: "X samples (Score: 75)" for sample_count
                    # Format: "Mixed Priorities (Score: 56)" for priority_strictness
                    if value:
                        formatted_value = config['format'](value)
                        st.markdown(f"• {name}: {formatted_value} (Score: **{score:.0f}**)")
                    else:
                        st.markdown(f"• {name}: Score **{score:.0f}**")
    
    return yearly_adj_info, final_price, final_min, final_max


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
    
    # Guard against None values
    if base_price is None or base_price <= 0:
        base_price = 0
    adj_min = adj.get('min', 0) or 0
    adj_max = adj.get('max', 0) or 0
    
    min_price = round(base_price * (1 + adj_min), 2)
    max_price = round(base_price * (1 + adj_max), 2)
    
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

def generate_price_report(
    query_params: Dict[str, Any],
    estimation: Dict[str, Any],
    matched_products: pd.DataFrame,
    customer_price_info: Dict[str, Any] = None,
    yearly_adjustment: Dict[str, Any] = None,
    priority_settings: Dict[str, str] = None
) -> str:
    """
    Generate an HTML report for price calculation that can be printed to PDF.
    
    Per manager's notes: Report includes selected options, data/records used for prediction,
    step-by-step formula explanation, and DateTime of calculation.
    Company: A PLUS MINERAL MATERIAL CORPORATION
    """
    from datetime import datetime
    
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    
    # Default priority settings
    if not priority_settings:
        priority_settings = {}
    
    # Calculate query volume
    length = query_params.get('length', 0)
    width = query_params.get('width', 0)
    height = query_params.get('height', 0)
    query_volume = (length * width * height) / 1_000_000
    
    # Define priority details
    priority_details = {
        'stone': priority_settings.get('stone_priority', 'N/A'),
        'processing': priority_settings.get('processing_priority', 'N/A'),
        'dimension': priority_settings.get('dimension_priority', 'N/A'),
        'region': priority_settings.get('region_priority', 'N/A')
    }
    
    # Generate detailed descriptions
    stone_desc = "Exact Match (Cùng loại đá)" if 'Ưu tiên 1' in str(priority_details['stone']) else "Family Match (Cùng nhóm màu)"
    proc_desc = "Exact Match (Cùng mã gia công)" if 'Ưu tiên 1' in str(priority_details['processing']) else "Group Match (Cùng nhóm gia công)"
    region_desc = "Regional Group Match (Cùng khu vực)" if 'Ưu tiên 1' in str(priority_details['region']) else "Billing Country / All (Mở rộng thị trường)"
    
    dim_p = str(priority_details['dimension'])
    if 'Ưu tiên 1' in dim_p:
        dim_desc = "Exact Dimensions (Δ=0)"
    elif 'Ưu tiên 2' in dim_p:
        dim_desc = "Small Deviation (H±1, W±5, L±10 cm)"
    elif 'Ưu tiên 3' in dim_p:
        dim_desc = "Large Deviation (H±5, W±15, L±30 cm)"
    else:
        dim_desc = "Custom / Unknown"

    # Build HTML report
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Stone Price Report - {timestamp}</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; font-size: 11px; }}
        h1 {{ color: #1f4e79; border-bottom: 2px solid #1f4e79; padding-bottom: 10px; font-size: 18px; }}
        h2 {{ color: #333; margin-top: 25px; font-size: 14px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 6px; text-align: left; }}
        th {{ background-color: #1f4e79; color: white; font-size: 10px; }}
        td {{ font-size: 10px; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .highlight {{ background-color: #e8f4fd; font-weight: bold; }}
        .price {{ font-size: 1.2em; color: #2e7d32; }}
        .company {{ color: #1f4e79; font-weight: bold; font-size: 12px; }}
        .step {{ background-color: #f5f5f5; padding: 8px; margin: 5px 0; border-left: 3px solid #1f4e79; }}
        .formula {{ font-family: monospace; background-color: #eee; padding: 2px 5px; }}
        .footer {{ margin-top: 30px; font-size: 0.85em; color: #666; border-top: 1px solid #ddd; padding-top: 10px; }}
        @media print {{ body {{ margin: 0; }} }}
    </style>
</head>
<body>
    <h1>💎 Stone Price Report</h1>
    <p><span class="company">A PLUS MINERAL MATERIAL CORPORATION</span></p>
    <p><strong>Report DateTime:</strong> {timestamp}</p>
    
    <h2>📋 Query Parameters</h2>
    <table>
        <tr><th>Parameter</th><th>Value</th></tr>
        <tr><td>Stone Color</td><td>{query_params.get('stone_color', 'N/A')}</td></tr>
        <tr><td>Dimensions (L×W×H)</td><td>{length}×{width}×{height} cm</td></tr>
        <tr><td>Query Volume</td><td>{query_volume:.6f} m³</td></tr>
        <tr><td>Processing</td><td>{query_params.get('processing_code', 'N/A')}</td></tr>
        <tr><td>Regional Group</td><td>{query_params.get('regional_group', 'N/A')}</td></tr>
        <tr><td>Application</td><td>{', '.join(query_params.get('applications', [])) or 'All'}</td></tr>
        <tr><td>Charge Unit</td><td>{query_params.get('charge_unit', 'USD/M3')}</td></tr>
        <tr><td>Customer Type</td><td>{query_params.get('customer_type', 'C')}</td></tr>
    </table>
    
    <h2>🎚️ Search Priority Settings (Mức độ ưu tiên tìm kiếm)</h2>
    <table>
        <tr><th>Criterion</th><th>Priority Level</th><th>Details</th></tr>
        <tr><td>Stone Type (Loại đá)</td><td>{priority_details['stone']}</td><td>{stone_desc}</td></tr>
        <tr><td>Processing (Gia công)</td><td>{priority_details['processing']}</td><td>{proc_desc}</td></tr>
        <tr><td>Dimensions (Kích thước)</td><td>{priority_details['dimension']}</td><td>{dim_desc}</td></tr>
        <tr><td>Market (Thị trường)</td><td>{priority_details['region']}</td><td>{region_desc}</td></tr>
    </table>
    
    <h2>💰 Price Estimation</h2>
    <table>
        <tr><td>Estimated Price</td><td>${estimation.get('estimated_price', 0):,.2f} {query_params.get('charge_unit', '')}</td></tr>
        <tr><td>Price Range</td><td>${estimation.get('min_price', 0):,.2f} – ${estimation.get('max_price', 0):,.2f}</td></tr>
        <tr><td>Median Price</td><td>${estimation.get('median_price', 0):,.2f}</td></tr>
        <tr><td>Match Count</td><td>{estimation.get('match_count', 0)} products</td></tr>
        <tr><td>Reference Years</td><td>{estimation.get('years_used', 'N/A')}</td></tr>
        <tr><td>Confidence</td><td>{estimation.get('confidence', 'N/A')}</td></tr>
    </table>
"""
    
    # Add yearly adjustment if present
    if yearly_adjustment and yearly_adjustment.get('applied'):
        html += f"""
    <h2>📈 Yearly Price Adjustment</h2>
    <table>
        <tr><td>Yearly Increase Rate</td><td>{yearly_adjustment.get('rate', 0):.1f}%</td></tr>
        <tr><td>Average Reference Year</td><td>{yearly_adjustment.get('avg_year', 'N/A')}</td></tr>
        <tr><td>Years Difference</td><td>{yearly_adjustment.get('years_diff', 0)} years</td></tr>
        <tr class="highlight"><td>Adjusted Price ({now.year})</td><td class="price">${yearly_adjustment.get('adjusted_price', 0):,.2f}</td></tr>
    </table>
"""
    
    # Add customer price info if present
    if customer_price_info:
        html += f"""
    <h2>👤 Customer Price Adjustment (Type {query_params.get('customer_type', 'C')})</h2>
    <table>
        <tr><td>Customer Description</td><td>{customer_price_info.get('customer_description', 'N/A')}</td></tr>
        <tr><td>Adjustment</td><td>{customer_price_info.get('adjustment_label', 'N/A')}</td></tr>
        <tr><td>Price Range</td><td>${customer_price_info.get('min_price', 0):,.2f} – ${customer_price_info.get('max_price', 0):,.2f}</td></tr>
        <tr><td>Authority Range</td><td>{customer_price_info.get('authority_range', 'N/A')}</td></tr>
    </table>
"""
    
    # Add matched products summary with ALL fields
    if len(matched_products) > 0:
        html += """
    <h2>Products Used for Estimation</h2>
    <table>
        <tr>
            <th>#</th>
            <th>SKU</th>
            <th>Main Processing</th>
            <th>Application</th>
            <th>L×W×H (cm)</th>
            <th>Price/m³</th>
            <th>Price/m²</th>
            <th>Original Price</th>
            <th>Unit</th>
            <th>Wt.Dens</th>
            <th>Size.F</th>
            <th>Year</th>
        </tr>
"""
        # Create lookup dicts for names (handle 3-element tuples: code, english, vietnamese)
        proc_lookup = {code: en for code, en, vn in PROCESSING_CODES}
        
        for i, (_, row) in enumerate(matched_products.head(20).iterrows(), 1):
            sku = str(row.get('sku', 'N/A'))[:15]
            proc_code = row.get('processing_code', 'N/A')
            proc_name = proc_lookup.get(proc_code, proc_code)[:20]
            
            # Get application name directly from dataframe (populated by salesforce_loader)
            app_name = row.get('application', 'N/A') or 'N/A'
            # Truncate if too long (e.g. "Paving stone / Paving slab" -> "Paving stone / Pa...")
            if len(app_name) > 25:
                app_name = app_name[:25]
            
            dims = f"{row.get('length_cm', 0):.0f}×{row.get('width_cm', 0):.0f}×{row.get('height_cm', 0):.0f}"
            
            # Calculate price/m³ and price/m² 
            l_cm = row.get('length_cm', 0) or 0
            w_cm = row.get('width_cm', 0) or 0
            h_cm = row.get('height_cm', 0) or 0
            original_price = row.get('sales_price', 0) or 0
            unit = row.get('charge_unit', 'N/A')
            tlr_val = row.get('specific_gravity', 2.7) or 2.7
            hs_val = row.get('hs_coefficient', 1.0) or 1.0
            
            # Convert to get price per m³ and m²
            vol_m3 = (l_cm * w_cm * h_cm) / 1_000_000 if l_cm and w_cm and h_cm else 0.001
            area_m2 = (l_cm * w_cm) / 10_000 if l_cm and w_cm else 0.01
            
            if unit == 'USD/M3':
                price_m3 = original_price
                price_m2 = original_price * (h_cm / 100) if h_cm else 0
            elif unit == 'USD/M2':
                price_m2 = original_price
                price_m3 = original_price / (h_cm / 100) if h_cm else 0
            elif unit == 'USD/PC':
                price_m3 = original_price / vol_m3 if vol_m3 > 0 else 0
                price_m2 = original_price / area_m2 if area_m2 > 0 else 0
            elif unit == 'USD/TON':
                price_m3 = original_price * tlr_val * hs_val
                price_m2 = price_m3 * (h_cm / 100) if h_cm else 0
            else:
                price_m3 = original_price
                price_m2 = original_price
            
            # Weight Density (TLR) and Size Factor (HS)
            tlr_str = f"{tlr_val:.2f}" if pd.notna(tlr_val) else "2.70"
            hs_str = f"{hs_val:.2f}" if pd.notna(hs_val) else "1.00"
            
            # Format year as integer
            year_val = row.get('fy_year')
            try:
                year = f"{int(float(year_val))}"
            except (ValueError, TypeError):
                year = "N/A"
            
            html += f"        <tr><td>{i}</td><td>{sku}</td><td>{proc_name}</td><td>{app_name}</td><td>{dims}</td><td>${price_m3:,.2f}</td><td>${price_m2:,.2f}</td><td>${original_price:,.2f}</td><td>{unit}</td><td>{tlr_str}</td><td>{hs_str}</td><td>{year}</td></tr>\n"
        
        if len(matched_products) > 20:
            html += f"        <tr><td colspan='12'>... and {len(matched_products) - 20} more products</td></tr>\n"
        html += "    </table>\n"
    
    # Add step-by-step calculation
    html += f"""
    <h2>📐 Step-by-Step Calculation</h2>
    
    <div class="step">
        <strong>Step 1: Calculate Query Volume</strong><br>
        <span class="formula">Volume = (L × W × H) / 1,000,000</span><br>
        Volume = ({length} × {width} × {height}) / 1,000,000 = <strong>{query_volume:.6f} m³</strong>
    </div>
    
    <div class="step">
        <strong>Step 2: Normalize Product Prices to USD/M³</strong><br>
        For each matched product, convert price to USD/M³ using TLR and HS:<br>
        <span class="formula">Price_M3 = convert_price(price, unit, 'USD/M3', dimensions, TLR, HS)</span><br>
        - If USD/PC: Price_M3 = Price / Product_Volume<br>
        - If USD/M2: Price_M3 = Price / Height(m)<br>
        - If USD/TON: Price_M3 = Price × TLR × HS
    </div>
    
    <div class="step">
        <strong>Step 3: Calculate Weighted Average (USD/M³)</strong><br>
        <span class="formula">Avg_Price_M3 = Σ(Price_M3 × Recency_Weight) / Σ(Recency_Weight)</span><br>
        Average Price (M³): <strong>${estimation.get('price_m3', estimation.get('estimated_price', 0)):,.2f}</strong>
    </div>
    
    <div class="step">
        <strong>Step 4: Convert to Target Unit ({query_params.get('charge_unit', 'USD/PC')})</strong><br>
        <span class="formula">Final_Price = Avg_Price_M3 × Query_Volume (for USD/PC)</span><br>
        <span class="formula">Final_Price = Avg_Price_M3 × Height_m (for USD/M2)</span><br>
        Estimated Price: <strong>${estimation.get('estimated_price', 0):,.2f}</strong>
    </div>
"""
    
    step_num = 5
    if yearly_adjustment and yearly_adjustment.get('applied'):
        html += f"""
    <div class="step">
        <strong>Step {step_num}: Apply Yearly Adjustment</strong><br>
        <span class="formula">Adjusted = Base × (1 + Rate%)^Years</span><br>
        Adjusted = ${estimation.get('estimated_price', 0):,.2f} × (1 + {yearly_adjustment.get('rate', 0):.1f}%)^{yearly_adjustment.get('years_diff', 0)} = <strong>${yearly_adjustment.get('adjusted_price', 0):,.2f}</strong>
    </div>
"""
        step_num += 1
    
    if customer_price_info:
        final_price = (customer_price_info.get('min_price', 0) + customer_price_info.get('max_price', 0)) / 2
        html += f"""
    <div class="step">
        <strong>Step {step_num}: Apply Customer Type Adjustment</strong><br>
        Customer Type: {query_params.get('customer_type', 'C')} - {customer_price_info.get('customer_description', 'N/A')}<br>
        Adjustment: {customer_price_info.get('adjustment_label', 'N/A')}<br>
        <strong>Final Price Range: ${customer_price_info.get('min_price', 0):,.2f} – ${customer_price_info.get('max_price', 0):,.2f}</strong>
    </div>
"""
    
    # Footer
    html += """
    <div class="footer">
        <p><strong>A PLUS MINERAL MATERIAL CORPORATION</strong></p>
        <p>Generated by Stone Price Predictor | Report Date: """ + timestamp + """</p>
        <p>To save as PDF: Print this page (Ctrl+P) and select "Save as PDF"</p>
    </div>
</body>
</html>
"""
    return html

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
        get_all_charge_units: bool = False,  # If True, skip charge_unit filter
        stone_priority: str = 'Ưu tiên 1',  # Exact, Same Family, All
        processing_priority: str = 'Ưu tiên 1',  # Exact, Group, All
        dimension_priority: str = 'Ưu tiên 1 - Đúng kích thước',
        region_priority: str = 'Ưu tiên 1',  # Billing Country, Regional Group, All
        no_length_limit: bool = False,  # For P3: unlimited length
        billing_country: str = None,  # For P1 market: specific country
        selected_processing_group: str = None,  # For P2: user-selected processing group
        special_shape: str = None,  # Special shape code (L, U, G, etc.)
    ) -> pd.DataFrame:
        """
        Find matching products based on priority criteria from notes.md.
        
        Priority Levels:
        - Ưu tiên 1: Exact match
        - Ưu tiên 2: Same family / group / small tolerance
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
        
        # 2. Processing Filter with Group Support
        if processing_priority == 'Ưu tiên 1' and processing_code:
            # Exact match
            mask &= df['processing_code'] == processing_code
        elif processing_priority == 'Ưu tiên 2':
            # Group match: use user-selected group or derive from processing_code
            if selected_processing_group and selected_processing_group in PROCESSING_GROUPS:
                group_codes = PROCESSING_GROUPS.get(selected_processing_group, [])
            else:
                query_group = PROCESSING_CODE_TO_GROUP.get(processing_code)
                group_codes = PROCESSING_GROUPS.get(query_group, [processing_code]) if query_group else [processing_code]
            mask &= df['processing_code'].isin(group_codes)
        # Ưu tiên 3: No filter (All processing types)
        
        # 3. Application Filter (extracted from SKU positions 3-4)
        # If application_codes is not empty, filter by those codes
        if application_codes and len(application_codes) > 0 and 'application_code' in df.columns:
            # Handle comma-separated codes like "4.1,4.2" and "7.1,7.2,7.3"
            expanded_codes = []
            for code in application_codes:
                if ',' in code:
                    expanded_codes.extend(code.split(','))
                else:
                    expanded_codes.append(code)
            mask &= df['application_code'].isin(expanded_codes)
        
        # 4. Charge Unit Filter
        if charge_unit and not get_all_charge_units:
            mask &= df['charge_unit'] == charge_unit
        
        # 5. Market/Region Filter based on priority
        if region_priority == 'Ưu tiên 1':
            # P1: Filter by Billing Country
            if billing_country and 'billing_country' in df.columns:
                mask &= df['billing_country'] == billing_country
        elif region_priority == 'Ưu tiên 2':
            # P2: Filter by Regional Group
            if customer_regional_group and 'customer_regional_group' in df.columns:
                mask &= df['customer_regional_group'] == customer_regional_group
        # Ưu tiên 3: No filter (All markets)
        
        # 6. Special Shape Filter (for special products like L-cut, U-profile, etc.)
        # SKU format: XX##XXX#-####Y### where Y at position after dash indicates shape
        if special_shape and 'sku' in df.columns:
            # Filter products that have the special shape code in their SKU
            # The shape code appears after the dash in the dimension section (position ~13)
            # Examples: BX6.0DOC1-10040UL10/3 (U=U-profile), BD4.1DOX0-1000350L30 (L=L-cut)
            shape_mask = df['sku'].str.contains(f'-\\d{{4,}}.*{special_shape}', case=False, na=False, regex=True)
            mask &= shape_mask
        
        # Apply initial filters
        df_filtered = df[mask].copy()
        
        if len(df_filtered) == 0:
            return pd.DataFrame()
        
        # 6. Dimension Filter with tolerances
        tolerances = DIMENSION_PRIORITY_LEVELS.get(dimension_priority, {'height': 0, 'width': 0, 'length': 0})
        
        # Handle unlimited length for P3
        length_tolerance = 9999 if no_length_limit else tolerances['length']
        
        dim_mask = (
            (abs(df_filtered['height_cm'] - height_cm) <= tolerances['height']) &
            (abs(df_filtered['width_cm'] - width_cm) <= tolerances['width']) &
            (abs(df_filtered['length_cm'] - length_cm) <= length_tolerance)
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
        get_all_charge_units: bool = False,
        stone_priority: str = 'Ưu tiên 1',
        processing_priority: str = 'Ưu tiên 1',
        dimension_priority: str = 'Ưu tiên 1 - Đúng kích thước',
        region_priority: str = 'Ưu tiên 1',
        no_length_limit: bool = False,
        billing_country: str = None,
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
        
        # Note: Charge unit and Region filters are applied LATER
        # to allow better diagnostics (showing available alternatives)
        
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
            else:
                diagnostics['reason'] = "Không tìm thấy sản phẩm với các tiêu chí đã chọn"
            return diagnostics
        
        # 4. Check charge_unit availability (don't filter, just analyze)
        # Get all available charge units in the filtered data
        available_units = df_filtered['charge_unit'].dropna().unique().tolist()
        diagnostics['available_charge_units'] = available_units
        
        # Check if requested charge_unit has data (skip when get_all_charge_units is True)
        if charge_unit and not get_all_charge_units:
            unit_mask = df_filtered['charge_unit'] == charge_unit
            diagnostics['filter_counts']['after_charge_unit'] = unit_mask.sum()
            
            if unit_mask.sum() == 0 and len(available_units) > 0:
                # Requested charge_unit not available, but others are
                other_units = [u for u in available_units if u != charge_unit]
                if other_units:
                    diagnostics['reason'] = f"Không có dữ liệu đơn vị '{charge_unit}' cho các tiêu chí này"
                    diagnostics['suggestions'].append(f"Đổi sang đơn vị có sẵn: {', '.join(other_units)}")
                    # Count matches for each available unit
                    unit_counts = df_filtered['charge_unit'].value_counts().to_dict()
                    diagnostics['charge_unit_counts'] = unit_counts
                    return diagnostics
            
            # Apply charge unit filter for further analysis
            df_filtered = df_filtered[unit_mask].copy()
        else:
            diagnostics['filter_counts']['after_charge_unit'] = len(df_filtered)
        
        # 5. Region
        if 'customer_regional_group' in df_filtered.columns and region_priority == 'Ưu tiên 1' and customer_regional_group:
            region_mask = df_filtered['customer_regional_group'] == customer_regional_group
            if region_mask.sum() == 0:
                # Check available regions
                available_regions = df_filtered['customer_regional_group'].dropna().unique().tolist()
                diagnostics['reason'] = f"Không có dữ liệu vùng '{customer_regional_group}'"
                if available_regions:
                    diagnostics['suggestions'].append(f"Các vùng có sẵn: {', '.join(str(r) for r in available_regions[:5])}")
            df_filtered = df_filtered[region_mask].copy()
        diagnostics['filter_counts']['after_region'] = len(df_filtered)
        
        if len(df_filtered) == 0:
            if not diagnostics['reason']:
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
            # SMART CHECK: Before suggesting dimension changes, check if OTHER charge units
            # would have matching dimensions. This prioritizes qualitative params (left column)
            # over priority settings (right column).
            # Skip this check when get_all_charge_units is True (already searching all units)
            other_units_with_matches = []
            if charge_unit and not get_all_charge_units and 'available_charge_units' in diagnostics:
                for other_unit in diagnostics['available_charge_units']:
                    if other_unit == charge_unit:
                        continue
                    # Try finding matches with this other unit
                    test_matches = self.find_matching_products(
                        stone_color_type, processing_code, length_cm, width_cm, height_cm,
                        application_codes, customer_regional_group, other_unit,
                        stone_priority, processing_priority, dimension_priority, region_priority
                    )
                    if len(test_matches) > 0:
                        other_units_with_matches.append((other_unit, len(test_matches)))
            
            if other_units_with_matches:
                # Other charge units have matching products - prioritize this suggestion!
                units_info = ", ".join([f"{u} ({c} sp)" for u, c in other_units_with_matches])
                diagnostics['reason'] = f"Không có dữ liệu '{charge_unit}' cho kích thước này"
                diagnostics['suggestions'] = [f"Đổi sang đơn vị có sẵn: {units_info}"]
                diagnostics['other_units_with_matches'] = other_units_with_matches
            else:
                # No other charge units have matches - suggest dimension changes
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
    
    def calculate_confidence_score(
        self,
        matches: pd.DataFrame,
        query_length_cm: float,
        query_width_cm: float, 
        query_height_cm: float,
        query_stone_color: str = None,
        query_processing_code: str = None,
        query_application_codes: list = None,
        query_charge_unit: str = None,
        stone_priority: str = 'Ưu tiên 1',
        processing_priority: str = 'Ưu tiên 1',
        dimension_priority: str = 'Ưu tiên 1 - Đúng kích thước',
        region_priority: str = 'Ưu tiên 1',
    ) -> Dict[str, Any]:
        """
        Calculate multi-factor confidence score (0-100).
        
        Factors and weights:
        - Data Recency: 20%
        - Sample Count: 15%
        - Dimensional Deviation: 15%
        - Stone Color Match: 10%
        - Processing Match: 10%
        - Application Match: 10%
        - Charge Unit Match: 10%
        - Priority Strictness: 10%
        
        Returns:
            Dict with 'score' (0-100), 'level' (high/medium/low/very_low), 
            and 'breakdown' (individual factor scores)
        """
        if len(matches) == 0:
            return {
                'score': 0,
                'level': 'none',
                'breakdown': {}
            }
        
        breakdown = {}
        current_year = pd.Timestamp.now().year
        
        # 1. Sample Count Score (15%)
        n = len(matches)
        if n >= 10:
            sample_score = 100
        elif n >= 5:
            sample_score = 75
        elif n >= 2:
            sample_score = 50
        else:
            sample_score = 25
        breakdown['sample_count'] = {'score': sample_score, 'value': n}
        
        # 2. Data Recency Score (20%)
        if 'fy_year' in matches.columns:
            fy_years = pd.to_numeric(matches['fy_year'], errors='coerce').dropna()
            if len(fy_years) > 0:
                avg_year = fy_years.mean()
                years_old = current_year - avg_year
                if years_old <= 0.5:
                    recency_score = 100
                elif years_old <= 1:
                    recency_score = 85
                elif years_old <= 2:
                    recency_score = 65
                elif years_old <= 3:
                    recency_score = 40
                else:
                    recency_score = 20
            else:
                recency_score = 50  # Unknown recency
        else:
            recency_score = 50
        breakdown['recency'] = {'score': recency_score, 'value': f'{current_year - avg_year:.1f}yr' if 'avg_year' in dir() and avg_year else 'N/A'}
        
        # 3. Dimensional Deviation Score (15%)
        if query_length_cm and query_width_cm and query_height_cm:
            deviations = []
            for _, row in matches.iterrows():
                l_dev = abs(row.get('length_cm', query_length_cm) - query_length_cm) / query_length_cm * 100
                w_dev = abs(row.get('width_cm', query_width_cm) - query_width_cm) / query_width_cm * 100
                h_dev = abs(row.get('height_cm', query_height_cm) - query_height_cm) / query_height_cm * 100
                deviations.append((l_dev + w_dev + h_dev) / 3)
            avg_deviation = np.mean(deviations) if deviations else 0
            if avg_deviation <= 5:
                dim_score = 100
            elif avg_deviation <= 10:
                dim_score = 85
            elif avg_deviation <= 20:
                dim_score = 65
            elif avg_deviation <= 30:
                dim_score = 40
            else:
                dim_score = 20
        else:
            dim_score = 50
            avg_deviation = None
        breakdown['dimensional'] = {'score': dim_score, 'value': f'{avg_deviation:.1f}%' if avg_deviation is not None else 'N/A'}
        
        # 4. Stone Color Match Score (10%)
        if query_stone_color and 'stone_color_type' in matches.columns:
            exact_matches = (matches['stone_color_type'] == query_stone_color).sum()
            stone_score = (exact_matches / len(matches)) * 100
        else:
            stone_score = 100  # No filter = full score
        breakdown['stone_match'] = {'score': stone_score, 'value': f'{stone_score:.0f}%'}
        
        # 5. Processing Match Score (10%)
        if query_processing_code and 'processing_code' in matches.columns:
            exact_matches = (matches['processing_code'] == query_processing_code).sum()
            proc_score = (exact_matches / len(matches)) * 100
        else:
            proc_score = 100
        breakdown['processing_match'] = {'score': proc_score, 'value': f'{proc_score:.0f}%'}
        
        # 6. Application Match Score (10%)
        if query_application_codes and len(query_application_codes) > 0 and 'application_code' in matches.columns:
            app_matches = matches['application_code'].isin(query_application_codes).sum()
            app_score = (app_matches / len(matches)) * 100
        else:
            app_score = 100
        breakdown['application_match'] = {'score': app_score, 'value': f'{app_score:.0f}%'}
        
        # 7. Charge Unit Match Score (10%)
        if query_charge_unit and 'charge_unit' in matches.columns:
            unit_matches = (matches['charge_unit'] == query_charge_unit).sum()
            unit_score = (unit_matches / len(matches)) * 100
        else:
            unit_score = 100
        breakdown['charge_unit_match'] = {'score': unit_score, 'value': f'{unit_score:.0f}%'}
        
        # 8. Priority Strictness Score (10%)
        # Score based on how strict the priority levels are
        priority_scores = {
            'Ưu tiên 1': 100,
            'Ưu tiên 2': 65,
            'Ưu tiên 3': 30,
        }
        dim_priority_scores = {
            'Ưu tiên 1 - Đúng kích thước': 100,
            'Ưu tiên 2 - Sai lệch nhỏ': 65,
            'Ưu tiên 3 - Sai lệch lớn': 30,
        }
        stone_p = priority_scores.get(stone_priority, 65)
        proc_p = priority_scores.get(processing_priority, 65)
        dim_p = dim_priority_scores.get(dimension_priority, 65)
        region_p = priority_scores.get(region_priority, 65)
        priority_score = (stone_p + proc_p + dim_p + region_p) / 4
        breakdown['priority_strictness'] = {'score': priority_score, 'value': f'{priority_score:.0f}%'}
        
        # Calculate weighted total
        weighted_score = (
            sample_score * 0.15 +
            recency_score * 0.20 +
            dim_score * 0.15 +
            stone_score * 0.10 +
            proc_score * 0.10 +
            app_score * 0.10 +
            unit_score * 0.10 +
            priority_score * 0.10
        )
        
        # Map to confidence level (use >79 to ensure 80.0 rounds to 'high')
        if weighted_score > 79:
            level = 'high'
        elif weighted_score >= 60:
            level = 'medium'
        elif weighted_score >= 40:
            level = 'low'
        else:
            level = 'very_low'
        
        return {
            'score': round(weighted_score, 1),
            'level': level,
            'breakdown': breakdown
        }

    
    def estimate_price(self, matches: pd.DataFrame, use_recent_only: bool = True, recent_count: int = 10,
                        query_length_cm: float = None, query_width_cm: float = None, query_height_cm: float = None,
                        target_charge_unit: str = 'USD/M3', stone_color_type: str = None, processing_code: str = None,
                        special_shape: str = None, shape_params: Dict[str, float] = None,
                        application_codes: list = None,
                        stone_priority: str = 'Ưu tiên 1',
                        processing_priority: str = 'Ưu tiên 1',
                        dimension_priority: str = 'Ưu tiên 1 - Đúng kích thước',
                        region_priority: str = 'Ưu tiên 1') -> Dict[str, Any]:
        """
        Estimate price from matching products.
        Uses recency-weighted average, optionally filtering to most recent products.
        
        IMPORTANT: Normalizes all prices to USD/M3 before averaging to account for 
        different product sizes. Then converts back to target_charge_unit using 
        query dimensions. This ensures that larger products are priced proportionally 
        higher than smaller similar products.
        
        For special shapes (L, U, G, C, K, T, V), uses the shape-specific volume 
        calculation formulas instead of standard L×W×H.
        
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
            special_shape: Optional special shape code (L, U, G, C, K, T, B, V)
            shape_params: Optional dict of shape-specific parameters (wall_thickness_cm, etc.)
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
            
            # Get TLR: prefer Salesforce value (specific_gravity), fallback to calculated
            sf_tlr = row.get('specific_gravity')
            tlr = sf_tlr if sf_tlr and pd.notna(sf_tlr) else get_tlr(match_stone, match_proc)
            
            # Get HS: prefer Salesforce value (hs_coefficient), fallback to calculated
            sf_hs = row.get('hs_coefficient')
            hs = sf_hs if sf_hs and pd.notna(sf_hs) else get_hs_factor((match_length, match_width, match_height), match_proc)
            
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
            
            # Calculate volume for the query product
            # For special shapes, use shape-specific volume calculation
            if special_shape and special_shape in SPECIAL_SHAPE_INPUTS:
                params = shape_params or {}
                query_volume_m3 = calculate_special_shape_volume_m3(
                    shape_code=special_shape,
                    length_cm=query_length_cm,
                    width_cm=query_width_cm,
                    height_cm=query_height_cm,
                    **params
                )
            else:
                # Standard rectangular volume
                query_volume_m3 = calculate_volume_m3(query_length_cm, query_width_cm, query_height_cm)
            
            # Convert using the calculated volume (handles special shapes correctly)
            # For USD/M3 to target, we multiply by volume then convert units
            if target_charge_unit == 'USD/M3':
                estimated_price = weighted_price_m3
            elif target_charge_unit == 'USD/PC':
                # Price per piece = price/m3 × volume_m3
                estimated_price = weighted_price_m3 * query_volume_m3
            elif target_charge_unit == 'USD/M2':
                # Price per m2 = price/m3 × height_m
                query_height_m = query_height_cm / 100
                estimated_price = weighted_price_m3 * query_height_m
            elif target_charge_unit == 'USD/TON':
                # Price per ton = price/m3 × (1 / (TLR × HS))
                estimated_price = weighted_price_m3 / (query_tlr * query_hs)
            else:
                estimated_price = weighted_price_m3
            
            # Calculate min/max/median using same volume conversion
            if target_charge_unit == 'USD/M3':
                min_price = prices_m3.min()
                max_price = prices_m3.max()
                median_price = prices_m3.median()
            elif target_charge_unit == 'USD/PC':
                min_price = prices_m3.min() * query_volume_m3
                max_price = prices_m3.max() * query_volume_m3
                median_price = prices_m3.median() * query_volume_m3
            elif target_charge_unit == 'USD/M2':
                query_height_m = query_height_cm / 100
                min_price = prices_m3.min() * query_height_m
                max_price = prices_m3.max() * query_height_m
                median_price = prices_m3.median() * query_height_m
            elif target_charge_unit == 'USD/TON':
                min_price = prices_m3.min() / (query_tlr * query_hs)
                max_price = prices_m3.max() / (query_tlr * query_hs)
                median_price = prices_m3.median() / (query_tlr * query_hs)
            else:
                min_price = prices_m3.min()
                max_price = prices_m3.max()
                median_price = prices_m3.median()
        else:
            # Fallback: use original method (direct averaging) if no query dimensions
            prices = matches['sales_price']
            estimated_price = np.average(prices, weights=weights)
            min_price = prices.min()
            max_price = prices.max()
            median_price = prices.median()
        
        # Calculate multi-factor confidence score
        confidence_result = self.calculate_confidence_score(
            matches=matches,
            query_length_cm=query_length_cm,
            query_width_cm=query_width_cm,
            query_height_cm=query_height_cm,
            query_stone_color=stone_color_type,
            query_processing_code=processing_code,
            query_application_codes=application_codes,
            query_charge_unit=target_charge_unit,
            stone_priority=stone_priority,
            processing_priority=processing_priority,
            dimension_priority=dimension_priority,
            region_priority=region_priority,
        )
        confidence = confidence_result['level']
        confidence_score = confidence_result['score']
        confidence_breakdown = confidence_result['breakdown']
        
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
        
        # Calculate average fiscal year for price adjustment
        avg_fy_year = None
        if 'fy_year' in matches.columns:
            fy_years_numeric = pd.to_numeric(matches['fy_year'], errors='coerce')
            fy_years_valid = fy_years_numeric.dropna()
            if len(fy_years_valid) > 0:
                avg_fy_year = fy_years_valid.mean()
        
        return {
            'estimated_price': round(estimated_price, 2),
            'min_price': round(min_price, 2),
            'max_price': round(max_price, 2),
            'median_price': round(median_price, 2),
            'match_count': len(matches),
            'total_matches': total_matches,
            'confidence': confidence,
            'confidence_score': confidence_score,
            'confidence_breakdown': confidence_breakdown,
            'years_used': years_used,
            'price_m3': round(weighted_price_m3, 2),
            'estimated_price_m3': round(weighted_price_m3, 2),  # Alias for step-by-step display
            'query_length_cm': query_length_cm,  # For step-by-step display
            'query_width_cm': query_width_cm,    # For step-by-step display
            'query_height_cm': query_height_cm,  # For step-by-step display
            'price_trend': price_trend,
            'trend_pct': round(trend_pct, 1) if trend_pct is not None else None,
            'avg_fy_year': round(avg_fy_year, 1) if avg_fy_year is not None else None
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
        get_all_charge_units: bool = False,
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
                get_all_charge_units=get_all_charge_units,
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
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    # Auto-load data on first app launch
    if not st.session_state.data_loaded and SALESFORCE_AVAILABLE:
        with st.spinner("🔄 Đang tải dữ liệu từ Salesforce..."):
            try:
                loader = SalesforceDataLoader()
                df = loader.get_contract_products()
                if len(df) > 0:
                    st.session_state.data = df
                    predictor = SimilarityPricePredictor()
                    count = predictor.load_data(df)
                    st.session_state.model = predictor
                    st.session_state.model_metrics = {'loaded_samples': count}
                    st.session_state.data_loaded = True
            except Exception as e:
                st.error(f"❌ Lỗi tự động tải dữ liệu: {str(e)}")
    
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
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔮 Dự đoán giá", 
        "🧊 Tính giá nâng cao",
        "📊 Phân tích dữ liệu", 
        "🔍 Tìm sản phẩm tương tự",
        "📐 Bảng tra cứu",
        "📋 Dữ liệu chi tiết"
    ])

    
    # Tab 1: Price Prediction
    with tab1:
        st.subheader("🔮 Dự đoán giá sản phẩm (Similarity-Based)")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Thông tin sản phẩm")
            
            # 1. Màu đá (Stone Color) - FIRST per manager's notes
            stone_color = st.selectbox(
                "Màu đá (Stone Color)",
                options=[code for code, label in STONE_COLOR_TYPES],
                format_func=lambda x: STONE_COLOR_LOOKUP.get(x, x)
            )
            
            # 2. Kích thước (Dimensions) - SECOND
            st.markdown("##### Kích thước")
            col_dim1, col_dim2, col_dim3 = st.columns(3)
            with col_dim1:
                length = st.number_input("Dài (cm)", min_value=0.1, max_value=300.0, value=30.0, step=0.5)
            with col_dim2:
                width = st.number_input("Rộng (cm)", min_value=0.1, max_value=300.0, value=30.0, step=0.5)
            with col_dim3:
                height = st.number_input("Dày (cm)", min_value=0.5, max_value=50.0, value=3.0, step=0.5)
            
            # Processing lookup for use in col2
            processing_lookup = {code: (eng, vn) for code, eng, vn in PROCESSING_CODES}
            
            # 5. Ứng dụng (Application) - FIFTH
            application_lookup = {code: name for code, name in APPLICATION_CODES}
            selected_applications = st.multiselect(
                "Ứng dụng sản phẩm (Application)",
                options=[code for code, name in APPLICATION_CODES],
                format_func=lambda x: application_lookup.get(x, 'Unknown'),
                default=[],
                help="Chọn một hoặc nhiều ứng dụng. Để trống = Tất cả"
            )
            
            # 6. Đơn vị tính (Unit) - SIXTH with height recommendation
            charge_unit = st.selectbox(
                "Đơn vị tính giá",
                CHARGE_UNITS,
                help="💡 **Khuyến nghị theo chiều dày:**\n- Dày > 4cm → USD/m³\n- Dày < 4cm → USD/m²"
            )
            # Show dynamic recommendation based on height
            if height > 4:
                st.caption("💡 *Khuyến nghị: USD/M³ (chiều dày > 4cm)*")
            elif height > 0:
                st.caption("💡 *Khuyến nghị: USD/M2 (chiều dày < 4cm)*")
            
            # Get all charge units checkbox
            get_all_charge_units = st.checkbox(
                "Get all charge units",
                value=False,
                help="Khi chọn, tìm tất cả sản phẩm không lọc theo đơn vị tính giá. "
                     "Giá dự đoán và tính lại giá sẽ được quy đổi về đơn vị đã chọn ở trên."
            )
            
            # 7. Phân loại khách hàng (Customer Classification) - SEVENTH
            customer_type = st.selectbox(
                "Phân loại khách hàng",
                ['C', 'A', 'B', 'D', 'E', 'F'],
                format_func=lambda x: f"{x} - {CUSTOMER_PRICING_RULES[x]['description']}"
            )
        
        with col2:
            # 8. Mức độ ưu tiên (Priority Levels) - EIGHTH
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
                    index=1  # Default: Ưu tiên 2 (Cùng chủng loại)
                )
                # Show stone priority info
                if stone_priority == 'Ưu tiên 2':
                    stone_family = STONE_FAMILY_MAP.get(stone_color, 'OTHER')
                    st.caption(f"🪨 Tìm tất cả đá {stone_family}")
                elif stone_priority == 'Ưu tiên 3':
                    st.caption("🪨 Tìm tất cả loại đá")
                
                processing_priority = st.selectbox(
                    "Gia công chính (Main Processing)",
                    options=['Ưu tiên 1', 'Ưu tiên 2', 'Ưu tiên 3'],
                    format_func=lambda x: {
                        'Ưu tiên 1': '1 - Đúng loại gia công',
                        'Ưu tiên 2': '2 - Đúng nhóm gia công',
                        'Ưu tiên 3': '3 - Tất cả gia công',
                    }[x],
                    index=2  # Default: Ưu tiên 3 (Tất cả gia công)
                )
                
                # Show processing code dropdown when Priority 1 is selected
                processing_code = None
                selected_processing_group = None
                if processing_priority == 'Ưu tiên 1':
                    processing_code = st.selectbox(
                        "Chọn loại gia công",
                        options=[code for code, eng, vn in PROCESSING_CODES],
                        format_func=lambda x: f"{x} - {processing_lookup.get(x, ('Other', 'Khác'))[0]} ({processing_lookup.get(x, ('Other', 'Khác'))[1]})",
                        index=0,
                        help="Lọc theo loại gia công cụ thể"
                    )
                elif processing_priority == 'Ưu tiên 2':
                    # Get default group
                    default_group = 'GIA_CONG_MAY_TAY'
                    group_options = list(PROCESSING_GROUP_NAMES.keys())
                    default_index = group_options.index(default_group) if default_group in group_options else 0
                    
                    selected_processing_group = st.selectbox(
                        "Chọn nhóm gia công",
                        options=group_options,
                        format_func=lambda x: f"{PROCESSING_GROUP_NAMES.get(x, x)} ({', '.join(PROCESSING_GROUPS.get(x, []))})",
                        index=default_index,
                        help="Lọc theo nhóm gia công thay vì loại gia công cụ thể"
                    )
                    group_codes = PROCESSING_GROUPS.get(selected_processing_group, [])
                    st.caption(f"⚙️ Tìm nhóm: {', '.join(group_codes)}")
                else:
                    st.caption("⚙️ Tìm tất cả loại gia công")
            with col_p2:
                dimension_priority = st.selectbox(
                    "Kích thước",
                    options=list(DIMENSION_PRIORITY_LEVELS.keys()),
                    index=0  # Default: Ưu tiên 1 (Đúng kích thước)
                )
                # Show tolerance info when not using exact match
                if dimension_priority != 'Ưu tiên 1 - Đúng kích thước':
                    tol = DIMENSION_PRIORITY_LEVELS[dimension_priority]
                    st.caption(f"📏 Cho phép sai lệch: Cao ±{tol['height']}cm, Rộng ±{tol['width']}cm, Dài ±{tol['length']}cm")
                
                # Show "unlimited length" checkbox when Priority 3 is selected
                no_length_limit = False
                if 'Ưu tiên 3' in dimension_priority:
                    no_length_limit = st.checkbox(
                        "Không giới hạn chiều dài",
                        value=False,
                        help="Bỏ giới hạn chiều dài khi tìm kiếm sản phẩm tương tự"
                    )
                
                region_priority = st.selectbox(
                    "Nhóm Khu vực KH (Regional Group)",
                    options=['Ưu tiên 1', 'Ưu tiên 2', 'Ưu tiên 3'],
                    format_func=lambda x: {
                        'Ưu tiên 1': '1 - Đúng nước (Billing)',
                        'Ưu tiên 2': '2 - Đúng nhóm KH',
                        'Ưu tiên 3': '3 - Tất cả thị trường',
                    }[x],
                    index=2  # Default: Ưu tiên 3 
                )

                # Dynamic selector based on region_priority
                billing_country_selected = None
                customer_regional_group = None
                if region_priority == 'Ưu tiên 1':
                    # Get unique billing countries from data
                    billing_countries = []
                    if st.session_state.data is not None and 'billing_country' in st.session_state.data.columns:
                        unique_countries = st.session_state.data['billing_country'].dropna().unique().tolist()
                        billing_countries = sorted([c for c in unique_countries if c])
                    
                    billing_country_selected = st.selectbox(
                        "Chọn nước (Billing Country)",
                        options=billing_countries,
                        help="Lọc theo quốc gia trong địa chỉ thanh toán"
                    )
                elif region_priority == 'Ưu tiên 2':
                    customer_regional_group = st.selectbox(
                        "Chọn Nhóm Khu vực KH",
                        options=[code for code, name in CUSTOMER_REGIONAL_GROUPS if code],
                        format_func=lambda x: x,
                        index=0,
                        help="Nhóm đầu 0-9 theo khu vực khách hàng"
                    )
                    st.caption(f"🌍 Tìm theo nhóm: {customer_regional_group}")
                else:
                    st.caption("🌍 Tìm tất cả thị trường")
            
            regional_group_selected = customer_regional_group  # Use the selected regional group
            
            st.divider()
            st.markdown("#### 📅 Cài đặt tính toán giá")
            use_recent_only = st.checkbox(
                "Chỉ sử dụng dữ liệu gần nhất",
                value=True,
                help="Chỉ sử dụng N sản phẩm gần nhất (theo năm tài chính) để ước tính giá chính xác hơn. Nên đặt từ 5 đến 10 sản phẩm tham khảo!"
            )
            recent_count = st.number_input(
                "Số lượng sản phẩm tham khảo",
                min_value=5,
                max_value=35,
                value=5,
                step=5,
                help="Số lượng sản phẩm gần nhất sử dụng để ước tính giá. Nên đặt từ 5 đến 10 sản phẩm tham khảo!",
                disabled=not use_recent_only
            )
            
            # Yearly price adjustment per manager's notes
            st.markdown("##### 📈 Điều chỉnh giá theo năm")
            apply_yearly_adjustment = st.checkbox(
                "Áp dụng điều chỉnh giá theo năm",
                value=True,
                help="Tỷ lệ tăng giá hàng năm do chi phí nguyên vật liệu và nhân công (thường 3-5%) hoặc điều chỉnh theo lạm phát. Xem thông tin lạm phát theo quốc gia [tại đây](https://www.tradingview.com/markets/world-economy/charts-global-trends/)."
            )
            yearly_increase_pct = st.slider(
                "Tỷ lệ tăng giá hàng năm (%)",
                min_value=0.0,
                max_value=10.0,
                value=0.5,
                step=0.5,
                format="%.1f%%",
                disabled=not apply_yearly_adjustment
            )
            
            predict_btn = st.button("🔍 Tìm kiếm & Ước tính giá", type="primary", use_container_width=True)
        
        # ============ FULL WIDTH RESULTS SECTION ============
        if predict_btn and st.session_state.model is not None:
            st.divider()
            
            # Use similarity-based predictor
            predictor = st.session_state.model
            
            matches = predictor.find_matching_products(
                stone_color_type=stone_color,
                processing_code=processing_code,
                length_cm=length,
                width_cm=width,
                height_cm=height,
                application_codes=selected_applications,
                customer_regional_group=regional_group_selected,
                charge_unit=charge_unit,
                get_all_charge_units=get_all_charge_units,
                stone_priority=stone_priority,
                processing_priority=processing_priority,
                dimension_priority=dimension_priority,
                region_priority=region_priority,
                no_length_limit=no_length_limit,
                billing_country=billing_country_selected,
                selected_processing_group=selected_processing_group,
            )
            
            # Store matches in session state to persist across reruns
            st.session_state.last_matches = matches.copy()
            
            # Clear manual estimation when new search is performed
            if 'manual_estimation' in st.session_state:
                st.session_state.manual_estimation = None
            
            estimation = predictor.estimate_price(
                matches, 
                use_recent_only=use_recent_only, 
                recent_count=recent_count,
                query_length_cm=length,
                query_width_cm=width,
                query_height_cm=height,
                target_charge_unit=charge_unit,
                stone_color_type=stone_color,
                processing_code=processing_code,
                application_codes=selected_applications,
                stone_priority=stone_priority,
                processing_priority=processing_priority,
                dimension_priority=dimension_priority,
                region_priority=region_priority,
            )
            
            # Store estimation and query params in session state to persist across reruns
            st.session_state.last_estimation = estimation.copy()
            st.session_state.last_query_params = {
                'stone_color': stone_color,
                'length': length,
                'width': width,
                'height': height,
                'processing_code': processing_code,
                'regional_group': customer_regional_group,
                'applications': selected_applications,
                'charge_unit': charge_unit,
                'customer_type': customer_type,
                'use_recent_only': use_recent_only,
                'recent_count': recent_count,
                'apply_yearly_adjustment': apply_yearly_adjustment,
                'yearly_increase_pct': yearly_increase_pct,
            }
            
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
                    'high': 'Cao',
                    'medium': 'Trung bình',
                    'low': 'Thấp',
                    'very_low': 'Rất thấp',
                }
                conf_color = confidence_colors.get(estimation['confidence'], '#808080')
                # Use dark text for medium (yellow) background for better readability
                text_color = '#000000' if estimation['confidence'] == 'medium' else 'white'
                conf_score = estimation.get('confidence_score', 0)
                conf_label = f"{confidence_labels.get(estimation['confidence'], 'N/A')} ({conf_score:.0f}%)"
                
                # Note: Main results are now displayed in the combined customer price card below
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
                
                # === Display estimation result using DRY helper ===
                yearly_adj_info, final_price, final_min, final_max = display_estimation_result(
                    estimation=estimation,
                    price_info=price_info,
                    charge_unit=charge_unit,
                    customer_type=customer_type,
                    conf_color=conf_color,
                    text_color=text_color,
                    conf_label=conf_label,
                    apply_yearly_adjustment=apply_yearly_adjustment,
                    yearly_increase_pct=yearly_increase_pct,
                )
                st.divider()
                st.markdown("#### 📄 Xuất báo cáo")
                
                # Prepare query params for report
                query_params = {
                    'stone_color': stone_color,
                    'length': length,
                    'width': width,
                    'height': height,
                    'processing_code': processing_code,
                    'regional_group': customer_regional_group,
                    'applications': selected_applications,
                    'charge_unit': charge_unit,
                    'customer_type': customer_type,
                }
                
                # yearly_adj_info is now returned by display_estimation_result()
                
                # Filter matches for report if use_recent_only is selected
                matches_for_report = matches.copy()
                if use_recent_only and len(matches_for_report) > recent_count:
                    if 'created_date' in matches_for_report.columns:
                        # Ensure date conversion
                        matches_for_report['created_date'] = pd.to_datetime(matches_for_report['created_date'], errors='coerce')
                        matches_for_report = matches_for_report.sort_values('created_date', ascending=False).head(recent_count)

                # CRITICAL FIX: Recalculate customer price based on yearly-adjusted price for report
                # Step 5 (yearly adjustment) must flow into Step 6 (customer adjustment)
                if yearly_adj_info and yearly_adj_info.get('applied'):
                    adjusted_price = yearly_adj_info['adjusted_price']
                    price_info_for_report = calculate_customer_price(
                        adjusted_price, customer_type, 
                        segment=segment, charge_unit=charge_unit
                    )
                else:
                    price_info_for_report = price_info

                # Generate HTML report
                report_html = generate_price_report(
                    query_params=query_params,
                    estimation=estimation,
                    matched_products=matches_for_report,
                    customer_price_info=price_info_for_report,  # Use adjusted price_info
                    yearly_adjustment=yearly_adj_info,
                    priority_settings={
                        'stone_priority': stone_priority,
                        'processing_priority': processing_priority,
                        'dimension_priority': dimension_priority,
                        'region_priority': region_priority,
                    }
                )
                
                st.download_button(
                    label="📥 Tải báo cáo (HTML/PDF)",
                    data=report_html,
                    file_name=f"stone_price_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                    mime="text/html",
                    help="Tải báo cáo HTML. Mở và in (Ctrl+P) để lưu PDF."
                )
                    
            else:
                # Get detailed diagnostics for why no matches found
                diagnostics = predictor.get_match_diagnostics(
                    stone_color_type=stone_color,
                    processing_code=processing_code,
                    length_cm=length,
                    width_cm=width,
                    height_cm=height,
                    application_codes=selected_applications,
                    customer_regional_group=regional_group_selected,
                    charge_unit=charge_unit,
                    get_all_charge_units=get_all_charge_units,
                    stone_priority=stone_priority,
                    processing_priority=processing_priority,
                    dimension_priority=dimension_priority,
                    region_priority=region_priority,
                    no_length_limit=no_length_limit,
                    billing_country=billing_country_selected,
                )
                
                st.warning(f"⚠️ Không tìm thấy sản phẩm phù hợp")
                
                if diagnostics.get('reason'):
                    st.error(f"**Lý do:** {diagnostics['reason']}")
                
                if diagnostics.get('suggestions'):
                    st.info("**💡 Gợi ý:**\n" + "\n".join([f"• {s}" for s in diagnostics['suggestions']]))
            
            # Product info summary with weight calculation (always show after search)
            st.divider()
            st.markdown("**📦 Thông tin sản phẩm:**")
            volume_m3 = calculate_volume_m3(length, width, height)
            area_m2 = calculate_area_m2(length, width)
            tlr = get_tlr(stone_color, processing_code)
            first_app = selected_applications[0] if selected_applications else ''
            hs = get_hs_factor((length, width, height), processing_code, first_app)
            weight_tons = calculate_weight_tons(volume_m3, stone_color, processing_code, (length, width, height), first_app)
            
            # Calculate price segment based on estimation
            estimated_price_m3 = estimation.get('estimated_price_m3', 0) if estimation else 0
            if estimated_price_m3 == 0 and estimation and estimation.get('estimated_price'):
                # Fallback calculation if price_m3 not directly available
                if volume_m3 > 0:
                    estimated_price_m3 = estimation['estimated_price'] / volume_m3 if charge_unit == 'USD/M3' else estimation['estimated_price'] / volume_m3
            
            product_segment = classify_segment(estimated_price_m3, height, stone_color, processing_code)
            segment_color = get_segment_color(product_segment)
            segment_bilingual = {
                'Super premium': 'Super Premium / Siêu cao cấp',
                'Premium': 'Premium / Cao cấp', 
                'Common': 'Common / Phổ thông',
                'Economy': 'Economy / Kinh tế'
            }.get(product_segment, product_segment)
            
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.markdown(f"- Kích thước: {length} x {width} x {height} cm")
                st.markdown(f"- Thể tích: {volume_m3:.6f} m³")
                st.markdown(f"- Diện tích: {area_m2:.4f} m²")
                # Price Segment with color badge - bilingual
                st.markdown(f"- Phân khúc giá: <span style='background-color:{segment_color}; color:white; padding:2px 8px; border-radius:4px; font-weight:bold'>{segment_bilingual}</span>", unsafe_allow_html=True)
            with col_info2:
                st.markdown(f"- TLR: {tlr} tấn/m³")
                st.markdown(f"- HS: {hs}")
                st.markdown(f"- Khối lượng: **{weight_tons:.4f} tấn**")
                # Show customer adjustment info for segment
                if customer_type and estimation:
                    customer_info = calculate_customer_price(estimation['estimated_price'], customer_type, product_segment)
                    if customer_info and customer_info.get('adjustment_label'):
                        st.markdown(f"- Điều chỉnh KH {customer_type}: {customer_info['adjustment_label']}")
        
        # ============ SHOW PERSISTED ESTIMATION RESULTS (when page reruns e.g. checkbox click) ============
        # Show estimation results from session state when predict_btn is not pressed but we have cached results
        if not predict_btn and 'last_estimation' in st.session_state and st.session_state.last_estimation is not None:
            estimation = st.session_state.last_estimation
            query_params = st.session_state.get('last_query_params', {})
            
            # Only show if we have a valid estimation
            if estimation.get('estimated_price') is not None:
                st.divider()
                st.markdown("#### 📊 Kết quả ước tính")
                
                # Confidence indicator
                confidence_colors = {
                    'high': '#6bcb77',
                    'medium': '#ffd93d',
                    'low': '#ff6b6b',
                    'very_low': '#9e7cc1',
                }
                confidence_labels = {
                    'high': 'Cao',
                    'medium': 'Trung bình',
                    'low': 'Thấp',
                    'very_low': 'Rất thấp',
                }
                conf_color = confidence_colors.get(estimation.get('confidence', ''), '#808080')
                # Use dark text for medium (yellow) background for better readability
                text_color = '#000000' if estimation.get('confidence', '') == 'medium' else 'white'
                conf_score = estimation.get('confidence_score', 0)
                conf_label = f"{confidence_labels.get(estimation.get('confidence', ''), 'N/A')} ({conf_score:.0f}%)"
                
                cached_charge_unit = query_params.get('charge_unit', charge_unit)
                cached_height = query_params.get('height', height)
                cached_length = query_params.get('length', length)
                cached_width = query_params.get('width', width)
                cached_stone_color = query_params.get('stone_color', stone_color)
                cached_processing_code = query_params.get('processing_code', processing_code)
                
                # Calculate segment and customer price
                first_app = selected_applications[0] if selected_applications else ''
                est_price_m3 = convert_price(
                    estimation['estimated_price'], cached_charge_unit, 'USD/M3',
                    height_cm=cached_height, length_cm=cached_length, width_cm=cached_width,
                    tlr=get_tlr(cached_stone_color, cached_processing_code)
                )
                segment = classify_segment(est_price_m3, height_cm=cached_height, family=first_app, processing_code=cached_processing_code)
                
                price_info = calculate_customer_price(
                    estimation['estimated_price'], customer_type, 
                    segment=segment, charge_unit=cached_charge_unit
                )
                
                # === Display estimation result using DRY helper ===
                display_estimation_result(
                    estimation=estimation,
                    price_info=price_info,
                    charge_unit=cached_charge_unit,
                    customer_type=customer_type,
                    conf_color=conf_color,
                    text_color=text_color,
                    conf_label=conf_label,
                    apply_yearly_adjustment=apply_yearly_adjustment,
                    yearly_increase_pct=yearly_increase_pct,
                )
        
        # ============ MATCHING PRODUCTS (Full Width) ============
        # Show matching products if we have stored matches from session state
        # This allows the table to persist when checkboxes are clicked (avoiding reload reset)
        if 'last_matches' in st.session_state and st.session_state.last_matches is not None and len(st.session_state.last_matches) > 0:
            matches = st.session_state.last_matches
            st.divider()
            st.markdown("#### 📋 Sản phẩm trong hệ thống khớp tiêu chí")
            
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
            # Calculate price_m2 if not present - using correct conversion logic
            if 'price_m2' not in matches.columns:
                def calc_price_m2(row):
                    unit = row.get('charge_unit', '')
                    sales_p = row.get('sales_price', 0) or 0
                    h_cm = row.get('height_cm', 0) or 0
                    l_cm = row.get('length_cm', 0) or 0
                    w_cm = row.get('width_cm', 0) or 0
                    tlr = row.get('specific_gravity', 2.7) or 2.7
                    hs = row.get('hs_coefficient', 1.0) or 1.0
                    
                    if unit == 'USD/M2':
                        # Already price per m² - no calculation needed!
                        return sales_p
                    elif unit == 'USD/M3':
                        # Price/m² = Price/m³ × height(m)
                        return sales_p * (h_cm / 100) if h_cm > 0 else 0
                    elif unit == 'USD/PC':
                        # Price/m² = Price per piece ÷ area(m²)
                        area_m2 = (l_cm * w_cm) / 10000 if l_cm > 0 and w_cm > 0 else 0.01
                        return sales_p / area_m2 if area_m2 > 0 else 0
                    elif unit == 'USD/TON':
                        # Price/m² = Price/ton × TLR × HS × height(m)
                        return sales_p * tlr * hs * (h_cm / 100) if h_cm > 0 else 0
                    else:
                        return sales_p
                matches['price_m2'] = matches.apply(calc_price_m2, axis=1)
            
            # Reorder columns per user request
            display_cols = [
                # Primary: Contract name, dimensions
                'contract_product_name',
                'contract_name',
                'length_cm', 'width_cm', 'height_cm',
                # Price info
                'sales_price', 'charge_unit', 'price_m2', 'price_m3',
                # Date
                'created_date',
                # Product identification
                'processing_name', 'application', 'segment', 'sku', 
                # Secondary columns
                'stone_color_type',
                'application_code'
                'processing_code', 
                'account_code', 'customer_regional_group', 'billing_country',
                'specific_gravity', 'hs_coefficient', 'fy_year',
            ]
            available_cols = [col for col in display_cols if col in matches.columns]
            
            # Column config for headers
            col_config = {
                'select': st.column_config.CheckboxColumn('Chọn', default=False, help='Chọn sản phẩm để tính giá'),
                'sku': st.column_config.TextColumn('SKU'),
                'stone_color_type': st.column_config.TextColumn('Stone Color'),
                'application_code': st.column_config.TextColumn('App Code'),
                'application': st.column_config.TextColumn('Application'),
                'processing_code': st.column_config.TextColumn('Main Processing Code'),
                'processing_name': st.column_config.TextColumn('Main Processing'),
                'customer_regional_group': st.column_config.TextColumn('Regional Group'),
                'billing_country': st.column_config.TextColumn('Billing Country'),
                'sales_price': st.column_config.NumberColumn('Sales Price', format="$%.2f"),
                'price_m2': st.column_config.NumberColumn('Price/m²', format="$%.2f"),
                'price_m3': st.column_config.NumberColumn('Price/m³', format="$%.2f"),
                'specific_gravity': st.column_config.NumberColumn('TLR', format="%.2f", help="Specific Gravity / Tỷ lệ khối lượng"),
                'hs_coefficient': st.column_config.NumberColumn('HS', format="%.2f", help="Bottom Cladding Coefficient / Hệ số hao hụt"),
            }
            
            with st.expander(f"📋 Chọn sản phẩm để tính giá ({len(matches)} sản phẩm khớp)", expanded=True):
                st.info("💡 **Chọn ít nhất 3 sản phẩm** để tính giá chính xác hơn. Bấm 'Tính lại giá' sau khi chọn.")
                
                # Add checkbox column for selection
                matches_display = matches[available_cols].copy()
                matches_display.insert(0, 'select', False)  # Add selection column at start
                
                # Use data_editor for editable checkboxes
                edited_df = st.data_editor(
                    matches_display, 
                    use_container_width=True, 
                    height=350, 
                    column_config=col_config,
                    hide_index=True,
                    key="product_selection_table"
                )
                
                # Calculate price from selected records
                selected_rows = edited_df[edited_df['select'] == True]
                selected_count = len(selected_rows)
                
                col_select_info, col_recalc = st.columns([2, 1])
                with col_select_info:
                    if selected_count == 0:
                        st.warning("⚠️ Chưa chọn sản phẩm nào")
                    elif selected_count < 3:
                        st.warning(f"⚠️ Đã chọn {selected_count}/3 sản phẩm (cần tối thiểu 3)")
                    else:
                        st.success(f"✅ Đã chọn {selected_count} sản phẩm")
                
                with col_recalc:
                    recalc_btn = st.button("🔄 Tính lại giá từ sản phẩm đã chọn", disabled=(selected_count < 3))
                    # Show yearly adjustment reminder
                    if apply_yearly_adjustment:
                        st.caption(f"📈 Tỷ lệ tăng giá hàng năm: **{yearly_increase_pct:.1f}%**")
                
                # Recalculate price from selected records with volume normalization
                if recalc_btn and selected_count >= 3:
                    # Calculate average FY year from selected products for yearly adjustment
                    avg_fy_year = None
                    if 'fy_year' in selected_rows.columns:
                        fy_years = pd.to_numeric(selected_rows['fy_year'], errors='coerce').dropna()
                        if len(fy_years) > 0:
                            avg_fy_year = int(fy_years.mean())
                    
                    # === Volume-normalized pricing (same logic as estimate_price) ===
                    # Convert all selected product prices to USD/M3 first
                    prices_m3 = []
                    for idx, row in selected_rows.iterrows():
                        price = row['sales_price']
                        unit = row.get('charge_unit', 'USD/M3')
                        match_length = row.get('length_cm', 10)
                        match_width = row.get('width_cm', 10)
                        match_height = row.get('height_cm', 3)
                        match_stone = row.get('stone_color_type', stone_color or 'ABSOLUTE BASALT')
                        match_proc = row.get('processing_code', processing_code)
                        
                        # Get TLR: prefer Salesforce value (specific_gravity), fallback to calculated
                        sf_tlr = row.get('specific_gravity')
                        tlr = sf_tlr if sf_tlr and pd.notna(sf_tlr) else get_tlr(match_stone, match_proc)
                        
                        # Get HS: prefer Salesforce value (hs_coefficient), fallback to calculated
                        sf_hs = row.get('hs_coefficient')
                        hs = sf_hs if sf_hs and pd.notna(sf_hs) else get_hs_factor((match_length, match_width, match_height), match_proc)
                        
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
                    
                    prices_m3 = pd.Series(prices_m3, index=selected_rows.index)
                    
                    # Calculate weighted average in USD/M3
                    avg_price_m3 = prices_m3.mean()
                    min_price_m3 = prices_m3.min()
                    max_price_m3 = prices_m3.max()
                    median_price_m3 = prices_m3.median()
                    
                    # Convert from USD/M3 to target unit using QUERY dimensions
                    query_tlr = get_tlr(stone_color or 'ABSOLUTE BASALT', processing_code)
                    query_hs = get_hs_factor((length, width, height), processing_code)
                    
                    estimated_price = convert_price(
                        avg_price_m3, 'USD/M3', charge_unit,
                        height_cm=height,
                        length_cm=length,
                        width_cm=width,
                        tlr=query_tlr,
                        hs=query_hs
                    )
                    min_price = convert_price(
                        min_price_m3, 'USD/M3', charge_unit,
                        height_cm=height, length_cm=length, width_cm=width,
                        tlr=query_tlr, hs=query_hs
                    )
                    max_price = convert_price(
                        max_price_m3, 'USD/M3', charge_unit,
                        height_cm=height, length_cm=length, width_cm=width,
                        tlr=query_tlr, hs=query_hs
                    )
                    median_price = convert_price(
                        median_price_m3, 'USD/M3', charge_unit,
                        height_cm=height, length_cm=length, width_cm=width,
                        tlr=query_tlr, hs=query_hs
                    )
                    
                    # Calculate confidence for manual selection
                    # Need to create predictor since it's not in this scope
                    recalc_predictor = SimilarityPricePredictor()
                    manual_conf = recalc_predictor.calculate_confidence_score(
                        matches=selected_rows,
                        query_length_cm=length,
                        query_width_cm=width,
                        query_height_cm=height,
                        query_stone_color=stone_color,
                        query_processing_code=processing_code,
                        query_application_codes=selected_applications,
                        query_charge_unit=charge_unit,
                        stone_priority=stone_priority,
                        processing_priority=processing_priority,
                        dimension_priority=dimension_priority,
                        region_priority=region_priority,
                    )
                    
                    # Store manual estimation in session state
                    st.session_state.manual_estimation = {
                        'estimated_price': estimated_price,
                        'min_price': min_price,
                        'max_price': max_price,
                        'median_price': median_price,
                        'match_count': selected_count,
                        'avg_fy_year': avg_fy_year,
                        'total_matches': len(matches),
                        'price_m3': avg_price_m3,
                        'confidence': manual_conf['level'],
                        'confidence_score': manual_conf['score'],
                        'confidence_breakdown': manual_conf['breakdown'],
                    }
                
                # Show results if we have a valid manual estimation from this session (even if button wasn't just clicked)
                # We check if match_count matches to ensure it corresponds to the current selection approximately
                if 'manual_estimation' in st.session_state and st.session_state.manual_estimation is not None:
                    manual_estimation = st.session_state.manual_estimation
                    
                    st.divider()
                    
                    # Confidence for manual selection (use multi-factor score if available)
                    manual_count = manual_estimation['match_count']
                    conf_score = manual_estimation.get('confidence_score', 0)
                    conf_level = manual_estimation.get('confidence', 'low')
                    confidence_colors = {
                        'high': '#6bcb77',
                        'medium': '#ffd93d',
                        'low': '#ff6b6b',
                        'very_low': '#9e7cc1',
                    }
                    confidence_labels = {
                        'high': 'Cao',
                        'medium': 'Trung bình',
                        'low': 'Thấp',
                        'very_low': 'Rất thấp',
                    }
                    conf_color = confidence_colors.get(conf_level, '#808080')
                    # Use dark text for medium (yellow) background for better readability
                    text_color = '#000000' if conf_level == 'medium' else 'white'
                    conf_label = f"{confidence_labels.get(conf_level, 'N/A')} ({conf_score:.0f}%)"
                    
                    # Calculate segment and customer price adjustment
                    first_app = selected_applications[0] if selected_applications else ''
                    est_price_m3 = convert_price(
                        manual_estimation['estimated_price'], charge_unit, 'USD/M3',
                        height_cm=height, length_cm=length, width_cm=width,
                        tlr=get_tlr(stone_color, processing_code)
                    )
                    segment = classify_segment(est_price_m3, height_cm=height, family=first_app, processing_code=processing_code)
                    
                    price_info = calculate_customer_price(
                        manual_estimation['estimated_price'], customer_type, 
                        segment=segment, charge_unit=charge_unit
                    )
                    
                    # === Display estimation result using DRY helper ===
                    yearly_adj_info, final_price, final_min, final_max = display_estimation_result(
                        estimation=manual_estimation,
                        price_info=price_info,
                        charge_unit=charge_unit,
                        customer_type=customer_type,
                        conf_color=conf_color,
                        text_color=text_color,
                        conf_label=conf_label,
                        apply_yearly_adjustment=apply_yearly_adjustment,
                        yearly_increase_pct=yearly_increase_pct,
                        is_manual=True,
                        manual_count=manual_count,
                    )
                    
                    # Show product info and segment (like main estimation)
                    st.markdown("##### 🧱 Thông tin sản phẩm:")
                    segment_color = get_segment_color(segment)
                    segment_bilingual = {
                        'Super premium': 'Super Premium / Siêu cao cấp',
                        'Premium': 'Premium / Cao cấp', 
                        'Common': 'Common / Phổ thông',
                        'Economy': 'Economy / Kinh tế'
                    }.get(segment, segment)
                    
                    volume_m3 = calculate_volume_m3(length, width, height)
                    area_m2 = calculate_area_m2(length, width)
                    
                    col_info1, col_info2 = st.columns(2)
                    with col_info1:
                        st.markdown(f"- Kích thước: {length} x {width} x {height} cm")
                        st.markdown(f"- Thể tích: {volume_m3:.6f} m³")
                        st.markdown(f"- Diện tích: {area_m2:.4f} m²")
                        st.markdown(f"- Phân khúc giá: <span style='background-color:{segment_color}; color:white; padding:2px 8px; border-radius:4px; font-weight:bold'>{segment_bilingual}</span>", unsafe_allow_html=True)
                    with col_info2:
                        tlr = get_tlr(stone_color, processing_code)
                        hs = get_hs_factor((length, width, height), processing_code)
                        weight_tons = volume_m3 * tlr
                        st.markdown(f"- TLR: {tlr} tấn/m³")
                        st.markdown(f"- HS: {hs}")
                        st.markdown(f"- Khối lượng: **{weight_tons:.4f} tấn**")
                        st.markdown(f"- Điều chỉnh KH {customer_type}: {price_info['adjustment_label']}")
                    
                    # Export Report Button
                    st.divider()
                    st.markdown("#### 📄 Xuất báo cáo")
                    
                    # Prepare query params for report
                    query_params = {
                        'stone_color': stone_color,
                        'length': length,
                        'width': width,
                        'height': height,
                        'processing_code': processing_code,
                        'regional_group': customer_regional_group,
                        'applications': selected_applications,
                        'charge_unit': charge_unit,
                        'customer_type': customer_type,
                    }
                    
                    # yearly_adj_info is now returned by display_estimation_result()
                    
                    # Generate HTML report (use selected products only)
                    selected_matches = matches[matches.index.isin(selected_rows.index)]
                    report_html = generate_price_report(
                        query_params=query_params,
                        estimation=manual_estimation,
                        matched_products=selected_matches,
                        customer_price_info=price_info,
                        yearly_adjustment=yearly_adj_info,
                        priority_settings={
                            'stone_priority': stone_priority,
                            'processing_priority': processing_priority,
                            'dimension_priority': dimension_priority,
                            'region_priority': region_priority,
                        }
                    )
                    
                    st.download_button(
                        label="📥 Tải báo cáo (HTML/PDF)",
                        data=report_html,
                        file_name=f"stone_price_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                        mime="text/html",
                        help="Tải báo cáo HTML. Mở và in (Ctrl+P) để lưu PDF."
                    )
                    
                    # Product info summary
                    st.divider()
                    st.markdown("**📦 Thông tin sản phẩm:**")
                    volume_m3 = calculate_volume_m3(length, width, height)
                    area_m2 = calculate_area_m2(length, width)
                    tlr = get_tlr(stone_color, processing_code)
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
        
        elif predict_btn and st.session_state.model is None:
            st.warning("⚠️ Vui lòng chuẩn bị tìm kiếm trước (nút 🔍 ở sidebar)")
        
        # ============ REFERENCE MATERIALS (Full Width, Always at End) ============
        st.divider()
        st.markdown("#### 📖 Tài liệu tham khảo")
        
        with st.expander("🔧 Nhóm gia công (Priority 2)"):
            st.markdown("""
**Nhóm gia công theo cách xử lý:**

| Nhóm | Mã gia công | Mô tả |
|------|-------------|-------|
| **Gia công Tay** | CTA, TLO, TDE | Chẻ tay, Tự nhiên lồi, Tước đẽo |
| **Gia công Máy + Tay** | CUA, CLO, QME, GCT | Cưa, Cưa lột, Quay mẻ, Giả cổ tay |
| **Gia công Máy** | DOT, DOC, DOX, GCR, MGI, PCA, BAM | Đốt, Đốt chải, Đốt xịt, Giả cổ rung, Mài giấy, Phun cát, Băm |
| **Gia công Máy Cao cấp** | HON, BON, CHA | Hone, Bóng, Chải |

*Khi chọn Ưu tiên 2 cho Gia công, hệ thống sẽ tìm các sản phẩm cùng nhóm gia công.*
            """)
        
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
**Nguyên tắc áp dụng bảng giá A-B-C-D-E-F** (9/11/2023)

| Loại | Mô tả | Thời gian GD | Sản lượng/năm | Chiến lược | Điều chỉnh giá | Quyền tự quyết |
|:----:|-------|:------------:|:-------------:|------------|----------------|----------------|
| **A** | Thân thiết đặc biệt | >10 năm | 50-150 cont | Cạnh tranh đối thủ | **-1.5% → -3%** so với B | Thảo luận chiến lược |
| **B** | Lớn, chuyên nghiệp | 3-10 năm | 20-50 cont | Đồng hành chiến lược | **-10→30 USD/m³** (2-4%) | Thảo luận chiến lược |
| **C** | Phổ thông | 1-5 năm | 5-20 cont | Phát triển, dịch vụ tốt | **Giá chuẩn** | ±10-20 USD/m³ theo phân khúc |
| **D** | Mới, khu vực cao | 1 năm | 1-10 cont | Tin cậy, phục vụ nhanh | **+15→45 USD/m³** (3-6%) | ±30-40 USD/m³ theo phân khúc |
| **E** | Sản phẩm mới/sáng tạo | 1 năm | 1-10 cont | Năng lực cao, đổi mới | **×1.08→1.15** | ±30-40 USD/m³ |
| **F** | Dự án cao cấp | 1-5 năm | 1-50 cont | Kinh nghiệm công trình | **×1.08→1.15** | ±30-40 USD/m³ |

**Quyền tự quyết theo phân khúc sản phẩm (cho KH loại C):**
- 🟢 **Economy (Kinh tế):** ±10.0 USD/m³
- 🔵 **Common (Phổ thông):** ±15.0 USD/m³  
- 🟣 **Premium (Cao cấp):** ±20.0 USD/m³ hoặc ±0.5 USD/m²

**Quyền tự quyết cho KH loại D, E, F:**
- 🟣 **Premium:** ±30.0 USD/m³ hoặc ±1.0 USD/m²
- 🔴 **Super Premium:** ±40.0 USD/m³ hoặc ±1.5 USD/m²
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
        
        with st.expander("🎯 Tiêu chí tìm kiếm"):
            st.markdown("""
| Tiêu chí | Ưu tiên 1 | Ưu tiên 2 | Ưu tiên 3 |
|----------|-----------|-----------|-----------| 
| **Loại đá** | Đúng màu đá | Cùng chủng loại | Tất cả loại đá |
| **Gia công** | Đúng loại gia công | Đúng nhóm gia công | Tất cả gia công |
| **Cao (cm)** | ±0 | ±1 | ±5 |
| **Rộng (cm)** | ±0 | ±5 | ±20 |
| **Dài (cm)** | ±0 | ±10 | ±30 (hoặc không giới hạn) |
| **Thị trường** | Đúng nước (Billing) | Đúng nhóm KH | Tất cả thị trường |
            """)
        
        with st.expander("📦 Quy tắc ứng dụng sản phẩm"):
            st.markdown("""
| ỨNG DỤNG | Code | Name (English) | Name (Vietnamese) |
|----------|------|----------------|-------------------|
| CUBE | 1.1 | Cubes / Cobbles | Cubic (Đá vuông) |
| PAVING | 1.3 | Paving stone | Đá lát ngoài trời |
| WALL_STONE | 2.1 | Wall stone | Đá xây tường rào |
| PALISADE | 3.1 | Palisades | Đá cây |
| KERB | 3.2 | Border / Kerbs | Đá bó vỉa hè |
| STEP | 4.1, 4.2 | Stair / Step | Đá bậc thang |
| POOL | 6.1 | Pool surrounding | Đá ghép hồ bơi |
| TILE | 7.1-7.3 | Tile / Paver | Đá lát quy cách |
| SLAB | 9.1 | Slab | Đá slab khổ lớn |
            """)
        
        with st.expander("🏷️ Quy định mã SKU sản phẩm"):
            st.markdown("""
**Cấu trúc mã SKU:**

| Vị trí | Định dạng | Mô tả |
|--------|-----------|-------|
| 1-2 | 2 chữ cái | **Nguyên vật liệu** (Mã loại đá) |
| 3-4 | 2 số | **Mục đích sử dụng** |
| 5-7 | 3 chữ cái | **Gia công bề mặt chính** |
| 8 | 1 số | **Gia công phụ** |
| 9-12 | 4 số (mm) | **Chiều dài** |
| 13 | 1 số/chữ | **Chiều rộng** |
| 14-16 | 3 số (mm) | **Chiều cao** |

---

**Nguyên vật liệu (Vị trí 1-2):**

| Mã | Tiếng Việt | English |
|----|-----------|---------|
| BD | Đá Bazan Đen | Basalt Black |
| BX | Đá Bazan Xám | Basalt Grey |
| BT | Đá Bazan Tổ ong | Basalt Hive |
| GX | Đá Granite Xám | Granite Grey |
| GT | Đá Granite Trắng | Granite White |
| GV | Đá Granite Vàng | Granite Yellow |
| GD | Đá Granite Đỏ | Granite Red |
| GH | Đá Granite Hồng | Granite Pink |
| MB | Marble Bluestone | Marble Blue |
| MT | Marble Trắng | Marble White |
| MV | Marble Vàng | Marble Yellow |

---

**Mục đích sử dụng (Vị trí 3-4):**

| Mã | Mô tả |
|----|-------|
| 01 | Đá lát nền ngoại thất (Cubic, tấm) |
| 02 | Tường rào (Đá khối, NTR) |
| 03 | Đá cây |
| 04 | Đá bậc thang (Nguyên khối, ốp BT) |
| 05 | Đá mỹ nghệ |
| 06 | Cao cấp hồ bơi, bộ cửa |
| 07 | Lát nền bên trong, đá bộ |
| 08 | Ốp tường |
| 09 | Slab, bàn bếp, cao cấp |

---

**Gia công bề mặt chính (Vị trí 5-7):**

| Mã | Gia công | Mã | Gia công |
|----|----------|----|----------|
| CTA | Chẻ tay tự nhiên | HON | Mặt hon |
| CUA | Mặt cưa | BON | Mặt bóng |
| CLO | Cưa lột tay | BAM | Mặt băm |
| TDE | Tẩy đẹp | GCR | Giả cổ rung |
| DOT | Mặt đốt | GCT | Giả cổ tay |
| DOC | Đốt chải | MGI | Mài giấy |
| DOX | Đốt xịt | PCA | Phun cát |
| TLO | Tách lồi | QME | Quay mẻ |

---

**Gia công phụ (Vị trí 8):**

| Mã | Mô tả |
|----|-------|
| 0 | Không có gia công phụ |
| 1 | Cạnh cưa |
| 2 | Cạnh chẻ tay tự nhiên |
| 3 | Cạnh hone |
| 4 | Cạnh đốt |
| 5 | Cạnh băm |
| 6 | Cạnh bo tròn R |
| 7 | Đáy băm |
| 8 | Gõ mẻ |
| 9 | Gia công khác (Có chú thích) |

---

**Ví dụ:** `BD01DOT2-06004060`
- **BD:** Bazan Đen
- **01:** Đá lát nền
- **DOT:** Mặt đốt
- **2:** Cạnh chẻ tay
- **0600:** 600mm dài
- **4:** 400mm rộng
- **060:** 60mm cao

→ *Bazan Đen Lát nền, mặt Đốt, cạnh Chẻ tay, KT 600×400×60mm*
            """)
        
        with st.expander("🔗 Nhóm loại đá (Stone Family)"):
            st.markdown("""
**Dùng cho Ưu tiên 2 - Cùng chủng loại:**

| Nhóm | Mã loại đá |
|------|------------|
| **BASALT** | BD (Black), BX (Grey), BT (Hive) |
| **GRANITE** | GX, GT, GV, GD, GH |
| **MARBLE** | MB, MT, MV |
            """)
    
    # Tab 2: Advanced Price Prediction with 3D Model
    with tab2:
        from tabs.advanced_calculator import render_advanced_calculator
        render_advanced_calculator()
    
    # Tab 3: Data Analysis (was Tab 2)
    with tab3:
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
    
    # Tab 4: Similar Products (was Tab 3)
    with tab4:
        st.subheader("🔍 Tìm sản phẩm tương tự")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### Tiêu chí tìm kiếm")
            search_family = st.selectbox("Loại sản phẩm", [''] + PRODUCT_FAMILIES, key='search_family')
            search_stone = st.selectbox(
                "Màu đá",
                options=[''] + [code for code, label in STONE_COLOR_TYPES],
                format_func=lambda x: STONE_COLOR_LOOKUP.get(x, 'Tất cả') if x else 'Tất cả',
                key='search_stone'
            )
            
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
                search_length = st.number_input("Dài (cm)", min_value=0.0, value=30.0, step=0.5, key='search_l')
            with search_col2:
                search_width = st.number_input("Rộng (cm)", min_value=0.0, value=30.0, step=0.5, key='search_w')
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
                                'billing_country',
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
                    'billing_country': st.column_config.TextColumn('Billing Country'),
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
    
    # Tab 5: Weight & Conversion Reference (was Tab 4)
    with tab5:
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
    
    # Tab 6: Detailed Data (was Tab 5)
    with tab6:
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
            'billing_country',         # Billing Country from Account.BillingAddress
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
            'billing_country': st.column_config.TextColumn('Billing Country', help='Billing country from Account.BillingAddress'),
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
