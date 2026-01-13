"""
Test cases for Stone Price Predictor
Run with: python -m pytest tests/test_predictor.py -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import pytest
from app import (
    SimilarityPricePredictor,
    convert_price,
    get_tlr,
    get_hs_factor,
    calculate_volume_m3,
    calculate_area_m2,
    calculate_weight_tons,
    classify_segment,
    DIMENSION_PRIORITY_LEVELS,
    STONE_FAMILY_MAP,
)


# ============ Sample Test Data ============
@pytest.fixture
def sample_data():
    """Create sample product data for testing."""
    return pd.DataFrame({
        'contract_product_name': [
            'Absolute Basalt Palisade 50258',
            'Absolute Basalt Palisade 75258',
            'Absolute Basalt Palisade 100258',
            'Black Basalt Cube 10108',
            'Black Basalt Tile 40203',
        ],
        'stone_color_type': ['ABSOLUTE BASALT', 'ABSOLUTE BASALT', 'ABSOLUTE BASALT', 'BLACK BASALT', 'BLACK BASALT'],
        'processing_code': ['DOX', 'DOX', 'DOX', 'CTA', 'DOT'],
        'application_code': ['3.1', '3.1', '3.1', '1.1', '1.3'],
        'length_cm': [50.0, 75.0, 100.0, 10.0, 40.0],
        'width_cm': [25.0, 25.0, 25.0, 10.0, 20.0],
        'height_cm': [8.0, 8.0, 8.0, 8.0, 3.0],
        'sales_price': [8.75, 13.82, 19.54, 1.50, 4.20],
        'charge_unit': ['USD/PC', 'USD/PC', 'USD/PC', 'USD/PC', 'USD/PC'],
        'customer_regional_group': ['Nhóm đầu 3', 'Nhóm đầu 3', 'Nhóm đầu 3', 'Nhóm đầu 1', 'Nhóm đầu 2'],
        'fy_year': [2025, 2025, 2024, 2024, 2025],
        'created_date': pd.to_datetime(['2025-06-01', '2025-05-01', '2024-11-01', '2024-08-01', '2025-03-01']),
    })


@pytest.fixture  
def predictor(sample_data):
    """Create predictor instance with sample data."""
    pred = SimilarityPricePredictor()
    pred.load_data(sample_data)
    return pred


# ============ Test Unit Conversion ============
class TestUnitConversion:
    """Test price unit conversion functions."""
    
    def test_convert_pc_to_m3(self):
        """USD/PC to USD/M3: larger pieces = lower price per m3."""
        # 50x25x8 cm piece at $8.75/PC
        price_m3 = convert_price(8.75, 'USD/PC', 'USD/M3', 
                                  length_cm=50, width_cm=25, height_cm=8)
        expected_vol = (50 * 25 * 8) / 1_000_000  # 0.01 m3
        expected_price_m3 = 8.75 / expected_vol  # 875 USD/M3
        assert abs(price_m3 - expected_price_m3) < 0.01
    
    def test_convert_m3_to_pc(self):
        """USD/M3 to USD/PC: converts using volume."""
        price_pc = convert_price(875.0, 'USD/M3', 'USD/PC',
                                 length_cm=50, width_cm=25, height_cm=8)
        expected = 875.0 * (50 * 25 * 8 / 1_000_000)  # 8.75
        assert abs(price_pc - expected) < 0.01
    
    def test_convert_m2_to_m3(self):
        """USD/M2 to USD/M3: divide by height."""
        price_m3 = convert_price(100.0, 'USD/M2', 'USD/M3', height_cm=3)
        expected = 100.0 / 0.03  # 3333.33
        assert abs(price_m3 - expected) < 1.0
    
    def test_convert_ton_to_m3(self):
        """USD/TON to USD/M3: multiply by TLR."""
        price_m3 = convert_price(300.0, 'USD/TON', 'USD/M3', tlr=2.95)
        expected = 300.0 * 2.95  # 885
        assert abs(price_m3 - expected) < 0.01


# ============ Test TLR and HS Factor ============
class TestMaterialProperties:
    """Test TLR and HS factor calculation."""
    
    def test_tlr_absolute_basalt(self):
        """Absolute Basalt should have TLR 2.95."""
        tlr = get_tlr('ABSOLUTE BASALT', 'DOX')
        assert tlr == 2.95
    
    def test_tlr_black_basalt(self):
        """Black Basalt should have TLR 2.65-2.70."""
        tlr = get_tlr('BLACK BASALT', 'CTA')
        assert 2.60 <= tlr <= 2.75
    
    def test_hs_cubic_small(self):
        """Small cubic 5x5x5 should have HS 1.0."""
        hs = get_hs_factor((5, 5, 5), 'CTA')
        assert 0.9 <= hs <= 1.1
    
    def test_hs_cubic_medium(self):
        """Medium cubic 10x10x8 should have lower HS due to chipping."""
        hs = get_hs_factor((10, 10, 8), 'CTA')
        assert hs <= 1.0


# ============ Test Volume and Weight Calculation ============
class TestVolumeWeight:
    """Test volume and weight calculations."""
    
    def test_volume_m3(self):
        """Volume calculation: 50x25x8 cm = 0.01 m3."""
        vol = calculate_volume_m3(50, 25, 8)
        assert abs(vol - 0.01) < 0.0001
    
    def test_area_m2(self):
        """Area calculation: 50x25 cm = 0.125 m2."""
        area = calculate_area_m2(50, 25)
        assert abs(area - 0.125) < 0.0001
    
    def test_weight_tons(self):
        """Weight calculation: volume * TLR * HS."""
        vol = 0.01  # m3
        weight = calculate_weight_tons(vol, 'ABSOLUTE BASALT', 'DOX', (50, 25, 8))
        # 0.01 * 2.95 * 1.0 = 0.0295 tons
        assert 0.02 <= weight <= 0.035


# ============ Test Matching Logic ============
class TestMatchingLogic:
    """Test product matching with priority filters."""
    
    def test_find_exact_match(self, predictor):
        """Priority 1 should find exact matches only."""
        matches = predictor.find_matching_products(
            stone_color_type='ABSOLUTE BASALT',
            processing_code='DOX',
            length_cm=50.0,
            width_cm=25.0,
            height_cm=8.0,
            application_codes=['3.1'],
            customer_regional_group='Nhóm đầu 3',
            charge_unit='USD/PC',
            stone_priority='Ưu tiên 1',
            processing_priority='Ưu tiên 1',
            dimension_priority='Ưu tiên 1 - Đúng kích thước',
            region_priority='Ưu tiên 1',
        )
        assert len(matches) == 1
        assert matches.iloc[0]['length_cm'] == 50.0
    
    def test_find_dimension_tolerance(self, predictor):
        """Priority 2/3 should find products within tolerance."""
        matches = predictor.find_matching_products(
            stone_color_type='ABSOLUTE BASALT',
            processing_code='DOX',
            length_cm=52.0,  # 2cm off (within ±5 for priority 2)
            width_cm=25.0,
            height_cm=8.0,
            application_codes=['3.1'],
            customer_regional_group='',
            charge_unit='USD/PC',
            dimension_priority='Ưu tiên 2 - Sai lệch nhỏ',
            region_priority='Ưu tiên 2',
        )
        # Should NOT find 50cm product due to tolerance
        # Let's check tolerance: ±10 for length in priority 2
        assert len(matches) >= 1  # Should find 50cm product (52-50=2, within ±10)
    
    def test_stone_family_matching(self, predictor):
        """Priority 2 should match same stone family."""
        matches = predictor.find_matching_products(
            stone_color_type='ABSOLUTE BASALT',
            processing_code='',
            length_cm=10.0,
            width_cm=10.0,
            height_cm=8.0,
            application_codes=[],
            customer_regional_group='',
            charge_unit='USD/PC',
            stone_priority='Ưu tiên 2',  # Same family (BASALT)
            processing_priority='Ưu tiên 2',
            dimension_priority='Ưu tiên 1 - Đúng kích thước',
            region_priority='Ưu tiên 2',
        )
        # Should find BLACK BASALT 10x10x8 cube
        assert len(matches) >= 1
    
    def test_no_match_returns_empty(self, predictor):
        """Should return empty DataFrame when no matches."""
        matches = predictor.find_matching_products(
            stone_color_type='GRANITE',  # Not in test data
            processing_code='DOX',
            length_cm=50.0,
            width_cm=25.0,
            height_cm=8.0,
            application_codes=[],
            customer_regional_group='',
            charge_unit='USD/PC',
        )
        assert len(matches) == 0


# ============ Test Price Estimation with Volume Normalization ============
class TestPriceEstimation:
    """Test price estimation with volume-based normalization."""
    
    def test_larger_product_higher_price(self, predictor):
        """Larger products should have proportionally higher prices."""
        # Find matches for a 50cm palisade
        matches_50 = predictor.find_matching_products(
            stone_color_type='ABSOLUTE BASALT',
            processing_code='DOX',
            length_cm=50.0,
            width_cm=25.0,
            height_cm=8.0,
            application_codes=['3.1'],
            customer_regional_group='',
            charge_unit='USD/PC',
            dimension_priority='Ưu tiên 3 - Sai lệch lớn',
            region_priority='Ưu tiên 2',
        )
        
        # Estimate price for 50cm
        est_50 = predictor.estimate_price(
            matches_50,
            query_length_cm=50.0,
            query_width_cm=25.0,
            query_height_cm=8.0,
            target_charge_unit='USD/PC',
            stone_color_type='ABSOLUTE BASALT',
            processing_code='DOX'
        )
        
        # Estimate price for 100cm (larger)
        est_100 = predictor.estimate_price(
            matches_50,  # Same matches
            query_length_cm=100.0,  # но larger query
            query_width_cm=25.0,
            query_height_cm=8.0,
            target_charge_unit='USD/PC',
            stone_color_type='ABSOLUTE BASALT',
            processing_code='DOX'
        )
        
        # 100cm should be roughly 2x the price of 50cm (double length = double volume)
        if est_50['estimated_price'] and est_100['estimated_price']:
            ratio = est_100['estimated_price'] / est_50['estimated_price']
            assert 1.8 <= ratio <= 2.2, f"Expected ~2x, got {ratio:.2f}x"
    
    def test_recency_filtering(self, predictor):
        """Recent products should be filtered correctly."""
        matches = predictor.find_matching_products(
            stone_color_type='ABSOLUTE BASALT',
            processing_code='DOX',
            length_cm=50.0,
            width_cm=25.0,
            height_cm=8.0,
            application_codes=['3.1'],
            customer_regional_group='',
            charge_unit='USD/PC',
            dimension_priority='Ưu tiên 3 - Sai lệch lớn',
            region_priority='Ưu tiên 2',
        )
        
        est = predictor.estimate_price(
            matches,
            use_recent_only=True,
            recent_count=2,  # Only top 2 recent
            query_length_cm=50.0,
            query_width_cm=25.0,
            query_height_cm=8.0,
            target_charge_unit='USD/PC',
        )
        
        # Should use only 2 matches
        if est['total_matches'] > 2:
            assert est['match_count'] == 2


# ============ Test Diagnostics ============
class TestDiagnostics:
    """Test match diagnostics for no-match scenarios."""
    
    def test_diagnostics_dimension_issue(self, predictor):
        """Diagnostics should identify dimension as the blocking filter."""
        diag = predictor.get_match_diagnostics(
            stone_color_type='ABSOLUTE BASALT',
            processing_code='DOX',
            length_cm=50.0,
            width_cm=25.0,
            height_cm=20.0,  # Much larger than available (8cm)
            application_codes=['3.1'],
            customer_regional_group='',
            charge_unit='USD/PC',
            dimension_priority='Ưu tiên 1 - Đúng kích thước',  # Exact match only
            region_priority='Ưu tiên 2',
        )
        
        assert 'Cao' in diag['reason'] or 'kích thước' in diag['reason'].lower()
        assert diag['closest_height'] == 8.0
    
    def test_diagnostics_stone_type_issue(self, predictor):
        """Diagnostics should identify stone type as blocking."""
        diag = predictor.get_match_diagnostics(
            stone_color_type='GRANITE RED',  # Not in data
            processing_code='DOX',
            length_cm=50.0,
            width_cm=25.0,
            height_cm=8.0,
            application_codes=['3.1'],
            customer_regional_group='',
            charge_unit='USD/PC',
        )
        
        assert 'loại đá' in diag['reason'].lower() or 'GRANITE' in diag['reason']


# ============ Test Dimension Tolerance Constants ============
class TestDimensionTolerances:
    """Verify dimension tolerance settings are correct."""
    
    def test_priority_1_exact(self):
        """Priority 1 should require exact match."""
        tol = DIMENSION_PRIORITY_LEVELS['Ưu tiên 1 - Đúng kích thước']
        assert tol['height'] == 0
        assert tol['width'] == 0
        assert tol['length'] == 0
    
    def test_priority_2_small_tolerance(self):
        """Priority 2 should have small tolerances."""
        tol = DIMENSION_PRIORITY_LEVELS['Ưu tiên 2 - Sai lệch nhỏ']
        assert tol['height'] == 1
        assert tol['width'] == 5
        assert tol['length'] == 10
    
    def test_priority_3_large_tolerance(self):
        """Priority 3 should have large tolerances."""
        tol = DIMENSION_PRIORITY_LEVELS['Ưu tiên 3 - Sai lệch lớn']
        assert tol['height'] == 5
        assert tol['width'] == 15
        assert tol['length'] == 30


# ============ Test Segment Classification ============
class TestSegmentClassification:
    """Test product segment classification."""
    
    def test_premium_segment_high_price(self):
        """High price products should be Premium."""
        segment = classify_segment(1000.0, height_cm=8, family='Palisades')
        assert segment == 'Premium'
    
    def test_common_segment_low_price(self):
        """Low price products should be Common."""
        segment = classify_segment(300.0, height_cm=8, family='Cubic')
        assert segment in ['Common', 'Economy']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
