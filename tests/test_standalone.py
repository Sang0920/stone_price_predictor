"""
Standalone test script for Stone Price Predictor
Run with: python tests/test_standalone.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from app import (
    SimilarityPricePredictor,
    convert_price,
    get_tlr,
    get_hs_factor,
    calculate_volume_m3,
    calculate_area_m2,
    calculate_weight_tons,
    DIMENSION_PRIORITY_LEVELS,
)


def create_sample_data():
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
        'billing_country': ['Germany', 'Germany', 'United States', 'Belgium', 'France'],
        'fy_year': [2025, 2025, 2024, 2024, 2025],
        'created_date': pd.to_datetime(['2025-06-01', '2025-05-01', '2024-11-01', '2024-08-01', '2025-03-01']),
    })


def test_result(name, passed, details=""):
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status}: {name}")
    if not passed and details:
        print(f"         → {details}")
    return passed


def run_tests():
    print("\n" + "="*60)
    print("Stone Price Predictor - Test Suite")
    print("="*60 + "\n")
    
    passed = 0
    failed = 0
    
    # Create sample data
    sample_data = create_sample_data()
    predictor = SimilarityPricePredictor()
    predictor.load_data(sample_data)
    
    # ============ Test 1: Unit Conversion ============
    print("📐 Testing Unit Conversion...")
    
    # Test PC to M3
    price_m3 = convert_price(8.75, 'USD/PC', 'USD/M3', length_cm=50, width_cm=25, height_cm=8)
    expected = 8.75 / 0.01  # 875
    if test_result("PC to M3 conversion", abs(price_m3 - expected) < 1, f"Got {price_m3:.2f}, expected ~{expected:.2f}"):
        passed += 1
    else:
        failed += 1
    
    # Test M3 to PC
    price_pc = convert_price(875.0, 'USD/M3', 'USD/PC', length_cm=50, width_cm=25, height_cm=8)
    if test_result("M3 to PC conversion", abs(price_pc - 8.75) < 0.1, f"Got {price_pc:.2f}, expected 8.75"):
        passed += 1
    else:
        failed += 1
    
    # ============ Test 2: TLR Values ============
    print("\n⚖️ Testing TLR Values...")
    
    tlr_abs = get_tlr('ABSOLUTE BASALT', 'DOX')
    if test_result("Absolute Basalt TLR = 2.95", tlr_abs == 2.95, f"Got {tlr_abs}"):
        passed += 1
    else:
        failed += 1
    
    tlr_black = get_tlr('BLACK BASALT', 'CTA')
    if test_result("Black Basalt TLR between 2.60-2.75", 2.60 <= tlr_black <= 2.75, f"Got {tlr_black}"):
        passed += 1
    else:
        failed += 1
    
    # ============ Test 3: Volume Calculation ============
    print("\n📦 Testing Volume Calculation...")
    
    vol = calculate_volume_m3(50, 25, 8)
    if test_result("Volume 50x25x8 = 0.01 m³", abs(vol - 0.01) < 0.0001, f"Got {vol}"):
        passed += 1
    else:
        failed += 1
    
    area = calculate_area_m2(50, 25)
    if test_result("Area 50x25 = 0.125 m²", abs(area - 0.125) < 0.0001, f"Got {area}"):
        passed += 1
    else:
        failed += 1
    
    # ============ Test 4: Matching Logic ============
    print("\n🔍 Testing Matching Logic...")
    
    # Exact match
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
    if test_result("Exact match finds 1 product", len(matches) == 1, f"Found {len(matches)} matches"):
        passed += 1
    else:
        failed += 1
    
    # No match for non-existent stone
    matches_none = predictor.find_matching_products(
        stone_color_type='GRANITE RED',
        processing_code='DOX',
        length_cm=50.0,
        width_cm=25.0,
        height_cm=8.0,
        application_codes=[],
        customer_regional_group='',
        charge_unit='USD/PC',
    )
    if test_result("Non-existent stone returns empty", len(matches_none) == 0, f"Found {len(matches_none)} matches"):
        passed += 1
    else:
        failed += 1
    
    # Dimension tolerance test
    matches_tol = predictor.find_matching_products(
        stone_color_type='ABSOLUTE BASALT',
        processing_code='DOX',
        length_cm=55.0,  # 5cm off from 50cm
        width_cm=25.0,
        height_cm=8.0,
        application_codes=['3.1'],
        customer_regional_group='',
        charge_unit='USD/PC',
        dimension_priority='Ưu tiên 2 - Sai lệch nhỏ',  # ±10 for length
        region_priority='Ưu tiên 2',
    )
    if test_result("Dimension tolerance (±10 length) finds match", len(matches_tol) >= 1, f"Found {len(matches_tol)} matches"):
        passed += 1
    else:
        failed += 1
    
    # ============ Test 5: Price Estimation ============
    print("\n💰 Testing Price Estimation with Volume Normalization...")
    
    # Get matches for estimation
    all_matches = predictor.find_matching_products(
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
    
    # Estimate for 50cm product
    est_50 = predictor.estimate_price(
        all_matches,
        query_length_cm=50.0,
        query_width_cm=25.0,
        query_height_cm=8.0,
        target_charge_unit='USD/PC',
        stone_color_type='ABSOLUTE BASALT',
        processing_code='DOX'
    )
    
    # Estimate for 100cm product (should be ~2x price)
    est_100 = predictor.estimate_price(
        all_matches,
        query_length_cm=100.0,
        query_width_cm=25.0,
        query_height_cm=8.0,
        target_charge_unit='USD/PC',
        stone_color_type='ABSOLUTE BASALT',
        processing_code='DOX'
    )
    
    if est_50['estimated_price'] and est_100['estimated_price']:
        ratio = est_100['estimated_price'] / est_50['estimated_price']
        if test_result("100cm price ~2x of 50cm price", 1.8 <= ratio <= 2.2, f"Ratio: {ratio:.2f}x"):
            passed += 1
        else:
            failed += 1
    else:
        test_result("Price estimation returns values", False, "Got None")
        failed += 1
    
    # ============ Test 6: Dimension Tolerances ============
    print("\n📏 Testing Dimension Tolerance Settings...")
    
    tol_1 = DIMENSION_PRIORITY_LEVELS['Ưu tiên 1 - Đúng kích thước']
    if test_result("Priority 1: exact match (0,0,0)", 
                   tol_1['height'] == 0 and tol_1['width'] == 0 and tol_1['length'] == 0):
        passed += 1
    else:
        failed += 1
    
    tol_2 = DIMENSION_PRIORITY_LEVELS['Ưu tiên 2 - Sai lệch nhỏ']
    if test_result("Priority 2: small tolerance (1,5,10)", 
                   tol_2['height'] == 1 and tol_2['width'] == 5 and tol_2['length'] == 10):
        passed += 1
    else:
        failed += 1
    
    tol_3 = DIMENSION_PRIORITY_LEVELS['Ưu tiên 3 - Sai lệch lớn']
    if test_result("Priority 3: large tolerance (5,20,30)", 
                   tol_3['height'] == 5 and tol_3['width'] == 20 and tol_3['length'] == 30):
        passed += 1
    else:
        failed += 1
    
    # ============ Test 7: Diagnostics ============
    print("\n🔬 Testing Match Diagnostics...")
    
    diag = predictor.get_match_diagnostics(
        stone_color_type='ABSOLUTE BASALT',
        processing_code='DOX',
        length_cm=50.0,
        width_cm=25.0,
        height_cm=20.0,  # 12cm off from available 8cm
        application_codes=['3.1'],
        customer_regional_group='',
        charge_unit='USD/PC',
        dimension_priority='Ưu tiên 1 - Đúng kích thước',
        region_priority='Ưu tiên 2',
    )
    
    if test_result("Diagnostics identifies dimension issue", 
                   'Cao' in diag.get('reason', '') or 'kích thước' in diag.get('reason', '').lower(),
                   f"Reason: {diag.get('reason', 'None')[:50]}..."):
        passed += 1
    else:
        failed += 1
    
    if test_result("Diagnostics shows closest height is 8cm", 
                   diag.get('closest_height') == 8.0,
                   f"Got closest_height: {diag.get('closest_height')}"):
        passed += 1
    else:
        failed += 1
    
    # ============ Summary ============
    print("\n" + "="*60)
    total = passed + failed
    print(f"Test Results: {passed}/{total} passed ({100*passed/total:.0f}%)")
    if failed == 0:
        print("🎉 All tests passed! Ready for deployment.")
    else:
        print(f"⚠️ {failed} test(s) failed. Please review before deploying.")
    print("="*60 + "\n")
    
    return failed == 0


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
