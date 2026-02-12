"""
Test Suite for Priority Selection States.
Tests all possible combinations of priority levels for Stone, Processing, Dimension, and Region.

Uses real data from docs/stone_price_data.csv to verify correct filtering logic.
"""

import os
import sys
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import (
    SimilarityPricePredictor,
    PROCESSING_GROUPS,
    PROCESSING_CODE_TO_GROUP,
    PROCESSING_GROUP_NAMES,
    STONE_FAMILY_MAP,
    DIMENSION_PRIORITY_LEVELS,
)

# Default test parameters - common product specs
DEFAULT_PARAMS = {
    'stone_color_type': 'BD',
    'processing_code': 'DOX',
    'length_cm': 100.0,
    'width_cm': 35.0,
    'height_cm': 15.0,
    'application_codes': [],  # Empty = no filter
    'customer_regional_group': 'Nhóm đầu 2',
    'charge_unit': 'USD/PC',
    'stone_priority': 'Ưu tiên 3',  # All stones
    'processing_priority': 'Ưu tiên 3',  # All processing
    'dimension_priority': 'Ưu tiên 3 - Sai lệch lớn',  # Large tolerance
    'region_priority': 'Ưu tiên 3',  # All regions
}


def load_test_data():
    """Load the exported CSV data for testing."""
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                            'docs', 'stone_price_data.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df)} records from stone_price_data.csv")
        return df
    else:
        print(f"❌ CSV file not found at {csv_path}")
        return None


def make_params(**overrides):
    """Create full parameter dict with overrides."""
    params = DEFAULT_PARAMS.copy()
    params.update(overrides)
    return params


def test_stone_priority_1(predictor, df):
    """Test Stone Priority 1: Exact stone color match."""
    print("\n📍 Testing Stone Priority 1 (Exact Match)...")
    
    params = make_params(stone_priority='Ưu tiên 1')
    matches = predictor.find_matching_products(**params)
    
    if len(matches) > 0:
        bd_matches = matches[matches['stone_color_type'] == 'BD']
        success = len(bd_matches) == len(matches)
        print(f"  {'✅' if success else '❌'} P1 Stone: All {len(matches)} matches are BD: {success}")
        return success
    else:
        print(f"  ⚠️ P1 Stone: No matches found for BD")
        return True


def test_stone_priority_2(predictor, df):
    """Test Stone Priority 2: Same stone family match."""
    print("\n📍 Testing Stone Priority 2 (Same Family)...")
    
    params = make_params(stone_priority='Ưu tiên 2')
    matches = predictor.find_matching_products(**params)
    
    if len(matches) > 0:
        basalt_family = ['BD', 'BX', 'BT']
        basalt_matches = matches[matches['stone_color_type'].isin(basalt_family)]
        success = len(basalt_matches) == len(matches)
        stone_types = matches['stone_color_type'].unique().tolist()
        print(f"  {'✅' if success else '❌'} P2 Stone: All matches in BASALT family: {stone_types}")
        return success
    else:
        print(f"  ⚠️ P2 Stone: No matches found for BD family")
        return True


def test_stone_priority_3(predictor, df):
    """Test Stone Priority 3: All stone types."""
    print("\n📍 Testing Stone Priority 3 (All Types)...")
    
    params = make_params(stone_priority='Ưu tiên 3')
    matches = predictor.find_matching_products(**params)
    
    if len(matches) > 0:
        stone_types = matches['stone_color_type'].unique().tolist()
        success = len(stone_types) > 1 or len(matches) > 0
        print(f"  {'✅' if success else '❌'} P3 Stone: Found {len(stone_types)} stone types: {stone_types[:5]}...")
        return success
    else:
        print(f"  ⚠️ P3 Stone: No matches found")
        return True


def test_processing_priority_1(predictor, df):
    """Test Processing Priority 1: Exact processing code match."""
    print("\n🔧 Testing Processing Priority 1 (Exact Match)...")
    
    params = make_params(processing_priority='Ưu tiên 1')
    matches = predictor.find_matching_products(**params)
    
    if len(matches) > 0:
        dox_matches = matches[matches['processing_code'] == 'DOX']
        success = len(dox_matches) == len(matches)
        print(f"  {'✅' if success else '❌'} P1 Processing: All {len(matches)} matches are DOX: {success}")
        return success
    else:
        print(f"  ⚠️ P1 Processing: No matches found for DOX")
        return True


def test_processing_priority_2(predictor, df):
    """Test Processing Priority 2: Same processing group match."""
    print("\n🔧 Testing Processing Priority 2 (Same Group)...")
    
    params = make_params(processing_priority='Ưu tiên 2')
    matches = predictor.find_matching_products(**params)
    
    if len(matches) > 0:
        gia_cong_may = PROCESSING_GROUPS.get('GIA_CONG_MAY', [])
        group_matches = matches[matches['processing_code'].isin(gia_cong_may)]
        success = len(group_matches) == len(matches)
        processing_codes = matches['processing_code'].unique().tolist()
        print(f"  {'✅' if success else '❌'} P2 Processing: All {len(matches)} in GIA_CONG_MAY: {processing_codes}")
        return success
    else:
        print(f"  ⚠️ P2 Processing: No matches found for DOX group")
        return True


def test_processing_priority_3(predictor, df):
    """Test Processing Priority 3: All processing types."""
    print("\n🔧 Testing Processing Priority 3 (All Types)...")
    
    params = make_params(processing_priority='Ưu tiên 3')
    matches = predictor.find_matching_products(**params)
    
    if len(matches) > 0:
        processing_codes = matches['processing_code'].unique().tolist()
        success = len(processing_codes) >= 1
        print(f"  {'✅' if success else '❌'} P3 Processing: Found {len(processing_codes)} types: {processing_codes[:5]}...")
        return success
    else:
        print(f"  ⚠️ P3 Processing: No matches found")
        return True


def test_dimension_priority_1(predictor, df):
    """Test Dimension Priority 1: Exact dimensions (±0 tolerance)."""
    print("\n📏 Testing Dimension Priority 1 (Exact Match)...")
    
    params = make_params(dimension_priority='Ưu tiên 1 - Đúng kích thước')
    matches = predictor.find_matching_products(**params)
    
    tolerances = DIMENSION_PRIORITY_LEVELS.get('Ưu tiên 1 - Đúng kích thước', {'height': 0, 'width': 0, 'length': 0})
    print(f"  ℹ️ P1 Dimension tolerances: H±{tolerances['height']}, W±{tolerances['width']}, L±{tolerances['length']}")
    
    if len(matches) > 0:
        success = True
        for _, row in matches.iterrows():
            h_diff = abs(row['height_cm'] - 15)
            w_diff = abs(row['width_cm'] - 35)
            l_diff = abs(row['length_cm'] - 100)
            if h_diff > tolerances['height'] or w_diff > tolerances['width'] or l_diff > tolerances['length']:
                success = False
                break
        print(f"  {'✅' if success else '❌'} P1 Dimension: {len(matches)} exact matches found")
        return success
    else:
        print(f"  ⚠️ P1 Dimension: No exact matches for 100x35x15")
        return True


def test_dimension_priority_2(predictor, df):
    """Test Dimension Priority 2: Small tolerance."""
    print("\n📏 Testing Dimension Priority 2 (Small Tolerance)...")
    
    params = make_params(dimension_priority='Ưu tiên 2 - Sai lệch nhỏ')
    matches = predictor.find_matching_products(**params)
    
    tolerances = DIMENSION_PRIORITY_LEVELS.get('Ưu tiên 2 - Sai lệch nhỏ', {'height': 1, 'width': 5, 'length': 10})
    print(f"  ℹ️ P2 Dimension tolerances: H±{tolerances['height']}, W±{tolerances['width']}, L±{tolerances['length']}")
    
    if len(matches) > 0:
        p1_params = make_params(dimension_priority='Ưu tiên 1 - Đúng kích thước')
        p1_matches = predictor.find_matching_products(**p1_params)
        success = len(matches) >= len(p1_matches)
        print(f"  {'✅' if success else '❌'} P2 Dimension: {len(matches)} matches (vs {len(p1_matches)} in P1)")
        return success
    else:
        print(f"  ⚠️ P2 Dimension: No matches with small tolerance")
        return True


def test_dimension_priority_3(predictor, df):
    """Test Dimension Priority 3: Large tolerance."""
    print("\n📏 Testing Dimension Priority 3 (Large Tolerance)...")
    
    params = make_params(dimension_priority='Ưu tiên 3 - Sai lệch lớn')
    matches = predictor.find_matching_products(**params)
    
    tolerances = DIMENSION_PRIORITY_LEVELS.get('Ưu tiên 3 - Sai lệch lớn', {'height': 5, 'width': 20, 'length': 30})
    print(f"  ℹ️ P3 Dimension tolerances: H±{tolerances['height']}, W±{tolerances['width']}, L±{tolerances['length']}")
    
    if len(matches) > 0:
        p2_params = make_params(dimension_priority='Ưu tiên 2 - Sai lệch nhỏ')
        p2_matches = predictor.find_matching_products(**p2_params)
        success = len(matches) >= len(p2_matches)
        print(f"  {'✅' if success else '❌'} P3 Dimension: {len(matches)} matches (vs {len(p2_matches)} in P2)")
        return success
    else:
        print(f"  ⚠️ P3 Dimension: No matches with large tolerance")
        return True


def test_dimension_priority_3_no_length_limit(predictor, df):
    """Test Dimension Priority 3 with no length limit."""
    print("\n📏 Testing Dimension Priority 3 (No Length Limit)...")
    
    params = make_params(
        dimension_priority='Ưu tiên 3 - Sai lệch lớn',
        no_length_limit=True
    )
    matches = predictor.find_matching_products(**params)
    
    print(f"  ℹ️ P3 Dimension with no_length_limit=True")
    
    if len(matches) > 0:
        p3_params = make_params(dimension_priority='Ưu tiên 3 - Sai lệch lớn', no_length_limit=False)
        p3_with_limit = predictor.find_matching_products(**p3_params)
        success = len(matches) >= len(p3_with_limit)
        lengths = sorted(matches['length_cm'].unique().tolist())
        print(f"  {'✅' if success else '❌'} No Length Limit: {len(matches)} matches, lengths: {lengths[:5]}...")
        return success
    else:
        print(f"  ⚠️ P3 No Length Limit: No matches found")
        return True


def test_region_priority_1(predictor, df):
    """Test Region Priority 1: Exact billing country match."""
    print("\n🌍 Testing Region Priority 1 (Billing Country)...")
    
    params = make_params(
        region_priority='Ưu tiên 1',
        billing_country='Germany'
    )
    matches = predictor.find_matching_products(**params)
    
    if len(matches) > 0:
        germany_matches = matches[matches['billing_country'] == 'Germany']
        success = len(germany_matches) == len(matches)
        print(f"  {'✅' if success else '❌'} P1 Region: All {len(matches)} matches are Germany: {success}")
        return success
    else:
        print(f"  ⚠️ P1 Region: No matches found for Germany")
        return True


def test_region_priority_2(predictor, df):
    """Test Region Priority 2: Same regional group match."""
    print("\n🌍 Testing Region Priority 2 (Regional Group)...")
    
    params = make_params(region_priority='Ưu tiên 2')
    matches = predictor.find_matching_products(**params)
    
    if len(matches) > 0:
        group_matches = matches[matches['customer_regional_group'] == 'Nhóm đầu 2']
        success = len(group_matches) == len(matches)
        print(f"  {'✅' if success else '❌'} P2 Region: All {len(matches)} matches are Nhóm đầu 2: {success}")
        return success
    else:
        print(f"  ⚠️ P2 Region: No matches found for Nhóm đầu 2")
        return True


def test_region_priority_3(predictor, df):
    """Test Region Priority 3: All regions."""
    print("\n🌍 Testing Region Priority 3 (All Regions)...")
    
    params = make_params(region_priority='Ưu tiên 3')
    matches = predictor.find_matching_products(**params)
    
    if len(matches) > 0:
        regions = matches['customer_regional_group'].unique().tolist()
        success = len(regions) >= 1
        print(f"  {'✅' if success else '❌'} P3 Region: Found {len(regions)} groups: {regions[:5]}...")
        return success
    else:
        print(f"  ⚠️ P3 Region: No matches found")
        return True


def test_processing_group_mapping():
    """Test that all processing codes are correctly mapped to groups."""
    print("\n🗺️ Testing Processing Code -> Group Mapping...")
    
    test_cases = [
        ('CTA', 'GIA_CONG_TAY', 'Gia công Tay'),
        ('CUA', 'GIA_CONG_MAY_TAY', 'Gia công Máy + Tay'),
        ('DOX', 'GIA_CONG_MAY', 'Gia công Máy'),
        ('HON', 'GIA_CONG_MAY_CAO_CAP', 'Gia công Máy Cao cấp'),
        ('CLO', 'GIA_CONG_MAY_TAY', 'Gia công Máy + Tay'),
        ('DOT', 'GIA_CONG_MAY', 'Gia công Máy'),
        ('QME', 'GIA_CONG_MAY_TAY', 'Gia công Máy + Tay'),
    ]
    
    all_pass = True
    for code, expected_group, expected_name in test_cases:
        actual_group = PROCESSING_CODE_TO_GROUP.get(code, 'UNKNOWN')
        actual_name = PROCESSING_GROUP_NAMES.get(actual_group, 'Unknown')
        success = actual_group == expected_group and actual_name == expected_name
        all_pass = all_pass and success
        print(f"  {'✅' if success else '❌'} {code} -> {actual_name}")
    
    return all_pass


def test_stone_family_mapping():
    """Test that stone family mapping is correct."""
    print("\n🪨 Testing Stone Family Mapping...")
    
    test_cases = [
        ('BD', 'BASALT'),
        ('BX', 'BASALT'),
        ('BT', 'BASALT'),
        ('GX', 'GRANITE'),
        ('GT', 'GRANITE'),
        ('MB', 'MARBLE'),
        ('MT', 'MARBLE'),
    ]
    
    all_pass = True
    for code, expected_family in test_cases:
        actual_family = STONE_FAMILY_MAP.get(code, 'UNKNOWN')
        success = actual_family == expected_family
        all_pass = all_pass and success
        print(f"  {'✅' if success else '❌'} {code} -> {actual_family}")
    
    return all_pass


def test_get_all_charge_units(predictor, df):
    """Test Get All Charge Units: bypasses charge_unit filter."""
    print("\n💱 Testing Get All Charge Units...")
    
    # Test 1: With get_all_charge_units=False (default), should filter by charge_unit
    params_filtered = make_params(charge_unit='USD/PC', get_all_charge_units=False)
    matches_filtered = predictor.find_matching_products(**params_filtered)
    
    # Test 2: With get_all_charge_units=True, should include all charge units
    params_all = make_params(charge_unit='USD/PC', get_all_charge_units=True)
    matches_all = predictor.find_matching_products(**params_all)
    
    all_pass = True
    
    # Check that filtered results only have the specified charge_unit
    if len(matches_filtered) > 0:
        units_filtered = matches_filtered['charge_unit'].unique().tolist()
        success = all(u == 'USD/PC' for u in units_filtered)
        all_pass = all_pass and success
        print(f"  {'✅' if success else '❌'} Filtered: All {len(matches_filtered)} matches are USD/PC: {units_filtered}")
    else:
        print(f"  ⚠️ Filtered: No matches found for USD/PC")
    
    # Check that unfiltered results include more/equal products
    if len(matches_all) > 0:
        units_all = matches_all['charge_unit'].unique().tolist()
        success = len(matches_all) >= len(matches_filtered)
        all_pass = all_pass and success
        print(f"  {'✅' if success else '❌'} All units: {len(matches_all)} matches >= {len(matches_filtered)} filtered, units: {units_all}")
        
        # If there are products with different units, the all-units count should be strictly more
        if len(units_all) > 1:
            success2 = len(matches_all) > len(matches_filtered)
            all_pass = all_pass and success2
            print(f"  {'✅' if success2 else '❌'} Multiple units found, all-units count ({len(matches_all)}) > filtered ({len(matches_filtered)})")
    else:
        print(f"  ⚠️ All units: No matches found")
    
    return all_pass


def run_all_tests():
    """Run all priority state tests."""
    print("=" * 70)
    print("PRIORITY STATE TEST SUITE")
    print("Tests all priority selection combinations")
    print("=" * 70)
    
    df = load_test_data()
    if df is None:
        print("❌ Cannot run tests without data")
        return False
    
    predictor = SimilarityPricePredictor()
    predictor.load_data(df)
    print(f"✅ Initialized predictor with {len(predictor.data)} valid records")
    
    results = []
    
    # Mapping Tests
    print("\n" + "=" * 50)
    print("MAPPING TESTS")
    print("=" * 50)
    results.append(("Processing Group Mapping", test_processing_group_mapping()))
    results.append(("Stone Family Mapping", test_stone_family_mapping()))
    
    # Stone Priority Tests
    print("\n" + "=" * 50)
    print("STONE PRIORITY TESTS")
    print("=" * 50)
    results.append(("Stone P1 (Exact)", test_stone_priority_1(predictor, df)))
    results.append(("Stone P2 (Family)", test_stone_priority_2(predictor, df)))
    results.append(("Stone P3 (All)", test_stone_priority_3(predictor, df)))
    
    # Processing Priority Tests
    print("\n" + "=" * 50)
    print("PROCESSING PRIORITY TESTS")
    print("=" * 50)
    results.append(("Processing P1 (Exact)", test_processing_priority_1(predictor, df)))
    results.append(("Processing P2 (Group)", test_processing_priority_2(predictor, df)))
    results.append(("Processing P3 (All)", test_processing_priority_3(predictor, df)))
    
    # Dimension Priority Tests
    print("\n" + "=" * 50)
    print("DIMENSION PRIORITY TESTS")
    print("=" * 50)
    results.append(("Dimension P1 (Exact)", test_dimension_priority_1(predictor, df)))
    results.append(("Dimension P2 (Small)", test_dimension_priority_2(predictor, df)))
    results.append(("Dimension P3 (Large)", test_dimension_priority_3(predictor, df)))
    results.append(("Dimension P3 (No Length)", test_dimension_priority_3_no_length_limit(predictor, df)))
    
    # Region Priority Tests
    print("\n" + "=" * 50)
    print("REGION PRIORITY TESTS")
    print("=" * 50)
    results.append(("Region P1 (Country)", test_region_priority_1(predictor, df)))
    results.append(("Region P2 (Group)", test_region_priority_2(predictor, df)))
    results.append(("Region P3 (All)", test_region_priority_3(predictor, df)))
    
    # Charge Unit Tests
    print("\n" + "=" * 50)
    print("CHARGE UNIT TESTS")
    print("=" * 50)
    results.append(("Get All Charge Units", test_get_all_charge_units(predictor, df)))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST RESULTS SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        print(f"  {'✅' if result else '❌'} {name}")
    
    print("\n" + "-" * 50)
    print(f"  TOTAL: {passed}/{total} tests passed ({100*passed/total:.0f}%)")
    
    if passed == total:
        print("  🎉 All tests passed!")
    else:
        print(f"  ⚠️ {total - passed} tests failed")
    
    print("=" * 70)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
