"""
Tests for special shape volume calculations and pricing.
Based on formulas from special_products_report.tex.
"""
import pytest
import math
import sys
import os

# Add parent directory to path to import app module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import (
    calculate_special_shape_volume_m3,
    calculate_volume_m3,
    SPECIAL_SHAPE_INPUTS,
    SPECIAL_SHAPES,
)


class TestSpecialShapeVolumeCalculations:
    """Test volume calculations for each special shape."""
    
    def test_l_profile_volume(self):
        """
        Test L-profile volume: V = L × (t×W + t×H - t²)
        Example: L=100cm, W=10cm, H=6cm, t=3cm
        V = 1.0 × (0.03×0.10 + 0.03×0.06 - 0.03²) = 0.0039 m³
        """
        volume = calculate_special_shape_volume_m3(
            shape_code='L',
            length_cm=100,
            width_cm=10,
            height_cm=6,
            wall_thickness_cm=3
        )
        # Calculate expected: L × (t×W + t×H - t²)
        L, W, H, t = 1.0, 0.10, 0.06, 0.03  # in meters
        expected = L * (t * W + t * H - t * t)
        assert abs(volume - expected) < 0.00001
        
    def test_l_profile_default_thickness(self):
        """Test L-profile uses default 3cm thickness when not specified."""
        volume = calculate_special_shape_volume_m3(
            shape_code='L',
            length_cm=100,
            width_cm=10,
            height_cm=6,
            wall_thickness_cm=None  # Uses default
        )
        # With default t=3cm
        L, W, H, t = 1.0, 0.10, 0.06, 0.03
        expected = L * (t * W + t * H - t * t)
        assert abs(volume - expected) < 0.00001
    
    def test_u_profile_volume(self):
        """
        Test U-profile volume: V = L × (W×H - (W-2t)(H-t))
        Example: L=100cm, W=10cm, H=4cm, t=3cm
        With W-2t = 4cm and H-t = 1cm (inner dimensions)
        """
        volume = calculate_special_shape_volume_m3(
            shape_code='U',
            length_cm=100,
            width_cm=10,
            height_cm=4,
            wall_thickness_cm=3
        )
        # Calculate expected
        L, W, H, t = 1.0, 0.10, 0.04, 0.03  # in meters
        w_in = W - 2 * t  # 0.04
        h_in = H - t  # 0.01
        expected = L * (W * H - w_in * h_in)
        assert abs(volume - expected) < 0.00001
        
    def test_u_profile_thick_walls(self):
        """Test U-profile with thick walls (inner dimension <= 0) falls back to solid."""
        volume = calculate_special_shape_volume_m3(
            shape_code='U',
            length_cm=100,
            width_cm=6,  # W = 6cm
            height_cm=4,  # H = 4cm
            wall_thickness_cm=3.5  # t = 3.5cm, so W-2t = -1 (negative)
        )
        # Should fallback to solid volume
        L, W, H = 1.0, 0.06, 0.04
        expected = L * W * H
        assert abs(volume - expected) < 0.00001
    
    def test_g_angle_cut_volume(self):
        """
        Test angle cut volume: V = (L×W - ½ab) × H
        Example: L=100cm, W=50cm, H=6cm, a=10cm, b=10cm
        Cut area = ½ × 0.10 × 0.10 = 0.005 m²
        Plan area = 1.0 × 0.50 - 0.005 = 0.495 m²
        V = 0.495 × 0.06 = 0.0297 m³
        """
        volume = calculate_special_shape_volume_m3(
            shape_code='G',
            length_cm=100,
            width_cm=50,
            height_cm=6,
            cut_leg_a_cm=10,
            cut_leg_b_cm=10
        )
        L, W, H, a, b = 1.0, 0.50, 0.06, 0.10, 0.10
        cut_area = 0.5 * a * b
        plan_area = L * W - cut_area
        expected = plan_area * H
        assert abs(volume - expected) < 0.00001
        
    def test_g_no_cut(self):
        """Test angle cut with zero cut dimensions equals rectangular volume."""
        volume = calculate_special_shape_volume_m3(
            shape_code='G',
            length_cm=100,
            width_cm=50,
            height_cm=6,
            cut_leg_a_cm=0,
            cut_leg_b_cm=0
        )
        expected = calculate_volume_m3(100, 50, 6)
        assert abs(volume - expected) < 0.00001
    
    def test_c_arc_cut_volume_90_degrees(self):
        """
        Test arc cut volume: V = (L×W - πr²×(θ/360)) × H
        Example: L=100cm, W=50cm, H=6cm, r=10cm, θ=90°
        Cut area = π × 0.10² × 0.25 = 0.00785 m²
        """
        volume = calculate_special_shape_volume_m3(
            shape_code='C',
            length_cm=100,
            width_cm=50,
            height_cm=6,
            arc_radius_cm=10,
            arc_angle_degrees=90
        )
        L, W, H, r = 1.0, 0.50, 0.06, 0.10
        theta = 90
        cut_area = math.pi * r * r * (theta / 360)
        plan_area = L * W - cut_area
        expected = plan_area * H
        assert abs(volume - expected) < 0.00001
        
    def test_c_arc_cut_full_circle(self):
        """Test arc cut with 360° angle (full hole)."""
        volume = calculate_special_shape_volume_m3(
            shape_code='C',
            length_cm=100,
            width_cm=50,
            height_cm=6,
            arc_radius_cm=5,
            arc_angle_degrees=360
        )
        L, W, H, r = 1.0, 0.50, 0.06, 0.05
        cut_area = math.pi * r * r  # Full circle
        plan_area = L * W - cut_area
        expected = plan_area * H
        assert abs(volume - expected) < 0.00001
    
    def test_k_drilled_hole_volume(self):
        """
        Test drilled hole volume: V = L×W×H - n×π(d/2)²×h
        Example: L=100cm, W=50cm, H=6cm, n=2, d=3cm, h=6cm (full depth)
        """
        volume = calculate_special_shape_volume_m3(
            shape_code='K',
            length_cm=100,
            width_cm=50,
            height_cm=6,
            hole_count=2,
            hole_diameter_cm=3,
            hole_depth_cm=6
        )
        L, W, H = 1.0, 0.50, 0.06
        n, d, h = 2, 0.03, 0.06
        base_volume = L * W * H
        hole_volume = n * math.pi * (d / 2) ** 2 * h
        expected = base_volume - hole_volume
        assert abs(volume - expected) < 0.00001
        
    def test_k_partial_depth_hole(self):
        """Test drilled hole with partial depth."""
        volume = calculate_special_shape_volume_m3(
            shape_code='K',
            length_cm=100,
            width_cm=50,
            height_cm=6,
            hole_count=1,
            hole_diameter_cm=2,
            hole_depth_cm=3  # Half depth
        )
        L, W, H = 1.0, 0.50, 0.06
        n, d, h = 1, 0.02, 0.03
        base_volume = L * W * H
        hole_volume = n * math.pi * (d / 2) ** 2 * h
        expected = base_volume - hole_volume
        assert abs(volume - expected) < 0.00001
    
    def test_t_solid_cylinder_volume(self):
        """
        Test solid cylinder volume: V = π(d/2)²×H
        Uses width as diameter when no outer_radius specified.
        Example: W=20cm (diameter), H=30cm
        """
        volume = calculate_special_shape_volume_m3(
            shape_code='T',
            length_cm=100,  # Not used for cylinder
            width_cm=20,     # Used as diameter
            height_cm=30
        )
        d, H = 0.20, 0.30
        expected = math.pi * (d / 2) ** 2 * H
        assert abs(volume - expected) < 0.00001
        
    def test_t_hollow_cylinder_volume(self):
        """Test hollow cylinder volume: V = π(Ro²-Ri²)×H"""
        volume = calculate_special_shape_volume_m3(
            shape_code='T',
            length_cm=100,  # Not used
            width_cm=20,    # Not used when radii specified
            height_cm=30,
            outer_radius_cm=10,
            inner_radius_cm=5
        )
        Ro, Ri, H = 0.10, 0.05, 0.30
        expected = math.pi * (Ro ** 2 - Ri ** 2) * H
        assert abs(volume - expected) < 0.00001
    
    def test_b_set_uses_rectangular_volume(self):
        """Test B (set/kit) uses standard rectangular volume as approximation."""
        volume = calculate_special_shape_volume_m3(
            shape_code='B',
            length_cm=100,
            width_cm=50,
            height_cm=6
        )
        expected = calculate_volume_m3(100, 50, 6)
        assert volume == expected
    
    def test_v_ring_full_360(self):
        """
        Test ring stone volume: V = π(Ro²-Ri²)×H×(θ/360)
        Example: Ro=50cm, Ri=30cm, H=6cm, θ=360°
        """
        volume = calculate_special_shape_volume_m3(
            shape_code='V',
            length_cm=100,  # Not used
            width_cm=100,   # Used as fallback Ro if not specified
            height_cm=6,
            outer_radius_cm=50,
            inner_radius_cm=30,
            arc_angle_degrees=360
        )
        Ro, Ri, H, theta = 0.50, 0.30, 0.06, 360
        expected = math.pi * (Ro ** 2 - Ri ** 2) * H * (theta / 360)
        assert abs(volume - expected) < 0.00001
        
    def test_v_ring_half_arc(self):
        """Test ring stone with 180° arc (half ring)."""
        volume = calculate_special_shape_volume_m3(
            shape_code='V',
            length_cm=100,
            width_cm=100,
            height_cm=6,
            outer_radius_cm=50,
            inner_radius_cm=30,
            arc_angle_degrees=180
        )
        Ro, Ri, H, theta = 0.50, 0.30, 0.06, 180
        expected = math.pi * (Ro ** 2 - Ri ** 2) * H * (theta / 360)
        assert abs(volume - expected) < 0.00001
    
    def test_unknown_shape_uses_rectangular(self):
        """Test unknown shape code falls back to rectangular volume."""
        volume = calculate_special_shape_volume_m3(
            shape_code='X',  # Unknown shape
            length_cm=100,
            width_cm=50,
            height_cm=6
        )
        expected = calculate_volume_m3(100, 50, 6)
        assert volume == expected


class TestSpecialShapeInputConfig:
    """Test the SPECIAL_SHAPE_INPUTS configuration."""
    
    def test_all_shapes_have_config(self):
        """Verify all SPECIAL_SHAPES have input configuration."""
        shape_codes = [code for code, _, _ in SPECIAL_SHAPES]
        for code in shape_codes:
            assert code in SPECIAL_SHAPE_INPUTS, f"Shape {code} missing from SPECIAL_SHAPE_INPUTS"
    
    def test_config_structure(self):
        """Verify each shape config has required fields."""
        required_fields = ['name_vn', 'name_en', 'inputs', 'formula']
        for code, config in SPECIAL_SHAPE_INPUTS.items():
            for field in required_fields:
                assert field in config, f"Shape {code} missing field {field}"
    
    def test_input_definitions(self):
        """Verify each input definition has required properties."""
        input_fields = ['key', 'label', 'unit', 'default', 'min', 'max']
        for code, config in SPECIAL_SHAPE_INPUTS.items():
            for input_def in config.get('inputs', []):
                for field in input_fields:
                    assert field in input_def, f"Shape {code} input missing field {field}"
    
    def test_l_shape_inputs(self):
        """Test L-shape has wall_thickness_cm input."""
        config = SPECIAL_SHAPE_INPUTS['L']
        input_keys = [inp['key'] for inp in config['inputs']]
        assert 'wall_thickness_cm' in input_keys
        
    def test_u_shape_inputs(self):
        """Test U-shape has wall_thickness_cm input."""
        config = SPECIAL_SHAPE_INPUTS['U']
        input_keys = [inp['key'] for inp in config['inputs']]
        assert 'wall_thickness_cm' in input_keys
        
    def test_g_shape_inputs(self):
        """Test G-shape has cut_leg_a_cm and cut_leg_b_cm inputs."""
        config = SPECIAL_SHAPE_INPUTS['G']
        input_keys = [inp['key'] for inp in config['inputs']]
        assert 'cut_leg_a_cm' in input_keys
        assert 'cut_leg_b_cm' in input_keys
        
    def test_c_shape_inputs(self):
        """Test C-shape has arc_radius_cm and arc_angle_degrees inputs."""
        config = SPECIAL_SHAPE_INPUTS['C']
        input_keys = [inp['key'] for inp in config['inputs']]
        assert 'arc_radius_cm' in input_keys
        assert 'arc_angle_degrees' in input_keys
        
    def test_k_shape_inputs(self):
        """Test K-shape has hole_count, hole_diameter_cm, hole_depth_cm inputs."""
        config = SPECIAL_SHAPE_INPUTS['K']
        input_keys = [inp['key'] for inp in config['inputs']]
        assert 'hole_count' in input_keys
        assert 'hole_diameter_cm' in input_keys
        assert 'hole_depth_cm' in input_keys
        
    def test_t_shape_inputs(self):
        """Test T-shape has outer_radius_cm and inner_radius_cm inputs."""
        config = SPECIAL_SHAPE_INPUTS['T']
        input_keys = [inp['key'] for inp in config['inputs']]
        assert 'outer_radius_cm' in input_keys
        assert 'inner_radius_cm' in input_keys
        
    def test_b_shape_no_inputs(self):
        """Test B-shape has no additional inputs (uses raw prices)."""
        config = SPECIAL_SHAPE_INPUTS['B']
        assert config['inputs'] == []
        assert 'note' in config  # Should have a note explaining no volume normalization
        
    def test_v_shape_inputs(self):
        """Test V-shape has radii and arc angle inputs."""
        config = SPECIAL_SHAPE_INPUTS['V']
        input_keys = [inp['key'] for inp in config['inputs']]
        assert 'outer_radius_cm' in input_keys
        assert 'inner_radius_cm' in input_keys
        assert 'arc_angle_degrees' in input_keys


class TestVolumeComparison:
    """Test that special shape volumes are less than rectangular volumes."""
    
    def test_l_profile_less_than_rectangular(self):
        """L-profile volume should be much less than rectangular block."""
        l_vol = calculate_special_shape_volume_m3(
            'L', 100, 10, 6, wall_thickness_cm=3
        )
        rect_vol = calculate_volume_m3(100, 10, 6)
        assert l_vol < rect_vol
        # L-profile should be roughly 50-80% less (depends on thickness ratio)
        
    def test_u_profile_less_than_rectangular(self):
        """U-profile volume should be less than rectangular block."""
        u_vol = calculate_special_shape_volume_m3(
            'U', 100, 10, 4, wall_thickness_cm=3
        )
        rect_vol = calculate_volume_m3(100, 10, 4)
        assert u_vol < rect_vol
        
    def test_g_cut_less_than_rectangular(self):
        """Angle cut volume should be less than rectangular block."""
        g_vol = calculate_special_shape_volume_m3(
            'G', 100, 50, 6, cut_leg_a_cm=10, cut_leg_b_cm=10
        )
        rect_vol = calculate_volume_m3(100, 50, 6)
        assert g_vol < rect_vol
        
    def test_c_cut_less_than_rectangular(self):
        """Arc cut volume should be less than rectangular block."""
        c_vol = calculate_special_shape_volume_m3(
            'C', 100, 50, 6, arc_radius_cm=10, arc_angle_degrees=90
        )
        rect_vol = calculate_volume_m3(100, 50, 6)
        assert c_vol < rect_vol
        
    def test_k_drilled_less_than_rectangular(self):
        """Drilled hole volume should be less than rectangular block."""
        k_vol = calculate_special_shape_volume_m3(
            'K', 100, 50, 6, hole_count=2, hole_diameter_cm=3, hole_depth_cm=6
        )
        rect_vol = calculate_volume_m3(100, 50, 6)
        assert k_vol < rect_vol
        
    def test_hollow_cylinder_less_than_solid(self):
        """Hollow cylinder should be less than solid cylinder."""
        hollow_vol = calculate_special_shape_volume_m3(
            'T', 100, 20, 30, outer_radius_cm=10, inner_radius_cm=5
        )
        solid_vol = calculate_special_shape_volume_m3(
            'T', 100, 20, 30  # Solid, uses width as diameter
        )
        assert hollow_vol < solid_vol


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
