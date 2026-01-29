"""
Advanced Calculator Tab (🧊 Tính giá nâng cao)
Extracted from app.py for better code organization.

This module provides the advanced price calculation with 3D model visualization,
multi-surface processing selection, and special shape volume calculations.
"""
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

# Import shared constants and functions from main app
from app import (
    # Constants
    STONE_COLOR_TYPES,
    STONE_COLOR_LOOKUP,
    SPECIAL_SHAPES,
    SPECIAL_SHAPE_INPUTS,
    PROCESSING_CODES,
    CUSTOMER_PRICING_RULES,
    CHARGE_UNITS,
    # Functions
    calculate_volume_m3,
    calculate_special_shape_volume_m3,
    generate_3d_textured_cuboid,
    generate_cuboid_stl,
    generate_cuboid_3mf,
    calculate_multi_surface_price,
    calculate_customer_price,
    classify_segment,
    get_segment_color,
)


def render_advanced_calculator():
    """
    Render the Advanced Calculator tab (Tab 2).
    This provides 3D model visualization with per-surface processing selection.
    """
    st.subheader("🧊 Tính giá nâng cao (beta) - 3D Model")
    st.markdown("Chọn gia công cho từng mặt của khối đá và xem mô hình 3D tương tác")
    
    col_input, col_3d = st.columns([1, 2])
    
    with col_input:
        st.markdown("#### 📦 Thông tin sản phẩm")
        
        # Stone type
        adv_stone_color = st.selectbox(
            "Màu đá (Stone Color)",
            options=[code for code, label in STONE_COLOR_TYPES],
            format_func=lambda x: STONE_COLOR_LOOKUP.get(x, x),
            key="adv_stone_color"
        )
        
        # Dimensions
        st.markdown("##### Kích thước")
        adv_length = st.number_input("Dài (cm)", min_value=1.0, max_value=300.0, value=60.0, step=1.0, key="adv_length")
        adv_width = st.number_input("Rộng (cm)", min_value=1.0, max_value=300.0, value=40.0, step=1.0, key="adv_width") 
        adv_height = st.number_input("Cao (cm)", min_value=0.5, max_value=50.0, value=5.0, step=0.5, key="adv_height")
        
        st.divider()
        
        # Special Shape Selection
        st.markdown("#### 🔷 Hình dạng đặc biệt (Special Shape)")
        
        special_shape_lookup = {code: f"{code} - {vn} ({en})" for code, vn, en in SPECIAL_SHAPES}
        adv_special_shape = st.selectbox(
            "Loại hình dạng",
            options=['R'] + [code for code, vn, en in SPECIAL_SHAPES if code != 'R'],
            format_func=lambda x: special_shape_lookup.get(x, x) if x in special_shape_lookup else "R - Hình chữ nhật (Rectangular)",
            key="adv_special_shape",
            help="Chọn hình dạng sản phẩm. R = Hình chữ nhật tiêu chuẩn"
        )
        
        # Shape-specific inputs
        adv_shape_params = {}
        shape_config = SPECIAL_SHAPE_INPUTS.get(adv_special_shape, {})
        if shape_config.get('inputs'):
            st.markdown("##### 📐 Thông số hình dạng")
            for input_def in shape_config['inputs']:
                input_key = input_def['key']
                label = f"{input_def['label']} ({input_def['unit']})" if input_def['unit'] else input_def['label']
                
                if input_key == 'hole_count':
                    adv_shape_params[input_key] = st.number_input(
                        label,
                        min_value=int(input_def['min']),
                        max_value=int(input_def['max']),
                        value=int(input_def['default'] or 1),
                        step=1,
                        key=f"adv_special_{input_key}"
                    )
                else:
                    adv_shape_params[input_key] = st.number_input(
                        label,
                        min_value=float(input_def['min']),
                        max_value=float(input_def['max']),
                        value=float(input_def['default'] or input_def['min']),
                        step=0.5,
                        key=f"adv_special_{input_key}"
                    )
        
        # Show formula for selected shape
        if shape_config:
            formula = shape_config.get('formula', 'V = L×W×H')
            note = shape_config.get('note', '')
            st.info(f"📏 *{formula}*" + (f"\n\n{note}" if note else ""))
        
        # STL File Upload for automatic volume calculation
        st.markdown("##### 📁 Hoặc tải file STL")
        uploaded_stl = st.file_uploader(
            "Tải file STL để tính thể tích tự động",
            type=['stl'],
            key="stl_upload",
            help="Tải lên file STL 3D để tính thể tích chính xác. Đơn vị file: mm"
        )
        
        adv_volume_m3 = None
        if uploaded_stl is not None:
            try:
                import tempfile
                from stl import mesh
                
                # Save to temp file and load
                with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmp:
                    tmp.write(uploaded_stl.getvalue())
                    tmp_path = tmp.name
                
                stl_mesh = mesh.Mesh.from_file(tmp_path)
                volume, cog, inertia = stl_mesh.get_mass_properties()
                adv_volume_m3 = abs(volume) / 1e9  # mm³ to m³
                
                st.success(f"✅ Thể tích từ STL: **{adv_volume_m3:.6f} m³** ({abs(volume):,.0f} mm³)")
                
                # Cleanup temp file
                import os
                os.unlink(tmp_path)
            except Exception as e:
                st.error(f"❌ Lỗi đọc file STL: {e}")
        else:
            # Calculate volume from shape
            if adv_special_shape and adv_special_shape != 'R':
                adv_volume_m3 = calculate_special_shape_volume_m3(
                    shape_code=adv_special_shape,
                    length_cm=adv_length,
                    width_cm=adv_width,
                    height_cm=adv_height,
                    **adv_shape_params
                )
            else:
                adv_volume_m3 = calculate_volume_m3(adv_length, adv_width, adv_height)
            
            st.metric("Thể tích ước tính", f"{adv_volume_m3:.6f} m³")
        
        st.divider()
        
        # 6-Surface Processing Selection
        st.markdown("#### 🔧 Gia công từng mặt")
        
        processing_lookup = {code: (eng, vn) for code, eng, vn in PROCESSING_CODES}
        processing_options = [code for code, eng, vn in PROCESSING_CODES]
        
        def format_proc(x):
            return f"{x} - {processing_lookup.get(x, ('Other', 'Khác'))[1]}"
        
        # Surface labels in Vietnamese
        surface_labels = {
            'top': '🔝 Mặt trên (Top)',
            'bottom': '🔻 Mặt đáy (Bottom)',
            'front': '⬛ Mặt trước (Front)',
            'back': '⬜ Mặt sau (Back)',
            'left': '◀️ Mặt trái (Left)',
            'right': '▶️ Mặt phải (Right)',
        }
        
        # Default processing codes for each surface
        default_processing = {
            'top': 'DOT',      # Flamed for top (visible)
            'bottom': 'CUA',   # Sawn for bottom
            'front': 'DOC',    # Flamed brush for front
            'back': 'CUA',     # Sawn for back
            'left': 'CUA',     # Sawn for left
            'right': 'CUA',    # Sawn for right
        }
        
        surface_processing = {}
        for surface in ['top', 'bottom', 'front', 'back', 'left', 'right']:
            default_idx = processing_options.index(default_processing[surface]) if default_processing[surface] in processing_options else 0
            surface_processing[surface] = st.selectbox(
                surface_labels[surface],
                options=processing_options,
                format_func=format_proc,
                index=default_idx,
                key=f"adv_proc_{surface}"
            )
        
        st.divider()
        
        # Customer classification
        adv_customer_type = st.selectbox(
            "Phân loại khách hàng",
            ['C', 'A', 'B', 'D', 'E', 'F'],
            format_func=lambda x: f"{x} - {CUSTOMER_PRICING_RULES[x]['description']}",
            key="adv_customer_type"
        )
        
        adv_charge_unit = st.selectbox("Đơn vị tính giá", CHARGE_UNITS, key="adv_charge_unit")
        
        # Predict button
        adv_predict_btn = st.button("🧮 Ước tính giá nâng cao", type="primary", use_container_width=True, key="adv_predict_btn")
    
    with col_3d:
        st.markdown("#### 🧊 Mô hình 3D")
        
        # Use Three.js 3D viewer with per-face textures
        html_3d = generate_3d_textured_cuboid(adv_length, adv_width, adv_height, surface_processing)
        components.html(html_3d, height=450)
        
        # Export for CAD/Modeling
        st.divider()
        st.markdown("##### 📥 Xuất file 3D")
        st.caption("Tải xuống file 3D để sử dụng trong phần mềm CAD/modeling")
        
        # Generate export content
        stl_content = generate_cuboid_stl(adv_length, adv_width, adv_height)
        threemf_content = generate_cuboid_3mf(adv_length, adv_width, adv_height, surface_processing)
        
        col_stl, col_3mf = st.columns(2)
        with col_stl:
            st.download_button(
                label="📦 STL",
                data=stl_content,
                file_name=f"stone_{int(adv_length)}x{int(adv_width)}x{int(adv_height)}.stl",
                mime="application/sla",
                use_container_width=True,
                key="download_stl_btn",
                help="STL format - geometry only, universal compatibility"
            )
        with col_3mf:
            st.download_button(
                label="🎨 3MF",
                data=threemf_content,
                file_name=f"stone_{int(adv_length)}x{int(adv_width)}x{int(adv_height)}.3mf",
                mime="application/vnd.ms-package.3dmanufacturing-3dmodel+xml",
                use_container_width=True,
                key="download_3mf_btn",
                help="3MF format - geometry + colors in one file"
            )
        
        st.caption("💡 **STL**: Geometry only | **3MF**: Includes face colors for each processing type")
    

    # Price calculation results
    if adv_predict_btn and st.session_state.model is not None:
        st.divider()
        st.markdown("### 📊 Kết quả ước tính giá nâng cao")
        
        predictor = st.session_state.model
        
        # Find base price using similarity matching
        main_proc = surface_processing.get('top', 'DOT')  # Use top surface as main processing
        matches = predictor.find_matching_products(
            stone_color_type=adv_stone_color,
            processing_code=main_proc,
            length_cm=adv_length,
            width_cm=adv_width,
            height_cm=adv_height,
            application_codes=[],
            customer_regional_group='',
            charge_unit='USD/M3',
            dimension_priority='Ưu tiên 3 - Sai lệch lớn',
            region_priority='Ưu tiên 3',
            special_shape=adv_special_shape if adv_special_shape != 'R' else None,
        )
        
        if len(matches) > 0:
            base_estimation = predictor.estimate_price(
                matches,
                query_length_cm=adv_length,
                query_width_cm=adv_width,
                query_height_cm=adv_height,
                target_charge_unit='USD/M3',
                stone_color_type=adv_stone_color,
                processing_code=main_proc,
                special_shape=adv_special_shape if adv_special_shape != 'R' else None,
                shape_params=adv_shape_params if adv_special_shape != 'R' else None
            )
            base_price_m3 = base_estimation.get('price_m3', 500)  # Default if not available
        else:
            base_price_m3 = 500  # Default base price
            st.warning("⚠️ Không tìm thấy sản phẩm tham khảo. Sử dụng giá cơ sở mặc định.")
        
        # Calculate multi-surface price using the accurate volume 
        # (either from STL or shape calculation)
        price_result = calculate_multi_surface_price(
            base_price_m3=base_price_m3,
            surface_processing=surface_processing,
            length_cm=adv_length,
            width_cm=adv_width,
            height_cm=adv_height,
            stone_color_type=adv_stone_color,
            custom_volume_m3=adv_volume_m3  # Use calculated/STL volume
        )
        
        # Apply customer adjustment
        segment = classify_segment(price_result['final_price_m3'], height_cm=adv_height)
        customer_price_info = calculate_customer_price(
            price_result['final_price_m3'] if adv_charge_unit == 'USD/M3' else 
            price_result['price_per_piece'] if adv_charge_unit == 'USD/PC' else
            price_result['price_per_m2'],
            adv_customer_type,
            segment=segment,
            charge_unit=adv_charge_unit
        )
        
        # Display results
        col_result1, col_result2, col_result3 = st.columns(3)
        
        with col_result1:
            st.metric("💰 Giá ước tính (USD/M³)", f"${price_result['final_price_m3']:,.2f}")
            st.metric("📦 Giá theo viên (USD/PC)", f"${price_result['price_per_piece']:,.2f}")
        
        with col_result2:
            st.metric("📐 Giá theo m² (USD/M²)", f"${price_result['price_per_m2']:,.2f}")
            st.metric("🧊 Thể tích (m³)", f"{price_result['volume_m3']:.6f}")
        
        with col_result3:
            st.metric("📊 Hệ số gia công TB", f"{price_result['weighted_factor']:.3f}")
            if price_result['complexity_premium'] > 0:
                st.metric("⚙️ Phụ thu phức tạp", f"+{price_result['complexity_premium']:.1f}%")
            else:
                st.metric("⚙️ Phụ thu phức tạp", "0%")
        
        # Customer price card
        conf_color = get_segment_color(segment)
        final_price = (customer_price_info['min_price'] + customer_price_info['max_price']) / 2
        
        st.markdown(f"""
        <div style="background-color: {conf_color}; padding: 20px; border-radius: 10px; margin-top: 20px;">
            <p style="color: white; margin: 0; font-size: 1.1em; font-weight: bold;">💵 Giá đề xuất cho khách hàng loại {adv_customer_type} ({adv_charge_unit}):</p>
            <h1 style="color: white; margin: 5px 0; font-size: 3em;">${final_price:,.2f}</h1>
            <p style="color: white; margin: 0;">Khoảng giá: <b>${customer_price_info['min_price']:,.2f}</b> – <b>${customer_price_info['max_price']:,.2f}</b></p>
            <hr style="margin: 10px 0; border-top: 1px solid rgba(255,255,255,0.3);">
            <p style="color: white; margin: 5px 0;">📊 Phân khúc: {segment} | 🔧 Số loại gia công: {price_result['unique_processes']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Surface area breakdown
        with st.expander("📐 Chi tiết diện tích từng mặt", expanded=False):
            area_data = []
            for surface, area in price_result['surface_areas'].items():
                proc = surface_processing.get(surface, 'CUA')
                proc_name = processing_lookup.get(proc, ('Unknown', 'Không xác định'))
                area_data.append({
                    'Mặt': surface_labels.get(surface, surface),
                    'Gia công': f"{proc} - {proc_name[1]}",
                    'Diện tích (m²)': f"{area:.4f}"
                })
            st.dataframe(pd.DataFrame(area_data), use_container_width=True, hide_index=True)
    
    elif adv_predict_btn and st.session_state.model is None:
        st.error("⚠️ Vui lòng tải dữ liệu từ Salesforce trước khi ước tính giá.")
