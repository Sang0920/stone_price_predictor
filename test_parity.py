#!/usr/bin/env python3
"""
Parity Test: FastAPI vs Streamlit Price Prediction

Runs the same set of test cases through both:
  1. The FastAPI endpoint (via HTTP POST)
  2. The Streamlit app's internal prediction logic (via direct function calls)

Then compares the results side-by-side and generates a markdown report.

Usage:
    # Test against local FastAPI server
    python test_parity.py

    # Test against deployed Vercel endpoint
    python test_parity.py --api-url https://stonepricepredictor-fastapi.vercel.app

    # Only run specific test cases
    python test_parity.py --cases 1 3 5
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests

# Add parent directory to path so we can import from app.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from app import (
    SimilarityPricePredictor,
    convert_price,
    classify_segment,
    calculate_customer_price,
    get_tlr,
    get_hs_factor,
    calculate_volume_m3,
    calculate_area_m2,
    calculate_weight_tons,
)
from salesforce_loader import SalesforceDataLoader


# ═══════════════════════════════════════════════════════════════════
#  Test Cases — diverse coverage of stone types, charge units, etc.
# ═══════════════════════════════════════════════════════════════════

TEST_CASES: List[Dict[str, Any]] = [
    # --- Case 1: The exact case from the screenshot (BD 30x30x3, USD/PC) ---
    {
        "name": "BD 30x30x3 USD/PC (screenshot case)",
        "params": {
            "stone_color": "BD",
            "length_cm": 30, "width_cm": 30, "height_cm": 3,
            "charge_unit": "USD/PC",
            "customer_type": "C",
            "stone_priority": "Ưu tiên 2",
            "processing_priority": "Ưu tiên 3",
            "dimension_priority": "Ưu tiên 1 - Đúng kích thước",
            "region_priority": "Ưu tiên 3",
            "use_recent_only": True, "recent_count": 5,
            "apply_yearly_adjustment": True, "yearly_increase_pct": 0.5,
        }
    },
    # --- Case 2: Cube product (BD 9x9x9, USD/PC) ---
    {
        "name": "BD 9x9x9 USD/PC (cube)",
        "params": {
            "stone_color": "BD",
            "length_cm": 9, "width_cm": 9, "height_cm": 9,
            "charge_unit": "USD/PC",
            "customer_type": "C",
            "stone_priority": "Ưu tiên 2",
            "processing_priority": "Ưu tiên 3",
            "dimension_priority": "Ưu tiên 1 - Đúng kích thước",
            "region_priority": "Ưu tiên 3",
            "use_recent_only": True, "recent_count": 5,
            "apply_yearly_adjustment": True, "yearly_increase_pct": 0.5,
        }
    },
    # --- Case 3: Granite exact match (GX, strict priorities) ---
    {
        "name": "GX 30x30x3 USD/M2 (strict)",
        "params": {
            "stone_color": "GX",
            "length_cm": 30, "width_cm": 30, "height_cm": 3,
            "charge_unit": "USD/M2",
            "customer_type": "C",
            "stone_priority": "Ưu tiên 1",
            "processing_priority": "Ưu tiên 3",
            "dimension_priority": "Ưu tiên 1 - Đúng kích thước",
            "region_priority": "Ưu tiên 3",
            "use_recent_only": True, "recent_count": 5,
            "apply_yearly_adjustment": True, "yearly_increase_pct": 0.5,
        }
    },
    # --- Case 4: Customer type D (higher price) ---
    {
        "name": "BD 30x30x3 USD/PC customer D",
        "params": {
            "stone_color": "BD",
            "length_cm": 30, "width_cm": 30, "height_cm": 3,
            "charge_unit": "USD/PC",
            "customer_type": "D",
            "stone_priority": "Ưu tiên 2",
            "processing_priority": "Ưu tiên 3",
            "dimension_priority": "Ưu tiên 1 - Đúng kích thước",
            "region_priority": "Ưu tiên 3",
            "use_recent_only": True, "recent_count": 5,
            "apply_yearly_adjustment": True, "yearly_increase_pct": 0.5,
        }
    },
    # --- Case 5: Customer type B (lower price) ---
    {
        "name": "BD 30x30x3 USD/PC customer B",
        "params": {
            "stone_color": "BD",
            "length_cm": 30, "width_cm": 30, "height_cm": 3,
            "charge_unit": "USD/PC",
            "customer_type": "B",
            "stone_priority": "Ưu tiên 2",
            "processing_priority": "Ưu tiên 3",
            "dimension_priority": "Ưu tiên 1 - Đúng kích thước",
            "region_priority": "Ưu tiên 3",
            "use_recent_only": True, "recent_count": 5,
            "apply_yearly_adjustment": True, "yearly_increase_pct": 0.5,
        }
    },
    # --- Case 6: No yearly adjustment ---
    {
        "name": "BD 30x30x3 USD/PC (no yearly adj)",
        "params": {
            "stone_color": "BD",
            "length_cm": 30, "width_cm": 30, "height_cm": 3,
            "charge_unit": "USD/PC",
            "customer_type": "C",
            "stone_priority": "Ưu tiên 2",
            "processing_priority": "Ưu tiên 3",
            "dimension_priority": "Ưu tiên 1 - Đúng kích thước",
            "region_priority": "Ưu tiên 3",
            "use_recent_only": True, "recent_count": 5,
            "apply_yearly_adjustment": False, "yearly_increase_pct": 0,
        }
    },
    # --- Case 7: Relaxed dimension tolerance ---
    {
        "name": "BD 30x30x3 USD/PC (relaxed dims)",
        "params": {
            "stone_color": "BD",
            "length_cm": 30, "width_cm": 30, "height_cm": 3,
            "charge_unit": "USD/PC",
            "customer_type": "C",
            "stone_priority": "Ưu tiên 2",
            "processing_priority": "Ưu tiên 3",
            "dimension_priority": "Ưu tiên 3 - Sai lệch lớn",
            "region_priority": "Ưu tiên 3",
            "use_recent_only": True, "recent_count": 5,
            "apply_yearly_adjustment": True, "yearly_increase_pct": 0.5,
        }
    },
    # --- Case 8: BX (Basalt Grey) with a different size ---
    {
        "name": "BX 20x10x8 USD/PC (different dims)",
        "params": {
            "stone_color": "BX",
            "length_cm": 20, "width_cm": 10, "height_cm": 8,
            "charge_unit": "USD/PC",
            "customer_type": "C",
            "stone_priority": "Ưu tiên 2",
            "processing_priority": "Ưu tiên 3",
            "dimension_priority": "Ưu tiên 1 - Đúng kích thước",
            "region_priority": "Ưu tiên 3",
            "use_recent_only": True, "recent_count": 5,
            "apply_yearly_adjustment": True, "yearly_increase_pct": 0.5,
        }
    },
    # --- Case 9: More recent_count = 10 ---
    {
        "name": "BD 30x30x3 USD/PC (recent_count=10)",
        "params": {
            "stone_color": "BD",
            "length_cm": 30, "width_cm": 30, "height_cm": 3,
            "charge_unit": "USD/PC",
            "customer_type": "C",
            "stone_priority": "Ưu tiên 2",
            "processing_priority": "Ưu tiên 3",
            "dimension_priority": "Ưu tiên 1 - Đúng kích thước",
            "region_priority": "Ưu tiên 3",
            "use_recent_only": True, "recent_count": 10,
            "apply_yearly_adjustment": True, "yearly_increase_pct": 0.5,
        }
    },
    # --- Case 10: Exact stone priority (Ưu tiên 1) ---
    {
        "name": "BD 30x30x3 USD/PC (exact stone)",
        "params": {
            "stone_color": "BD",
            "length_cm": 30, "width_cm": 30, "height_cm": 3,
            "charge_unit": "USD/PC",
            "customer_type": "C",
            "stone_priority": "Ưu tiên 1",
            "processing_priority": "Ưu tiên 3",
            "dimension_priority": "Ưu tiên 1 - Đúng kích thước",
            "region_priority": "Ưu tiên 3",
            "use_recent_only": True, "recent_count": 5,
            "apply_yearly_adjustment": True, "yearly_increase_pct": 0.5,
        }
    },
    # --- Case 11: USD/TON charge unit ---
    {
        "name": "BD 30x30x3 USD/TON",
        "params": {
            "stone_color": "BD",
            "length_cm": 30, "width_cm": 30, "height_cm": 3,
            "charge_unit": "USD/TON",
            "customer_type": "C",
            "stone_priority": "Ưu tiên 2",
            "processing_priority": "Ưu tiên 3",
            "dimension_priority": "Ưu tiên 1 - Đúng kích thước",
            "region_priority": "Ưu tiên 3",
            "use_recent_only": True, "recent_count": 5,
            "apply_yearly_adjustment": True, "yearly_increase_pct": 0.5,
        }
    },
    # --- Case 12: No-match scenario (likely no matches) ---
    {
        "name": "MV 100x100x1 USD/M3 (likely no match)",
        "params": {
            "stone_color": "MV",
            "length_cm": 100, "width_cm": 100, "height_cm": 1,
            "charge_unit": "USD/M3",
            "customer_type": "C",
            "stone_priority": "Ưu tiên 1",
            "processing_priority": "Ưu tiên 1",
            "processing_code": "BON",
            "dimension_priority": "Ưu tiên 1 - Đúng kích thước",
            "region_priority": "Ưu tiên 1",
            "use_recent_only": True, "recent_count": 5,
            "apply_yearly_adjustment": False, "yearly_increase_pct": 0,
        }
    },
    # --- Case 13: Customer E (premium markup) ---
    {
        "name": "BD 30x30x3 USD/PC customer E",
        "params": {
            "stone_color": "BD",
            "length_cm": 30, "width_cm": 30, "height_cm": 3,
            "charge_unit": "USD/PC",
            "customer_type": "E",
            "stone_priority": "Ưu tiên 2",
            "processing_priority": "Ưu tiên 3",
            "dimension_priority": "Ưu tiên 1 - Đúng kích thước",
            "region_priority": "Ưu tiên 3",
            "use_recent_only": True, "recent_count": 5,
            "apply_yearly_adjustment": True, "yearly_increase_pct": 0.5,
        }
    },
    # --- Case 14: Higher yearly adjustment rate ---
    {
        "name": "BD 30x30x3 USD/PC (yearly 2%)",
        "params": {
            "stone_color": "BD",
            "length_cm": 30, "width_cm": 30, "height_cm": 3,
            "charge_unit": "USD/PC",
            "customer_type": "C",
            "stone_priority": "Ưu tiên 2",
            "processing_priority": "Ưu tiên 3",
            "dimension_priority": "Ưu tiên 1 - Đúng kích thước",
            "region_priority": "Ưu tiên 3",
            "use_recent_only": True, "recent_count": 5,
            "apply_yearly_adjustment": True, "yearly_increase_pct": 2.0,
        }
    },
    # --- Case 15: All-relaxed priorities ---
    {
        "name": "BD 30x30x3 USD/PC (all relaxed)",
        "params": {
            "stone_color": "BD",
            "length_cm": 30, "width_cm": 30, "height_cm": 3,
            "charge_unit": "USD/PC",
            "customer_type": "C",
            "stone_priority": "Ưu tiên 3",
            "processing_priority": "Ưu tiên 3",
            "dimension_priority": "Ưu tiên 3 - Sai lệch lớn",
            "region_priority": "Ưu tiên 3",
            "use_recent_only": True, "recent_count": 5,
            "apply_yearly_adjustment": True, "yearly_increase_pct": 0.5,
        }
    },
]


# ═══════════════════════════════════════════════════════════════════
#  Streamlit-side prediction (direct function calls)
# ═══════════════════════════════════════════════════════════════════

_streamlit_predictor: Optional[SimilarityPricePredictor] = None
_streamlit_df = None


def _get_streamlit_predictor() -> SimilarityPricePredictor:
    """Load predictor using the same logic as the Streamlit app."""
    global _streamlit_predictor, _streamlit_df
    if _streamlit_predictor is not None:
        return _streamlit_predictor

    print("  Loading Salesforce data for Streamlit predictor...")
    loader = SalesforceDataLoader()
    _streamlit_df = loader.get_contract_products()
    _streamlit_predictor = SimilarityPricePredictor()
    n = _streamlit_predictor.load_data(_streamlit_df)
    print(f"  Loaded {n:,} products")
    return _streamlit_predictor


def run_streamlit_prediction(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the prediction pipeline exactly as the Streamlit app does it.
    Returns a dict with the same structure as the FastAPI response.
    """
    predictor = _get_streamlit_predictor()

    stone_color = params["stone_color"]
    length = params["length_cm"]
    width = params["width_cm"]
    height = params["height_cm"]
    charge_unit = params["charge_unit"]
    customer_type = params["customer_type"]
    processing_code = params.get("processing_code")
    application_codes = params.get("application_codes", [])
    stone_priority = params.get("stone_priority", "Ưu tiên 2")
    processing_priority = params.get("processing_priority", "Ưu tiên 3")
    dimension_priority = params.get("dimension_priority", "Ưu tiên 1 - Đúng kích thước")
    region_priority = params.get("region_priority", "Ưu tiên 3")
    customer_regional_group = params.get("customer_regional_group")
    billing_country = params.get("billing_country")
    get_all_charge_units = params.get("get_all_charge_units", False)
    no_length_limit = params.get("no_length_limit", False)
    use_recent_only = params.get("use_recent_only", True)
    recent_count = params.get("recent_count", 5)
    apply_yearly_adjustment = params.get("apply_yearly_adjustment", True)
    yearly_increase_pct = params.get("yearly_increase_pct", 0.5)
    selected_processing_group = params.get("processing_group")

    # Step 1: Find matches (same as app.py line 3753)
    matches = predictor.find_matching_products(
        stone_color_type=stone_color,
        processing_code=processing_code,
        length_cm=length,
        width_cm=width,
        height_cm=height,
        application_codes=application_codes,
        customer_regional_group=customer_regional_group,
        charge_unit=charge_unit,
        get_all_charge_units=get_all_charge_units,
        stone_priority=stone_priority,
        processing_priority=processing_priority,
        dimension_priority=dimension_priority,
        region_priority=region_priority,
        no_length_limit=no_length_limit,
        billing_country=billing_country,
        selected_processing_group=selected_processing_group,
    )

    total_matches = len(matches)

    if total_matches == 0:
        return {
            "success": False,
            "predicted_price": None,
            "charge_unit": charge_unit,
            "price_m3": None,
            "segment": None,
            "match_count": 0,
            "total_matches": 0,
        }

    # Step 2: Estimate price (same as app.py line 3779)
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
        application_codes=application_codes,
        stone_priority=stone_priority,
        processing_priority=processing_priority,
        dimension_priority=dimension_priority,
        region_priority=region_priority,
    )

    estimated_price = estimation.get("estimated_price")
    if estimated_price is None:
        return {
            "success": False,
            "predicted_price": None,
            "charge_unit": charge_unit,
            "price_m3": None,
            "segment": None,
            "match_count": estimation.get("match_count", 0),
            "total_matches": estimation.get("total_matches", 0),
        }

    price_m3 = estimation.get("estimated_price_m3") or estimation.get("price_m3", 0)

    # Step 3: Segment classification (same as app.py line 3841)
    first_app = application_codes[0] if application_codes else ""
    est_price_m3 = convert_price(
        estimated_price, charge_unit, "USD/M3",
        height_cm=height, length_cm=length, width_cm=width,
        tlr=get_tlr(stone_color, processing_code)
    )
    segment = classify_segment(est_price_m3, height_cm=height, family=first_app, processing_code=processing_code)

    # Step 4: Customer adjustment (same as app.py line 3849)
    price_info = calculate_customer_price(
        estimated_price, customer_type,
        segment=segment, charge_unit=charge_unit
    )
    final_price = (price_info["min_price"] + price_info["max_price"]) / 2

    # Step 5: Yearly adjustment (same as display_estimation_result)
    yearly_adj = None
    if apply_yearly_adjustment and yearly_increase_pct > 0:
        current_year = datetime.now().year
        avg_fy_year = estimation.get("avg_fy_year", current_year)
        if avg_fy_year and avg_fy_year < current_year:
            years_diff = current_year - int(avg_fy_year)
            factor = (1 + yearly_increase_pct / 100) ** years_diff
            final_price *= factor
            yearly_adj = {
                "applied": True,
                "rate_pct": yearly_increase_pct,
                "avg_data_year": int(avg_fy_year),
                "years_diff": years_diff,
                "adjusted_price": round(final_price, 2),
            }

    return {
        "success": True,
        "predicted_price": round(final_price, 2),
        "charge_unit": charge_unit,
        "price_m3": round(price_m3, 2) if price_m3 else None,
        "segment": segment,
        "confidence_score": estimation.get("confidence_score", 0),
        "confidence_level": estimation.get("confidence", "none"),
        "match_count": estimation.get("match_count", 0),
        "total_matches": estimation.get("total_matches", total_matches),
        "min_price": round(price_info["min_price"], 2),
        "max_price": round(price_info["max_price"], 2),
        "yearly_adjustment": yearly_adj,
        "median_price": estimation.get("median_price"),
        "avg_fy_year": estimation.get("avg_fy_year"),
    }


# ═══════════════════════════════════════════════════════════════════
#  FastAPI-side prediction (HTTP request)
# ═══════════════════════════════════════════════════════════════════

def run_fastapi_prediction(api_url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Send a POST request to the FastAPI endpoint and parse the response."""
    url = f"{api_url.rstrip('/')}/predict-price"
    try:
        resp = requests.post(url, json=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        # Normalize to same structure as Streamlit result
        predicted_price = data.get("predicted_price")
        yearly_adj = data.get("yearly_adjustment")

        return {
            "success": data.get("success", False),
            "predicted_price": predicted_price,
            "charge_unit": data.get("charge_unit"),
            "price_m3": data.get("price_m3"),
            "segment": data.get("segment"),
            "confidence_score": data.get("confidence", {}).get("score", 0) if data.get("confidence") else 0,
            "confidence_level": data.get("confidence", {}).get("level", "none") if data.get("confidence") else "none",
            "match_count": data.get("match_count", 0),
            "total_matches": data.get("total_matches", 0),
            "min_price": data.get("price_range", {}).get("min") if data.get("price_range") else None,
            "max_price": data.get("price_range", {}).get("max") if data.get("price_range") else None,
            "yearly_adjustment": yearly_adj,
            "median_price": data.get("median_price"),
            "avg_fy_year": yearly_adj.get("avg_data_year") if yearly_adj else None,
        }
    except requests.RequestException as e:
        return {"success": False, "error": str(e), "predicted_price": None}


# ═══════════════════════════════════════════════════════════════════
#  Comparison logic
# ═══════════════════════════════════════════════════════════════════

def compare_results(
    case_name: str,
    streamlit_result: Dict[str, Any],
    fastapi_result: Dict[str, Any],
    tolerance: float = 0.01,
) -> Dict[str, Any]:
    """Compare two prediction results and return a comparison report."""
    fields_to_compare = [
        ("success", "exact"),
        ("predicted_price", "numeric"),
        ("price_m3", "numeric"),
        ("segment", "exact"),
        ("match_count", "exact"),
        ("total_matches", "exact"),
        ("confidence_score", "numeric_loose"),  # 1.0 tolerance
        ("confidence_level", "exact"),
        ("min_price", "numeric"),
        ("max_price", "numeric"),
        ("median_price", "numeric"),
    ]

    diffs: List[Dict[str, Any]] = []
    all_pass = True

    # Skip confidence fields on no-match cases (not meaningful)
    both_no_match = (
        not streamlit_result.get("success") and not fastapi_result.get("success")
    )

    for field, compare_type in fields_to_compare:
        s_val = streamlit_result.get(field)
        f_val = fastapi_result.get(field)

        # Skip confidence comparison on no-match cases
        if both_no_match and field in ("confidence_score", "confidence_level"):
            passed = True  # Not meaningful when no matches
        elif compare_type == "exact":
            passed = s_val == f_val
        elif compare_type == "numeric":
            if s_val is None and f_val is None:
                passed = True
            elif s_val is None or f_val is None:
                passed = False
            else:
                passed = abs(float(s_val) - float(f_val)) <= tolerance
        elif compare_type == "numeric_loose":
            if s_val is None and f_val is None:
                passed = True
            elif s_val is None or f_val is None:
                passed = False
            else:
                passed = abs(float(s_val) - float(f_val)) <= 1.0
        else:
            passed = s_val == f_val

        if not passed:
            all_pass = False

        diffs.append({
            "field": field,
            "streamlit": s_val,
            "fastapi": f_val,
            "passed": passed,
        })

    return {
        "case_name": case_name,
        "all_pass": all_pass,
        "diffs": diffs,
    }


# ═══════════════════════════════════════════════════════════════════
#  Report generation
# ═══════════════════════════════════════════════════════════════════

def generate_report(comparisons: List[Dict[str, Any]], api_url: str) -> str:
    """Generate a markdown comparison report."""
    total = len(comparisons)
    passed = sum(1 for c in comparisons if c["all_pass"])
    failed = total - passed
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        f"# 🧪 Parity Test Report",
        f"",
        f"**Date:** {timestamp}",
        f"**FastAPI URL:** `{api_url}`",
        f"**Tolerance:** ±$0.01 for prices, exact for categories",
        f"",
        f"## Summary",
        f"",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Total test cases | {total} |",
        f"| ✅ Passed | {passed} |",
        f"| ❌ Failed | {failed} |",
        f"| Pass rate | {passed/total*100:.0f}% |",
        f"",
    ]

    # Detailed results
    lines.append("## Detailed Results")
    lines.append("")

    for i, comp in enumerate(comparisons, 1):
        status = "✅" if comp["all_pass"] else "❌"
        lines.append(f"### {status} Case {i}: {comp['case_name']}")
        lines.append("")
        lines.append("| Field | Streamlit | FastAPI | Match |")
        lines.append("|---|---|---|---|")

        for diff in comp["diffs"]:
            s_val = diff["streamlit"]
            f_val = diff["fastapi"]
            match = "✅" if diff["passed"] else "❌"

            # Format values
            if isinstance(s_val, float):
                s_display = f"{s_val:,.2f}"
            elif s_val is None:
                s_display = "—"
            else:
                s_display = str(s_val)

            if isinstance(f_val, float):
                f_display = f"{f_val:,.2f}"
            elif f_val is None:
                f_display = "—"
            else:
                f_display = str(f_val)

            lines.append(f"| {diff['field']} | {s_display} | {f_display} | {match} |")

        lines.append("")

    # Failure summary
    if failed > 0:
        lines.append("## ⚠️ Failures Summary")
        lines.append("")
        for i, comp in enumerate(comparisons, 1):
            if not comp["all_pass"]:
                failed_fields = [d["field"] for d in comp["diffs"] if not d["passed"]]
                lines.append(f"- **Case {i}** ({comp['case_name']}): {', '.join(failed_fields)}")
        lines.append("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Parity test: FastAPI vs Streamlit")
    parser.add_argument(
        "--api-url",
        default="http://localhost:8001",
        help="FastAPI base URL (default: http://localhost:8001)",
    )
    parser.add_argument(
        "--cases",
        nargs="*",
        type=int,
        help="Run only specific test cases (1-indexed), e.g. --cases 1 3 5",
    )
    parser.add_argument(
        "--output",
        default="parity_report.md",
        help="Output markdown file (default: parity_report.md)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="Numeric tolerance for price comparisons (default: 0.01)",
    )
    args = parser.parse_args()

    # Filter test cases if needed
    cases = TEST_CASES
    if args.cases:
        cases = [TEST_CASES[i - 1] for i in args.cases if 1 <= i <= len(TEST_CASES)]

    print(f"╔══════════════════════════════════════════════════╗")
    print(f"║  Parity Test: FastAPI vs Streamlit               ║")
    print(f"╠══════════════════════════════════════════════════╣")
    print(f"║  API URL:   {args.api_url:<38}║")
    print(f"║  Cases:     {len(cases):<38}║")
    print(f"║  Tolerance: ±${args.tolerance:<35}║")
    print(f"╚══════════════════════════════════════════════════╝")
    print()

    # Check API is reachable
    print("🔍 Checking FastAPI health...")
    try:
        health = requests.get(f"{args.api_url.rstrip('/')}/health", timeout=10).json()
        print(f"   Status: {health.get('status')} | Records: {health.get('record_count', 'N/A')}")
    except Exception as e:
        print(f"   ⚠️  Could not reach FastAPI at {args.api_url}: {e}")
        print(f"   Make sure the server is running.")
        sys.exit(1)

    print()

    comparisons: List[Dict[str, Any]] = []

    for i, case in enumerate(cases, 1):
        case_name = case["name"]
        params = case["params"]

        print(f"[{i}/{len(cases)}] {case_name}")

        # Run Streamlit prediction
        print(f"  ├─ Streamlit... ", end="", flush=True)
        t0 = time.time()
        streamlit_result = run_streamlit_prediction(params)
        t1 = time.time()
        s_price = streamlit_result.get("predicted_price")
        s_time = t1 - t0
        print(f"${s_price:,.2f} ({s_time:.1f}s)" if s_price else f"no match ({s_time:.1f}s)")

        # Run FastAPI prediction
        print(f"  ├─ FastAPI...   ", end="", flush=True)
        t0 = time.time()
        fastapi_result = run_fastapi_prediction(args.api_url, params)
        t1 = time.time()
        f_price = fastapi_result.get("predicted_price")
        f_time = t1 - t0
        if fastapi_result.get("error"):
            print(f"ERROR: {fastapi_result['error']}")
        else:
            print(f"${f_price:,.2f} ({f_time:.1f}s)" if f_price else f"no match ({f_time:.1f}s)")

        # Compare
        comp = compare_results(case_name, streamlit_result, fastapi_result, tolerance=args.tolerance)
        comparisons.append(comp)

        status = "✅ PASS" if comp["all_pass"] else "❌ FAIL"
        if not comp["all_pass"]:
            failed_fields = [d["field"] for d in comp["diffs"] if not d["passed"]]
            status += f" ({', '.join(failed_fields)})"
        print(f"  └─ {status}")
        print()

    # Generate report
    report = generate_report(comparisons, args.api_url)
    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    # Print summary
    passed = sum(1 for c in comparisons if c["all_pass"])
    failed = len(comparisons) - passed
    print(f"{'='*50}")
    print(f"  RESULTS: {passed}/{len(comparisons)} passed", end="")
    if failed > 0:
        print(f", {failed} failed ⚠️")
    else:
        print(f" ✅")
    print(f"  Report: {report_path}")
    print(f"{'='*50}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
