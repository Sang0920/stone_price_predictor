# 🧪 Parity Test Report

**Date:** 2026-03-11 03:20:46
**FastAPI URL:** `http://localhost:8001`
**Tolerance:** ±$0.01 for prices, exact for categories

## Summary

| Metric | Value |
|---|---|
| Total test cases | 15 |
| ✅ Passed | 15 |
| ❌ Failed | 0 |
| Pass rate | 100% |

## Detailed Results

### ✅ Case 1: BD 30x30x3 USD/PC (screenshot case)

| Field | Streamlit | FastAPI | Match |
|---|---|---|---|
| success | True | True | ✅ |
| predicted_price | 2.91 | 2.91 | ✅ |
| price_m3 | 1,066.67 | 1,066.67 | ✅ |
| segment | Premium | Premium | ✅ |
| match_count | 3 | 3 | ✅ |
| total_matches | 3 | 3 | ✅ |
| confidence_score | 71.10 | 71.10 | ✅ |
| confidence_level | medium | medium | ✅ |
| min_price | 2.88 | 2.88 | ✅ |
| max_price | 2.88 | 2.88 | ✅ |
| median_price | 2.88 | 2.88 | ✅ |

### ✅ Case 2: BD 9x9x9 USD/PC (cube)

| Field | Streamlit | FastAPI | Match |
|---|---|---|---|
| success | True | True | ✅ |
| predicted_price | 0.34 | 0.34 | ✅ |
| price_m3 | 466.39 | 466.39 | ✅ |
| segment | Common | Common | ✅ |
| match_count | 5 | 5 | ✅ |
| total_matches | 16 | 16 | ✅ |
| confidence_score | 74.90 | 74.90 | ✅ |
| confidence_level | medium | medium | ✅ |
| min_price | 0.34 | 0.34 | ✅ |
| max_price | 0.34 | 0.34 | ✅ |
| median_price | 0.34 | 0.34 | ✅ |

### ✅ Case 3: GX 30x30x3 USD/M2 (strict)

| Field | Streamlit | FastAPI | Match |
|---|---|---|---|
| success | False | False | ✅ |
| predicted_price | — | — | ✅ |
| price_m3 | — | — | ✅ |
| segment | — | — | ✅ |
| match_count | 0 | 0 | ✅ |
| total_matches | 0 | 0 | ✅ |
| confidence_score | — | 0 | ✅ |
| confidence_level | — | none | ✅ |
| min_price | — | — | ✅ |
| max_price | — | — | ✅ |
| median_price | — | — | ✅ |

### ✅ Case 4: BD 30x30x3 USD/PC customer D

| Field | Streamlit | FastAPI | Match |
|---|---|---|---|
| success | True | True | ✅ |
| predicted_price | 3.04 | 3.04 | ✅ |
| price_m3 | 1,066.67 | 1,066.67 | ✅ |
| segment | Premium | Premium | ✅ |
| match_count | 3 | 3 | ✅ |
| total_matches | 3 | 3 | ✅ |
| confidence_score | 71.10 | 71.10 | ✅ |
| confidence_level | medium | medium | ✅ |
| min_price | 2.97 | 2.97 | ✅ |
| max_price | 3.05 | 3.05 | ✅ |
| median_price | 2.88 | 2.88 | ✅ |

### ✅ Case 5: BD 30x30x3 USD/PC customer B

| Field | Streamlit | FastAPI | Match |
|---|---|---|---|
| success | True | True | ✅ |
| predicted_price | 2.82 | 2.82 | ✅ |
| price_m3 | 1,066.67 | 1,066.67 | ✅ |
| segment | Premium | Premium | ✅ |
| match_count | 3 | 3 | ✅ |
| total_matches | 3 | 3 | ✅ |
| confidence_score | 71.10 | 71.10 | ✅ |
| confidence_level | medium | medium | ✅ |
| min_price | 2.76 | 2.76 | ✅ |
| max_price | 2.82 | 2.82 | ✅ |
| median_price | 2.88 | 2.88 | ✅ |

### ✅ Case 6: BD 30x30x3 USD/PC (no yearly adj)

| Field | Streamlit | FastAPI | Match |
|---|---|---|---|
| success | True | True | ✅ |
| predicted_price | 2.88 | 2.88 | ✅ |
| price_m3 | 1,066.67 | 1,066.67 | ✅ |
| segment | Premium | Premium | ✅ |
| match_count | 3 | 3 | ✅ |
| total_matches | 3 | 3 | ✅ |
| confidence_score | 71.10 | 71.10 | ✅ |
| confidence_level | medium | medium | ✅ |
| min_price | 2.88 | 2.88 | ✅ |
| max_price | 2.88 | 2.88 | ✅ |
| median_price | 2.88 | 2.88 | ✅ |

### ✅ Case 7: BD 30x30x3 USD/PC (relaxed dims)

| Field | Streamlit | FastAPI | Match |
|---|---|---|---|
| success | True | True | ✅ |
| predicted_price | 3.28 | 3.28 | ✅ |
| price_m3 | 1,206.33 | 1,206.33 | ✅ |
| segment | Premium | Premium | ✅ |
| match_count | 5 | 5 | ✅ |
| total_matches | 185 | 185 | ✅ |
| confidence_score | 75.10 | 75.10 | ✅ |
| confidence_level | medium | medium | ✅ |
| min_price | 3.26 | 3.26 | ✅ |
| max_price | 3.26 | 3.26 | ✅ |
| median_price | 3.78 | 3.78 | ✅ |

### ✅ Case 8: BX 20x10x8 USD/PC (different dims)

| Field | Streamlit | FastAPI | Match |
|---|---|---|---|
| success | True | True | ✅ |
| predicted_price | 0.44 | 0.44 | ✅ |
| price_m3 | 276.25 | 276.25 | ✅ |
| segment | Economy | Economy | ✅ |
| match_count | 3 | 3 | ✅ |
| total_matches | 3 | 3 | ✅ |
| confidence_score | 81.10 | 81.10 | ✅ |
| confidence_level | high | high | ✅ |
| min_price | 0.44 | 0.44 | ✅ |
| max_price | 0.44 | 0.44 | ✅ |
| median_price | 0.44 | 0.44 | ✅ |

### ✅ Case 9: BD 30x30x3 USD/PC (recent_count=10)

| Field | Streamlit | FastAPI | Match |
|---|---|---|---|
| success | True | True | ✅ |
| predicted_price | 2.91 | 2.91 | ✅ |
| price_m3 | 1,066.67 | 1,066.67 | ✅ |
| segment | Premium | Premium | ✅ |
| match_count | 3 | 3 | ✅ |
| total_matches | 3 | 3 | ✅ |
| confidence_score | 71.10 | 71.10 | ✅ |
| confidence_level | medium | medium | ✅ |
| min_price | 2.88 | 2.88 | ✅ |
| max_price | 2.88 | 2.88 | ✅ |
| median_price | 2.88 | 2.88 | ✅ |

### ✅ Case 10: BD 30x30x3 USD/PC (exact stone)

| Field | Streamlit | FastAPI | Match |
|---|---|---|---|
| success | False | False | ✅ |
| predicted_price | — | — | ✅ |
| price_m3 | — | — | ✅ |
| segment | — | — | ✅ |
| match_count | 0 | 0 | ✅ |
| total_matches | 0 | 0 | ✅ |
| confidence_score | — | 0 | ✅ |
| confidence_level | — | none | ✅ |
| min_price | — | — | ✅ |
| max_price | — | — | ✅ |
| median_price | — | — | ✅ |

### ✅ Case 11: BD 30x30x3 USD/TON

| Field | Streamlit | FastAPI | Match |
|---|---|---|---|
| success | False | False | ✅ |
| predicted_price | — | — | ✅ |
| price_m3 | — | — | ✅ |
| segment | — | — | ✅ |
| match_count | 0 | 0 | ✅ |
| total_matches | 0 | 0 | ✅ |
| confidence_score | — | 0 | ✅ |
| confidence_level | — | none | ✅ |
| min_price | — | — | ✅ |
| max_price | — | — | ✅ |
| median_price | — | — | ✅ |

### ✅ Case 12: MV 100x100x1 USD/M3 (likely no match)

| Field | Streamlit | FastAPI | Match |
|---|---|---|---|
| success | False | False | ✅ |
| predicted_price | — | — | ✅ |
| price_m3 | — | — | ✅ |
| segment | — | — | ✅ |
| match_count | 0 | 0 | ✅ |
| total_matches | 0 | 0 | ✅ |
| confidence_score | — | 0 | ✅ |
| confidence_level | — | none | ✅ |
| min_price | — | — | ✅ |
| max_price | — | — | ✅ |
| median_price | — | — | ✅ |

### ✅ Case 13: BD 30x30x3 USD/PC customer E

| Field | Streamlit | FastAPI | Match |
|---|---|---|---|
| success | True | True | ✅ |
| predicted_price | 3.24 | 3.24 | ✅ |
| price_m3 | 1,066.67 | 1,066.67 | ✅ |
| segment | Premium | Premium | ✅ |
| match_count | 3 | 3 | ✅ |
| total_matches | 3 | 3 | ✅ |
| confidence_score | 71.10 | 71.10 | ✅ |
| confidence_level | medium | medium | ✅ |
| min_price | 3.11 | 3.11 | ✅ |
| max_price | 3.31 | 3.31 | ✅ |
| median_price | 2.88 | 2.88 | ✅ |

### ✅ Case 14: BD 30x30x3 USD/PC (yearly 2%)

| Field | Streamlit | FastAPI | Match |
|---|---|---|---|
| success | True | True | ✅ |
| predicted_price | 3.00 | 3.00 | ✅ |
| price_m3 | 1,066.67 | 1,066.67 | ✅ |
| segment | Premium | Premium | ✅ |
| match_count | 3 | 3 | ✅ |
| total_matches | 3 | 3 | ✅ |
| confidence_score | 71.10 | 71.10 | ✅ |
| confidence_level | medium | medium | ✅ |
| min_price | 2.88 | 2.88 | ✅ |
| max_price | 2.88 | 2.88 | ✅ |
| median_price | 2.88 | 2.88 | ✅ |

### ✅ Case 15: BD 30x30x3 USD/PC (all relaxed)

| Field | Streamlit | FastAPI | Match |
|---|---|---|---|
| success | True | True | ✅ |
| predicted_price | 3.28 | 3.28 | ✅ |
| price_m3 | 1,206.33 | 1,206.33 | ✅ |
| segment | Premium | Premium | ✅ |
| match_count | 5 | 5 | ✅ |
| total_matches | 277 | 277 | ✅ |
| confidence_score | 74.20 | 74.20 | ✅ |
| confidence_level | medium | medium | ✅ |
| min_price | 3.26 | 3.26 | ✅ |
| max_price | 3.26 | 3.26 | ✅ |
| median_price | 3.78 | 3.78 | ✅ |
