# Stone Price Predictor

A sophisticated Streamlit-based application for predicting and analyzing natural stone product pricing, leveraging historical Salesforce contract data and intelligent similarity matching algorithms.

## Overview

Stone Price Predictor helps sales teams and pricing analysts estimate pricing for stone products based on historical contract data. The application connects directly to Salesforce to fetch contract product records and provides intelligent price predictions using multi-criteria similarity matching.

## Features

### 🔮 Price Prediction (Tab 1)
- **Multi-criteria matching** with configurable priority levels for stone color, processing type, dimensions, and regional group
- **Application-based filtering** with multi-select support (filter by product applications like Cubes, Tiles, Palisades, etc.)
- **Automatic escalation** through priority levels when exact matches aren't found
- **Customer type adjustments** with segment-aware pricing (A, B, C, D customer classifications)
- **Confidence indicators** based on match quality and sample size

### 📊 Data Analysis (Tab 2)
- **Distribution charts** by segment, family, and stone color
- **Price trends** over fiscal years
- **Application & Processing analysis** with average prices
- **Regional group comparison**
- **Correlation matrix** for dimensional and pricing factors

### 🔍 Similar Product Search (Tab 3)
- Exact and fuzzy matching by dimensions
- Filter by family, stone color, processing, and regional group
- Detailed product comparison with pricing statistics

### 📐 Lookup Tables (Tab 4)
- TLR (Tile Loss Rate) reference
- HS Factor calculations
- Price conversion formulas

### 📋 Detailed Data View (Tab 5)
- Full data exploration with filtering
- Export to CSV functionality

## Architecture

```
stone_price_predictor/
├── app.py                    # Main Streamlit application
├── salesforce_loader.py      # Salesforce API integration & data extraction
├── contract_query.txt        # SOQL query template
├── requirements.txt          # Python dependencies
├── .env                      # Environment configuration (not in repo)
└── docs/                     # Documentation and reference files
    ├── Application Mapping.pdf
    ├── Code Rule AND Product list.pdf
    └── stone_price_data.csv
```

## Installation

### Prerequisites
- Python 3.9+
- Salesforce credentials with API access

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd stone_price_predictor
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your Salesforce credentials
```

Required environment variables:
```
SALESFORCE_USERNAME=your_username
SALESFORCE_PASSWORD=your_password
SALESFORCE_SECURITY_TOKEN=your_token
SALESFORCE_DOMAIN=login  # or 'test' for sandbox
```

## Usage

### Running Locally
```bash
streamlit run app.py
```

The application will open at `http://localhost:8501`

### Basic Workflow
1. Click **"🔄 Tải / Làm mới dữ liệu từ Salesforce"** to load data
2. Configure search criteria (Application, Stone Color, Processing, Dimensions)
3. Set priority levels for each matching criterion
4. Click **"🔍 Dự đoán giá"** to get price estimates

## Data Model

### SKU Structure
The application extracts key information from product SKUs:
- **Positions 1-2**: Brand/Model prefix
- **Positions 3-5**: Application code (e.g., `5.1` for Block, `4.1` for Stair)
- **Positions 6-8**: Processing code (e.g., `DOT` for Flamed, `HON` for Honed)

### Application Codes
| Code | English | Vietnamese |
|------|---------|------------|
| 1.1 | Cubes / Cobbles | Cubic (Đá vuông) |
| 1.3 | Paving stone / Paving slab | Đá lát ngoài trời |
| 3.1 | Palisades | Đá cây |
| 4.1 | Stair / Step (Block) | Đá bậc thang nguyên khối |
| 4.2 | Step (Cladding) | Đá bao/bọc bậc cầu thang |
| 5.1 | Block | Khối |
| ... | ... | ... |

### Processing Codes
| Code | English | Vietnamese |
|------|---------|------------|
| DOT | Flamed | Đốt |
| HON | Honed | Hon/Mài Mịn |
| CTA | Split Handmade | Chẻ Tay |
| DOX | Flamed Water | Đốt Xịt Nước |
| ... | ... | ... |

## Price Segments

| Segment | Price Range (USD/m³) |
|---------|---------------------|
| Economy | < $400 |
| Common | $400 - $800 |
| Premium | $800 - $1,500 |
| Super Premium | > $1,500 |

## Priority Matching System

The application uses a hierarchical priority system for finding matching products:

| Criterion | Priority 1 | Priority 2 | Priority 3 |
|-----------|-----------|-----------|-----------|
| Stone Color | Exact match | Same family | All types |
| Processing | Exact code | All types | - |
| Dimensions | Exact ±1cm | ±20% tolerance | ±100% tolerance |
| Regional Group | Exact match | All regions | - |

## API Integration

The application connects to Salesforce using the `simple_salesforce` library and queries the `Contract_Product__c` object with related records from:
- `Contract__c`
- `Account`
- `Product2`

## Contributing

1. Create a feature branch
2. Make changes with appropriate tests
3. Submit a pull request

## License

Proprietary - Internal use only.

## Support

For issues or feature requests, contact the development team.
