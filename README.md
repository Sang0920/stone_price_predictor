# 💎 Stone Price Predictor

**English** | [Tiếng Việt](#-tiếng-việt)

A web application for estimating natural stone product prices using **similarity-based search** with Salesforce data.

## 🌟 Features

### 1. Price Estimation (Tab 1)
- Enter product info (stone type, dimensions, color, charge unit)
- **Similarity search** with adjustable priority levels
- Customer-type pricing (A-F) with decision authority ranges
- Display TLR, HS, and estimated weight
- Collapsible sections: Pricing rules, Customer classification, Formulas, Search criteria

### 2. Data Analysis (Tab 2)
- Filter products with valid prices (> 0, not null)
- Distribution charts by segment
- Average price comparison by product family
- Price by stone color (Box plot)
- Scatter plot: Price vs Volume

### 3. Similar Products Search (Tab 3)
- Find products matching criteria with filters
- "Show related products" checkbox with quantity slider
- Price statistics (min, max, average, median)

### 4. Reference Tables (Tab 4)
- **TLR table** (Specific Weight) by stone type
- **HS table** (Coating Factor) by product dimensions
- **Calculation formulas** for m³, m², Tons, price conversion
- **Container weight standards** by market

### 5. Detailed Data (Tab 5)
- Full data table from Salesforce
- Filters by Family, Segment, Region, Price range

## 🎯 Search Priority Criteria

| Criteria | Priority 1 | Priority 2 | Priority 3 |
|----------|-----------|-----------|-----------|
| **Stone Type** | Exact color | Same family | All types |
| **Processing** | Exact match | All types | - |
| **Height (cm)** | ±0 | ±1 | ±2 |
| **Width (cm)** | ±0 | ±5 | ±10 |
| **Length (cm)** | ±0 | ±10 | ±20 |
| **Region** | Exact region | All regions | - |

## 📊 Price Segments

| Segment | Price (USD/m³) | Products |
|---------|----------------|----------|
| 🟣 Super Premium | ≥ 1,500 | Thin paving 1-1.5cm, wall covering, decorative |
| 🔴 Premium | ≥ 800 | Interior/exterior tiles 2-5cm, slabs, stairs |
| 🟡 Common | ≥ 400 | Palisades, flamed cubes, tumbled |
| 🟢 Economy | < 400 | Hand-split cubes, natural split pavers |

## 👥 Customer Classification (A-F)

| Type | Description | Price Adjustment |
|------|-------------|------------------|
| A | Special loyal (>10 years, 50-150 containers) | -1.5% to -3% |
| B | Large professional (3-10 years, 20-50 containers) | -2% to -4% |
| C | Standard (1-5 years, 5-20 containers) | Base price |
| D | New, small (1 year, 1-10 containers) | +3% to +6% |
| E | New/premium products | ×1.08-1.15 |
| F | Project customers | ×1.08-1.15 |

## ⚖️ TLR & HS Reference

| Stone Type | TLR (tons/m³) |
|------------|---------------|
| Absolute Basalt (Dak Nong) | 2.95 |
| Black Basalt (sawn) | 2.70 |
| Black Basalt (hand-split) | 2.65 |
| Dark Grey Granite | 2.90 |
| Granite / Bluestone | 2.70 |

| Product | HS Factor |
|---------|-----------|
| Cube 5×5×5 | 1.00 |
| Cube 8×8×8 | 0.95 |
| Cube 10×10×8 | 0.875 |
| Flamed tile 6cm | 0.97 |
| Sawn palisade | 1.05 |

## 🚀 Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Configure .env file
SALESFORCE_USERNAME=your_username@company.com
SALESFORCE_PASSWORD=your_password
SALESFORCE_SECURITY_TOKEN=your_security_token

# Run application
streamlit run app.py
```

## � Calculation Formulas

```
m³ = (Length × Width × Height) / 1,000,000 × Quantity
m² = (Length × Width) / 10,000 × Quantity
Tons = m³ × TLR × HS

Price/m² = Price/m³ × Height(m)
Price/m³ = Price/Ton × TLR × HS
```

---

# 💎 Tiếng Việt

Ứng dụng web ước tính giá sản phẩm đá tự nhiên sử dụng **tìm kiếm tương tự** và dữ liệu từ Salesforce.

## 🌟 Tính năng

### 1. Ước tính giá (Tab 1)
- Nhập thông tin sản phẩm (loại đá, kích thước, màu sắc, đơn vị tính giá)
- **Tìm kiếm tương tự** với mức độ ưu tiên có thể điều chỉnh
- Tính giá theo loại khách hàng (A-F) với quyền tự quyết
- Hiển thị TLR, HS, và trọng lượng ước tính

### 2. Phân tích dữ liệu (Tab 2)
- Biểu đồ phân bố theo phân khúc
- So sánh giá trung bình theo loại sản phẩm

### 3. Tìm sản phẩm tương tự (Tab 3)
- Tìm sản phẩm khớp tiêu chí với các bộ lọc

### 4. Bảng tra cứu (Tab 4)
- Bảng TLR (Trọng Lượng Riêng)
- Bảng HS (Hệ Số Ốp Đáy)
- Công thức tính toán
- Quy chuẩn container

### 5. Dữ liệu chi tiết (Tab 5)
- Bảng dữ liệu đầy đủ từ Salesforce

## 🎯 Tiêu chí tìm kiếm

| Tiêu chí | Ưu tiên 1 | Ưu tiên 2 | Ưu tiên 3 |
|----------|-----------|-----------|-----------|
| **Loại đá** | Đúng màu đá | Cùng chủng loại | Tất cả loại đá |
| **Gia công** | Đúng loại | Tất cả | - |
| **Cao (cm)** | ±0 | ±1 | ±2 |
| **Rộng (cm)** | ±0 | ±5 | ±10 |
| **Dài (cm)** | ±0 | ±10 | ±20 |
| **Khu vực** | Đúng khu vực | Tất cả | - |

## 📊 Phân khúc giá

| Phân khúc | Giá (USD/m³) | Sản phẩm |
|-----------|--------------|----------|
| 🟣 Super Premium | ≥ 1,500 | Đá mỏng 1-1.5cm, nắp tường, mỹ nghệ |
| 🔴 Premium | ≥ 800 | Đá lát 2-5cm, slab, bậc thang |
| 🟡 Common | ≥ 400 | Đá cây, cubic đốt, quay mẻ |
| 🟢 Economy | < 400 | Đá gõ tay, cubic chẻ tay |

## 👥 Phân loại khách hàng

| Loại | Mô tả | Điều chỉnh giá |
|------|-------|----------------|
| A | Thân thiết đặc biệt (>10 năm) | -1.5% đến -3% |
| B | Lớn, chuyên nghiệp (3-10 năm) | -2% đến -4% |
| C | Phổ thông (1-5 năm) | Giá chuẩn |
| D | Mới, nhỏ (1 năm) | +3% đến +6% |
| E | Sản phẩm mới | ×1.08-1.15 |
| F | Dự án | ×1.08-1.15 |

## ⚖️ TLR & HS

| Loại đá | TLR (tấn/m³) |
|---------|--------------|
| Đá đen Đak Nông | 2.95 |
| Đá Phước Hòa (cưa) | 2.70 |
| Đá Phước Hòa (chẻ tay) | 2.65 |
| Dark Grey Granite | 2.90 |
| Granite / Bluestone | 2.70 |

## 📐 Công thức

```
m³ = (Dài × Rộng × Cao) / 1.000.000 × Số viên
Tấn = m³ × TLR × HS
Giá/m² = Giá/m³ × Cao(m)
```

## 🚢 Quy chuẩn Container

| Thị trường | Trọng lượng (tấn) |
|------------|-------------------|
| Mỹ | 20-21 |
| Châu Âu | 27-28 |
| Úc | 24-26 |
| Nhật | 27.5-28 |

---

Made with ❤️ for APlus Mineral Material Corporation
