# 💎 Stone Price Predictor

Ứng dụng web dự đoán giá sản phẩm đá tự nhiên sử dụng Machine Learning và dữ liệu từ Salesforce.

## 🌟 Tính năng

### 1. Dự đoán giá sản phẩm (Tab 1)
- Nhập thông tin sản phẩm (loại đá, kích thước, màu sắc, đơn vị tính giá)
- Nhận dự đoán giá `sales_price` trực tiếp theo đơn vị tính (USD/PC, USD/M2, USD/M3, etc.)
- Tự động phân loại phân khúc từ giá dự đoán
- Tính giá theo từng loại khách hàng (A, B, C, D, E, F)
- **Hiển thị sản phẩm khớp chính xác** trong hệ thống với thống kê giá

### 2. Phân tích dữ liệu (Tab 2)
- Lọc sản phẩm có giá hợp lệ (> 0, không null)
- Biểu đồ phân bố theo phân khúc
- So sánh giá trung bình theo loại sản phẩm (Family)
- Phân tích giá theo màu đá (Box plot)
- Scatter plot: Giá vs Thể tích

### 3. Tìm sản phẩm tương tự (Tab 3)
- Tìm **sản phẩm khớp chính xác** với tiêu chí
- Checkbox "Hiển thị sản phẩm liên quan" với slider số lượng
- Thống kê giá (min, max, trung bình, trung vị)
- Sản phẩm liên quan được sắp xếp theo độ tương tự kích thước

### 4. Hiệu suất Model ML (Tab 4)
- Metrics: MAE, R² Score, CV MAE Mean, CV R² Mean
- Biểu đồ Feature Importance
- Thông tin hyperparameters và data cleaning

### 5. Dữ liệu chi tiết (Tab 5)
- Bảng dữ liệu đầy đủ từ Salesforce
- Bộ lọc theo Family, Segment, Khoảng giá
- Hiển thị tất cả các trường từ Contract_Product__c

## 📊 Phân khúc giá

| Phân khúc | Giá (USD/m³) | Mô tả |
|-----------|--------------|-------|
| Super Premium | ≥ 1,500 | Đá mỹ nghệ, gia công đặc biệt, quy cách riêng |
| Premium | ≥ 800 | Đá bậc thang, đá cây xử lý nhiều mặt |
| Common | ≥ 400 | Đá lát thông dụng, đá 1 mặt đốt |
| Economy | < 400 | Đá cubic gõ tay, đá tấm gõ tay dày 6cm+ |

## 👥 Phân loại khách hàng

| Loại | Mô tả | Điều chỉnh giá |
|------|-------|----------------|
| A | Khách thân thiết đặc biệt (>10 năm, 50-150 cont) | Bớt 1.5-3% so với B |
| B | Khách lớn, chuyên nghiệp (3-10 năm, 20-50 cont) | Thấp hơn C: 10-30 USD/m³ |
| C | Khách hàng phổ thông (1-5 năm, 5-20 cont) | Giá chuẩn |
| D | Khách mới, size nhỏ (1 năm, 1-10 cont) | Cao hơn C: 15-45 USD/m³ |
| E | Sản phẩm mới, cao cấp | Giá riêng |
| F | Khách hàng dự án | Tùy dự án |

## 🚀 Cài đặt

### Yêu cầu
- Python 3.9+
- Salesforce credentials

### Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### Cấu hình Salesforce

Tạo file `.env`:

```env
SALESFORCE_USERNAME=your_username@company.com
SALESFORCE_PASSWORD=your_password
SALESFORCE_SECURITY_TOKEN=your_security_token
```

### Chạy ứng dụng

```bash
streamlit run app.py
```

Truy cập: http://localhost:8501

## 📁 Cấu trúc project

```
stone_price_predictor/
├── app.py                  # Main Streamlit application
├── salesforce_loader.py    # Salesforce data integration
├── contract_query.txt      # SOQL query template
├── requirements.txt        # Python dependencies
├── README.md              # Documentation
└── .env                   # Environment variables (create this)
```

## 🔧 Dữ liệu Salesforce

### Object: Contract_Product__c

Các trường được sử dụng từ query:
- `Name`, `Contract__r.Name`, `Account_Code_C__c`
- `Product__r.STONE_Color_Type__c`, `Product__r.ProductCode`, `Product__r.Family`
- `Segment__c`, `Created_Date__c`
- `Length__c`, `Width__c`, `Height__c`
- `Quantity__c`, `Crates__c`, `m2__c`, `m3__c`, `ml__c`, `Tons__c`
- `Sales_Price__c`, `Charge_Unit__c`, `Total_Price_USD__c`

### Calculated Fields
- `price_m3` = Total_Price_USD / m3 (nếu m3 > 0)
- `volume_m3` = length × width × height / 1,000,000
- `area_m2` = length × width / 10,000
- `fy_year` = Năm tài chính từ Created_Date

## 📈 Machine Learning Model

### Target: `sales_price`
Model dự đoán giá bán trực tiếp theo đơn vị tính giá (charge_unit), không chuyển đổi sang USD/m³.

### Features

**Categorical:**
- `family` - Loại sản phẩm (STAIR, TILES, SLAB, etc.)
- `stone_color_type` - Màu đá (ABSOLUTE BASALT, BLACK BASALT, etc.)
- `charge_unit` - Đơn vị tính giá (USD/PC, USD/M2, USD/M3, USD/TON, USD/ML)

**Numerical:**
- `length_cm`, `width_cm`, `height_cm` - Kích thước
- `volume_m3`, `area_m2` - Thể tích và diện tích

> ⚠️ **Note:** `segment` đã được loại bỏ khởi features để tránh data leakage (segment được tính từ giá).

### Model: Gradient Boosting Regressor (Optimized)

```python
model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    min_samples_split=10,
    min_samples_leaf=5,
    subsample=0.8,
    max_features='sqrt',
    n_iter_no_change=10,  # Early stopping
    validation_fraction=0.1
)
```

### Data Cleaning (trước khi train)
- Loại bỏ giá = 0, âm, hoặc missing
- Loại bỏ outliers ngoài 1st-99th percentile
- Loại bỏ rows có missing values trong features

### Model Metrics (típ)
- **R² Score**: ~0.85-0.90 (test set)
- **CV R² Mean**: ~0.80-0.85 (cross-validation)
- **MAE**: ~$4-6 USD

## 📞 Hỗ trợ

- Tạo issue trên GitHub repository
- Liên hệ qua email

## 📄 License

MIT License - Free to use and modify.

---

Made with ❤️ for APlus Mineral Material Corporation
