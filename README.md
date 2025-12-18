# 💎 Stone Price Predictor

Ứng dụng web dự đoán giá sản phẩm đá tự nhiên sử dụng Machine Learning và dữ liệu từ Salesforce.

## 🌟 Tính năng

### 1. Dự đoán giá sản phẩm
- Nhập thông tin sản phẩm (loại đá, kích thước, màu sắc)
- Nhận dự đoán giá theo USD/m³
- Tự động phân loại phân khúc (Economy, Common, Premium, Super Premium)
- Tính giá theo từng loại khách hàng (A, B, C, D, E, F)

### 2. Phân tích dữ liệu
- Biểu đồ phân bố giá theo phân khúc
- So sánh giá theo loại sản phẩm
- Phân tích giá theo loại đá
- Scatter plot giá vs kích thước

### 3. Tìm sản phẩm tương tự
- Tìm sản phẩm có kích thước và đặc điểm tương tự
- So sánh giá giữa các sản phẩm
- Tham khảo giá thị trường

### 4. Machine Learning Model
- Gradient Boosting Regressor
- Cross-validation
- Feature importance analysis
- Model metrics (MAE, R², CV scores)

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
- Salesforce credentials (cho dữ liệu thực)

### Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### Cấu hình Salesforce (tùy chọn)

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
├── requirements.txt        # Python dependencies
├── README.md              # Documentation
└── .env                   # Environment variables (create this)
```

## 🔧 Tích hợp Salesforce

### Các object được sử dụng:

1. **PricebookEntry** - Bảng giá sản phẩm
   - UnitPrice, Charge_Unit__c
   - Liên kết với Product2 và Pricebook2

2. **Contract_Product__c** - Sản phẩm trong hợp đồng
   - Sales_Price__c, Price_m3__c, Segment__c
   - Lịch sử giao dịch thực tế

3. **Product2** - Danh mục sản phẩm
   - Long__c, Width__c, High__c (kích thước)
   - STONE_Class__c, STONE_Color_Type__c, Family

### SOQL Queries mẫu:

```sql
-- Lấy giá từ Pricebook
SELECT Id, UnitPrice, Charge_Unit__c, 
       Product2.Name, Product2.Family,
       Product2.Long__c, Product2.Width__c, Product2.High__c
FROM PricebookEntry
WHERE IsActive = true

-- Lấy giá từ Contract
SELECT Id, Sales_Price__c, Price_m3__c, Segment__c,
       Product__r.Name, Contract__r.Account__r.Account_Code__c
FROM Contract_Product__c
WHERE Contract__r.Status__c = 'Active'
```

## 📈 Machine Learning

### Features sử dụng:

**Categorical:**
- `family` - Loại sản phẩm
- `stone_class` - Loại đá (BASALT, GRANITE, BLUE STONE)
- `stone_color_type` - Màu đá
- `charge_unit` - Đơn vị tính giá

**Numerical:**
- `length_cm` - Chiều dài
- `width_cm` - Chiều rộng
- `height_cm` - Chiều cao/dày
- `volume_m3` - Thể tích
- `area_m2` - Diện tích bề mặt

### Model: Gradient Boosting Regressor

```python
model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2
)
```

## 🎯 Công thức chuyển đổi giá

```python
# USD/M2 -> USD/M3
price_m3 = price_m2 * 100 / height_cm

# USD/PC -> USD/M3  
price_m3 = price_pc / volume_m3

# USD/TON -> USD/M3
price_m3 = price_ton * specific_gravity * coefficient

# USD/ML -> USD/M3
price_m3 = price_ml * 10000 / (width_cm * height_cm)
```

## 📞 Hỗ trợ

- Tạo issue trên GitHub repository
- Liên hệ qua email

## 📄 License

MIT License - Free to use and modify.

---

Made with ❤️ for APlus Mineral Material Corporation
