"""
Salesforce Integration Module for Stone Price Predictor
Provides functions to fetch real data from Salesforce CRM
"""

import os
from typing import Optional, Dict, Any, List
from simple_salesforce import Salesforce
import pandas as pd
from datetime import datetime
import json


# Main Processing Code mapping (SKU positions 5-7)
# Ký hiệu -> English name
PROCESSING_CODE_MAP = {
    'CUA': 'Sawn',
    'DOT': 'Flamed',
    'DOC': 'Flamed Brush',
    'DOX': 'Flamed Water',
    'HON': 'Honed',
    'CTA': 'Split Handmade',
    'CLO': 'Sawn then Cleaved',
    'TDE': 'Chiseled',
    'GCR': 'Vibrated Honed Tumbled',
    'GCT': 'Old Imitation',
    'MGI': 'Scraped',
    'PCA': 'Sandblasted',
    'MCA': 'Sandblasted',
    'QME': 'Tumbled',
    'TLO': 'Cleaved',
    'BON': 'Polished',
    'BAM': 'Bush Hammered',
    'CHA': 'Brush',
}


# Application Code mapping (SKU positions 3-4)
# Maps code like "1.1", "1.3" to application names
# Per "Copy of Code Rule AND Product list" and "Application Mapping" docs
APPLICATION_CODE_MAP = {
    # Format: 'X.Y' -> (English Name, Vietnamese Name)
    '1.1': ('Cubes / Cobbles', 'Cubic (Đá vuông)'),
    '1.3': ('Paving stone / Paving slab', 'Đá lát ngoài trời'),
    '2.1': ('Wall stone / Wall brick', 'Đá xây tường rào'),
    '2.2': ('Wall covering / Wall top', 'Đá ốp tường rào'),
    '2.3': ('Rockface Walling', 'Đá mặt lỗi ốp tường'),
    '3.1': ('Palisades', 'Đá cây'),
    '3.2': ('Border / Kerbs', 'Đá bó vỉa hè loại thẳng'),
    '3.3': ('Corner', 'Đá bó vỉa hè, loại góc hoặc cong'),
    '4.1': ('Stair / Step (Block)', 'Đá bậc thang nguyên khối'),
    '4.2': ('Step (Cladding)', 'Đá ốp bậc thang'),
    '5.1': ('Block', 'Đá khối'),
    '6.1': ('Pool surrounding', 'Đá ghép hồ bơi'),
    '6.2': ('Window sill', 'Đá bệ cửa sổ, gờ tường'),
    '7.2': ('Tile / Paver', 'Đá lát, cắt quy cách'),
    '8.1': ('Skirtings', 'Đá len chân tường'),
    '9.1': ('Slab', 'Đá slab kích thước khổ lớn'),
    # Keep numeric versions for backward compatibility
    '01': ('Cubes / Cobbles', 'Cubic (Đá vuông)'),
    '11': ('Cubes / Cobbles', 'Cubic (Đá vuông)'),
    '13': ('Paving stone / Paving slab', 'Đá lát ngoài trời'),
}


def extract_application_code(sku: str) -> tuple:
    """
    Extract application code from SKU.
    
    SKU format variations:
    1. New format with dot: BD5.1DOX0-... (positions 3-5 = "5.1")
    2. Old format numeric: BD01DOT2-... (positions 3-4 = "01" -> "0.1")
    
    Examples:
    - MB4.1GCT0-1000300160 -> 4.1 (Stair / Step)
    - BD5.1DOX0-0500500500 -> 5.1 (Block)
    - BD7.2DOX1-0900600030 -> 7.2 (Tile / Paver)
    - BX1.0CTA9-0100100090 -> 1.1 (Cubes) - note: 1.0 may map to 1.1
    
    Args:
        sku: The SKU/StockKeepingUnit string
        
    Returns:
        Tuple of (code, english_name, vietnamese_name)
    """
    if not sku or not isinstance(sku, str):
        return ('', 'Unknown', 'Không xác định')
    
    sku_upper = sku.upper().strip()
    
    # Strategy 1: Extract X.Y format from positions 3-5 (index 2-4)
    # e.g., "BD5.1DOX0-..." -> "5.1"
    if len(sku_upper) >= 5 and sku_upper[3] == '.':
        dotted_code = sku_upper[2:5]  # e.g., "5.1"
        if dotted_code in APPLICATION_CODE_MAP:
            eng, vn = APPLICATION_CODE_MAP[dotted_code]
            return (dotted_code, eng, vn)
        # Handle special case: X.0 might map to X.1
        if dotted_code.endswith('.0'):
            alt_code = dotted_code[0] + '.1'
            if alt_code in APPLICATION_CODE_MAP:
                eng, vn = APPLICATION_CODE_MAP[alt_code]
                return (alt_code, eng, vn)
        # Not in mapping - return code with "Other" label
        return (dotted_code, f'Other ({dotted_code})', f'Khác ({dotted_code})')
    
    # Strategy 2: Old numeric format - positions 3-4 (index 2-3)
    # e.g., "BD01DOT2-..." -> "01" -> "0.1"
    if len(sku_upper) >= 4:
        numeric_code = sku_upper[2:4]
        if numeric_code.isdigit() and len(numeric_code) == 2:
            dotted_code = f"{numeric_code[0]}.{numeric_code[1]}"
            if dotted_code in APPLICATION_CODE_MAP:
                eng, vn = APPLICATION_CODE_MAP[dotted_code]
                return (dotted_code, eng, vn)
            # Not in mapping - return code with "Other" label
            return (dotted_code, f'Other ({dotted_code})', f'Khác ({dotted_code})')
    
    # Fallback: No application code found
    return ('', 'Unknown', 'Không xác định')


def extract_processing_code(product_code: str) -> tuple:
    """
    Extract main processing code from product SKU.
    
    Tries multiple extraction strategies:
    1. Positions 5-7 (standard format like BD01DOT2-06004060)
    2. After first dash (like X-DOT-123)
    3. Search anywhere in the string
    
    Args:
        product_code: The product code/SKU string
        
    Returns:
        Tuple of (code, english_name)
    """
    if not product_code or not isinstance(product_code, str):
        return ('', 'Unknown')
    
    code_upper = product_code.upper()
    
    # Strategy 1: Standard format - positions 5-7 (index 4-6)
    if len(code_upper) >= 7:
        code = code_upper[4:7]
        if code in PROCESSING_CODE_MAP:
            return (code, PROCESSING_CODE_MAP[code])
    
    # Strategy 2: Search for any known processing code in the string
    for proc_code in PROCESSING_CODE_MAP.keys():
        if proc_code in code_upper:
            return (proc_code, PROCESSING_CODE_MAP[proc_code])
    
    # No processing code found
    return ('', 'Unknown')


class SalesforceDataLoader:
    """Load pricing data from Salesforce for price prediction."""
    
    def __init__(self, username: str = None, password: str = None, security_token: str = None):
        """
        Initialize Salesforce connection.
        
        Args:
            username: Salesforce username
            password: Salesforce password  
            security_token: Salesforce security token
        """
        # Try to get credentials from Streamlit secrets first (for Streamlit Cloud)
        # Then fall back to environment variables (for local development)
        try:
            import streamlit as st
            self.username = username or st.secrets.get("SALESFORCE_USERNAME") or os.getenv("SALESFORCE_USERNAME")
            self.password = password or st.secrets.get("SALESFORCE_PASSWORD") or os.getenv("SALESFORCE_PASSWORD")
            self.security_token = security_token or st.secrets.get("SALESFORCE_SECURITY_TOKEN") or os.getenv("SALESFORCE_SECURITY_TOKEN")
        except Exception:
            # Fallback to environment variables only
            self.username = username or os.getenv("SALESFORCE_USERNAME")
            self.password = password or os.getenv("SALESFORCE_PASSWORD")
            self.security_token = security_token or os.getenv("SALESFORCE_SECURITY_TOKEN")
        self._sf = None
        
    @property
    def sf(self) -> Salesforce:
        """Get authenticated Salesforce client."""
        if self._sf is None:
            if not all([self.username, self.password, self.security_token]):
                raise Exception("Salesforce credentials not configured")
            self._sf = Salesforce(
                username=self.username,
                password=self.password,
                security_token=self.security_token
            )
        return self._sf
    
    def get_pricebook_entries(self, pricebook_name: Optional[str] = None) -> pd.DataFrame:
        """
        Get PricebookEntry records with Product2 details.
        
        Args:
            pricebook_name: Optional filter by pricebook name
            
        Returns:
            DataFrame with pricing data
        """
        query = """
        SELECT
            Id,
            Name,
            UnitPrice,
            Charge_Unit__c,
            ProductCode,
            Product2Id,
            Product2.Name,
            Product2.ProductCode,
            Product2.StockKeepingUnit,
            Product2.Description,
            Product2.Family,
            Product2.STONE_Class__c,
            Product2.STONE_Color_Type__c,
            Product2.Long__c,
            Product2.Width__c,
            Product2.High__c,
            Product2.Packing__c,
            Product2.specific_gravity__c,
            Product2.Bottom_cladding_coefficient__c,
            Pricebook2Id,
            Pricebook2.Name,
            IsActive,
            CreatedDate,
            LastModifiedDate
        FROM PricebookEntry
        WHERE IsActive = true
        """
        
        if pricebook_name:
            safe_name = pricebook_name.replace("'", "\\'")
            query += f" AND Pricebook2.Name LIKE '%{safe_name}%'"
        
        query += " ORDER BY Product2.Name ASC LIMIT 2000"
        
        result = self.sf.query_all(query)
        records = result.get("records", [])
        
        # Flatten nested Product2 data
        data = []
        for r in records:
            product = r.get("Product2") or {}
            pricebook = r.get("Pricebook2") or {}
            
            # Calculate volume and area
            length = product.get("Long__c") or 0
            width = product.get("Width__c") or 0
            height = product.get("High__c") or 0
            volume_m3 = (length * width * height) / 1000000
            area_m2 = (length * width) / 10000
            
            # Calculate price per m3
            unit_price = r.get("UnitPrice") or 0
            charge_unit = r.get("Charge_Unit__c") or "USD/M3"
            specific_gravity = product.get("specific_gravity__c") or 2.8
            
            if charge_unit == "USD/M2" and height > 0:
                price_m3 = unit_price * 100 / height
            elif charge_unit == "USD/PC" and volume_m3 > 0:
                price_m3 = unit_price / volume_m3
            elif charge_unit == "USD/TON":
                coeff = product.get("Bottom_cladding_coefficient__c") or 1.1
                price_m3 = unit_price * specific_gravity * coeff
            elif charge_unit == "USD/ML" and width > 0 and height > 0:
                price_m3 = unit_price * 10000 / (width * height)
            else:
                price_m3 = unit_price
            
            # Classify segment
            if price_m3 >= 1500:
                segment = "Super premium"
            elif price_m3 >= 800:
                segment = "Premium"
            elif price_m3 >= 400:
                segment = "Common"
            else:
                segment = "Economy"
            
            data.append({
                "pricebook_entry_id": r.get("Id"),
                "product_id": r.get("Product2Id"),
                "product_name": product.get("Name"),
                "product_code": product.get("ProductCode"),
                "sku": product.get("StockKeepingUnit"),
                "description": product.get("Description"),
                "family": product.get("Family"),
                "stone_class": product.get("STONE_Class__c"),
                "stone_color_type": product.get("STONE_Color_Type__c"),
                "length_cm": length,
                "width_cm": width,
                "height_cm": height,
                "packing": product.get("Packing__c"),
                "specific_gravity": specific_gravity,
                "volume_m3": volume_m3,
                "area_m2": area_m2,
                "charge_unit": charge_unit,
                "list_price": unit_price,
                "price_m3": round(price_m3, 2),
                "segment": segment,
                "pricebook_id": r.get("Pricebook2Id"),
                "pricebook_name": pricebook.get("Name"),
                "created_date": r.get("CreatedDate"),
                "last_modified": r.get("LastModifiedDate")
            })
        
        return pd.DataFrame(data)
    
    def get_contract_products(self, account_code: Optional[str] = None) -> pd.DataFrame:
        """
        Get Contract_Product__c records for historical pricing analysis.
        Uses the query pattern from contract_query.txt.
        
        Args:
            account_code: Optional filter by account code
            
        Returns:
            DataFrame with contract product data
        """
        # Query based on contract_query.txt pattern
        query = """
        SELECT 
            Name,
            Contract__r.Name,
            Contract__r.Account__r.Account_Code__c,
            Contract__r.Account__r.Nhom_Khu_vuc_KH__c,
            Contract__r.Account__r.BillingAddress,
            Product__r.STONE_Color_Type__c,
            Product__r.StockKeepingUnit,
            Product__r.Family,
            Segment__c,
            Created_Date__c,
            Delivery_Date__c,
            Product_Discription__c,
            Product__r.Product_description_in_Vietnamese__c,
            Length__c,
            Width__c,
            Height__c,
            Quantity__c,
            Crates__c,
            m2__c,
            m3__c,
            ml__c,
            Tons__c,
            Sales_Price__c,
            Charge_Unit__c,
            Total_Price_USD__c
        FROM Contract_Product__c
        """
        
        conditions = []
        if account_code:
            safe_code = account_code.replace("'", "\\'")
            conditions.append(f"Contract__r.Account__r.Account_Code__c = '{safe_code}'")
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY Created_Date__c DESC NULLS LAST"
        
        result = self.sf.query_all(query)
        records = result.get("records", [])
        
        data = []
        for r in records:
            product = r.get("Product__r") or {}
            contract = r.get("Contract__r") or {}
            # Get Account from nested Contract__r.Account__r
            account = contract.get("Account__r") or {}
            # Extract billing country from BillingAddress compound field
            billing_address = account.get("BillingAddress") or {}
            
            # Use dimensions from contract product
            length = r.get("Length__c") or 0
            width = r.get("Width__c") or 0
            height = r.get("Height__c") or 0
            
            volume_m3 = (length * width * height) / 1000000
            area_m2 = (length * width) / 10000
            
            # Calculate price per m3 from Total_Price_USD__c and m3__c
            total_price = r.get("Total_Price_USD__c") or 0
            m3_value = r.get("m3__c") or 0
            price_m3 = total_price / m3_value if m3_value > 0 else 0
            
            segment = r.get("Segment__c") or "Unknown"
            
            # Get date: use Created_Date__c, fallback to Delivery_Date__c
            created_date = r.get("Created_Date__c")
            if not created_date:
                created_date = r.get("Delivery_Date__c")
            
            # Extract fiscal year from date
            fy_year = None
            if created_date:
                try:
                    dt = datetime.fromisoformat(created_date.replace('Z', '+00:00'))
                    fy_year = dt.year
                except:
                    pass
            
            data.append({
                "contract_product_name": r.get("Name"),
                "contract_name": contract.get("Name"),
                "account_code": account.get("Account_Code__c"),  # From Contract__r.Account__r.Account_Code__c
                "customer_regional_group": account.get("Nhom_Khu_vuc_KH__c"),  # Customer Regional Group
                "billing_country": billing_address.get("country") if billing_address else None,  # Billing Country
                "stone_color_type": product.get("STONE_Color_Type__c"),
                "sku": product.get("StockKeepingUnit"),  # SKU like BD01DOT2-06004060
                "family": product.get("Family"),
                "segment": segment,
                "created_date": created_date,
                "fy_year": fy_year,
                "product_description": r.get("Product_Discription__c"),
                "product_description_vn": product.get("Product_description_in_Vietnamese__c"),
                "length_cm": length,
                "width_cm": width,
                "height_cm": height,
                "quantity": r.get("Quantity__c"),
                "crates": r.get("Crates__c"),
                "m2": r.get("m2__c"),
                "m3": m3_value,
                "ml": r.get("ml__c"),
                "tons": r.get("Tons__c"),
                "sales_price": r.get("Sales_Price__c"),
                "charge_unit": r.get("Charge_Unit__c"),
                "total_price_usd": total_price,
                "price_m3": price_m3,
                "volume_m3": volume_m3,
                "area_m2": area_m2,
                # Main processing code extracted from SKU (positions 5-7)
                "processing_code": extract_processing_code(product.get("StockKeepingUnit"))[0],
                "processing_name": extract_processing_code(product.get("StockKeepingUnit"))[1],
                # Application code extracted from SKU (positions 3-4)
                "application_code": extract_application_code(product.get("StockKeepingUnit"))[0],
                "application": extract_application_code(product.get("StockKeepingUnit"))[1],
                "application_vn": extract_application_code(product.get("StockKeepingUnit"))[2],
            })
        
        return pd.DataFrame(data)
    
    def get_products_catalog(
        self,
        stone_class: Optional[str] = None,
        family: Optional[str] = None,
        only_active: bool = True
    ) -> pd.DataFrame:
        """
        Get Product2 catalog for reference.
        
        Args:
            stone_class: Filter by stone class
            family: Filter by product family
            only_active: Only return active products
            
        Returns:
            DataFrame with product catalog
        """
        where_clauses = []
        if only_active:
            where_clauses.append("IsActive = true")
        if stone_class:
            where_clauses.append(f"STONE_Class__c = '{stone_class}'")
        if family:
            where_clauses.append(f"Family = '{family}'")
        
        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)
        
        query = f"""
        SELECT
            Id,
            Name,
            ProductCode,
            StockKeepingUnit,
            Description,
            Family,
            STONE_Class__c,
            STONE_Color_Type__c,
            Long__c,
            Width__c,
            High__c,
            Packing__c,
            specific_gravity__c,
            Bottom_cladding_coefficient__c,
            Charge_Unit__c,
            List_Price__c,
            IsActive,
            CreatedDate
        FROM Product2
        {where_sql}
        ORDER BY Name ASC
        LIMIT 2000
        """
        
        result = self.sf.query_all(query)
        records = result.get("records", [])
        
        data = []
        for r in records:
            length = r.get("Long__c") or 0
            width = r.get("Width__c") or 0
            height = r.get("High__c") or 0
            
            data.append({
                "product_id": r.get("Id"),
                "product_name": r.get("Name"),
                "product_code": r.get("ProductCode"),
                "sku": r.get("StockKeepingUnit"),
                "description": r.get("Description"),
                "family": r.get("Family"),
                "stone_class": r.get("STONE_Class__c"),
                "stone_color_type": r.get("STONE_Color_Type__c"),
                "length_cm": length,
                "width_cm": width,
                "height_cm": height,
                "packing": r.get("Packing__c"),
                "specific_gravity": r.get("specific_gravity__c"),
                "charge_unit": r.get("Charge_Unit__c"),
                "list_price": r.get("List_Price__c"),
                "is_active": r.get("IsActive"),
                "created_date": r.get("CreatedDate")
            })
        
        return pd.DataFrame(data)
    
    def get_combined_pricing_data(self) -> pd.DataFrame:
        """
        Get combined pricing data from multiple sources for ML training.
        
        Returns:
            DataFrame combining PricebookEntry and Contract_Product__c data
        """
        # Get pricebook entries
        pricebook_df = self.get_pricebook_entries()
        pricebook_df["data_source"] = "pricebook"
        
        # Get contract products
        contract_df = self.get_contract_products()
        contract_df["data_source"] = "contract"
        
        # Standardize column names
        pricebook_cols = [
            "product_id", "product_name", "family", "stone_class", "stone_color_type",
            "length_cm", "width_cm", "height_cm", "volume_m3", "area_m2",
            "charge_unit", "list_price", "price_m3", "segment", "data_source"
        ]
        
        contract_cols = [
            "product_id", "product_name", "family", "stone_class", "stone_color_type", 
            "length_cm", "width_cm", "height_cm", "volume_m3", "area_m2",
            "charge_unit", "sales_price", "price_m3", "segment", "data_source"
        ]
        
        # Rename sales_price to list_price for consistency
        contract_df = contract_df.rename(columns={"sales_price": "list_price"})
        
        # Select common columns
        common_cols = list(set(pricebook_cols) & set(contract_df.columns))
        
        # Combine datasets
        combined = pd.concat([
            pricebook_df[common_cols],
            contract_df[common_cols]
        ], ignore_index=True)
        
        # Remove duplicates and invalid data
        combined = combined.dropna(subset=["price_m3"])
        combined = combined[combined["price_m3"] > 0]
        combined = combined.drop_duplicates(
            subset=["product_id", "length_cm", "width_cm", "height_cm", "data_source"]
        )
        
        return combined


def fetch_salesforce_data_for_prediction():
    """
    Convenience function to fetch all necessary data for price prediction.
    
    Returns:
        Dict with DataFrames for different data types
    """
    try:
        loader = SalesforceDataLoader()
        
        return {
            "success": True,
            "pricebook_entries": loader.get_pricebook_entries(),
            "contract_products": loader.get_contract_products(),
            "products_catalog": loader.get_products_catalog(),
            "combined_data": loader.get_combined_pricing_data()
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


# For testing
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    loader = SalesforceDataLoader()
    
    print("Fetching Pricebook Entries...")
    pricebook_df = loader.get_pricebook_entries()
    print(f"Got {len(pricebook_df)} pricebook entries")
    print(pricebook_df.head())
    
    print("\nFetching Contract Products...")
    contract_df = loader.get_contract_products()
    print(f"Got {len(contract_df)} contract products")
    print(contract_df.head())
