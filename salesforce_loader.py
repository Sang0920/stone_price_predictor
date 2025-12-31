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
            Account_Code_C__c,
            Product__r.STONE_Color_Type__c,
            Product__r.ProductCode,
            Product__r.Family,
            Segment__c,
            Created_Date__c,
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
            conditions.append(f"Account_Code_C__c = '{safe_code}'")
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY Created_Date__c DESC"
        
        result = self.sf.query_all(query)
        records = result.get("records", [])
        
        data = []
        for r in records:
            product = r.get("Product__r") or {}
            contract = r.get("Contract__r") or {}
            
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
            
            # Extract fiscal year from Created_Date__c
            created_date = r.get("Created_Date__c")
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
                "account_code": r.get("Account_Code_C__c"),
                "stone_color_type": product.get("STONE_Color_Type__c"),
                "product_code": product.get("ProductCode"),
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
                "area_m2": area_m2
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
