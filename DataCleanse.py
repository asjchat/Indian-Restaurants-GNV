import pandas as pd
import numpy as np
from collections import Counter
import re

def read_and_analyze_data():
    """Read the Excel file and analyze its structure"""
    try:
        # Read the Excel file
        df = pd.read_excel('IndianRestaurantsGNV.xlsx')
        
        print("Dataset Shape:", df.shape)
        print("\nColumn Names:")
        print(df.columns.tolist())
        print("\nFirst few rows:")
        print(df.head())
        print("\nData types:")
        print(df.dtypes)
        print("\nBasic info:")
        print(df.info())
        
        return df
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def analyze_menu_items(df):
    """Analyze menu items to identify inconsistencies"""
    if 'Item ' in df.columns:
        menu_items = df['Item '].dropna().str.strip()
        print(f"\nTotal unique menu items: {menu_items.nunique()}")
        print(f"Total menu items: {len(menu_items)}")
        
        # Show most common items
        print("\nMost common menu items:")
        print(menu_items.value_counts().head(30))
        
        return menu_items
    else:
        print("Available columns:", df.columns.tolist())
        return None

def analyze_categories(df):
    """Analyze categories to identify inconsistencies"""
    if 'Restaurant Menu Classification' in df.columns:
        categories = df['Restaurant Menu Classification'].dropna().str.strip()
        print(f"\nTotal unique categories: {categories.nunique()}")
        print(f"Total categories: {len(categories)}")
        
        # Show all categories
        print("\nAll categories:")
        print(categories.value_counts())
        
        return categories
    else:
        print("Available columns:", df.columns.tolist())
        return None

def get_detailed_analysis(df):
    """Get detailed analysis of menu items and categories"""
    print("\n" + "="*50)
    print("DETAILED CATEGORY ANALYSIS")
    print("="*50)
    
    categories = df['Restaurant Menu Classification'].dropna().str.strip()
    print("All categories with counts:")
    for cat, count in categories.value_counts().items():
        print(f"  {cat}: {count}")
    
    print("\n" + "="*50)
    print("DETAILED MENU ITEM ANALYSIS")
    print("="*50)
    
    menu_items = df['Item '].dropna().str.strip()
    print("All menu items with counts:")
    for item, count in menu_items.value_counts().items():
        print(f"  {item}: {count}")
    
    # Save detailed analysis to file
    with open('detailed_analysis.txt', 'w', encoding='utf-8') as f:
        f.write("DETAILED CATEGORY ANALYSIS\n")
        f.write("="*50 + "\n")
        for cat, count in categories.value_counts().items():
            f.write(f"  {cat}: {count}\n")
        
        f.write("\nDETAILED MENU ITEM ANALYSIS\n")
        f.write("="*50 + "\n")
        for item, count in menu_items.value_counts().items():
            f.write(f"  {item}: {count}\n")
    
    print("\nDetailed analysis saved to 'detailed_analysis.txt'")
    
    return menu_items, categories

def create_standardization_mappings():
    """Create mappings for standardizing menu items and categories"""
    
    # Category standardization mappings
    category_mappings = {
        # Appetizers/Starters
        'Appetizer': 'Appetizers',
        'Apps': 'Appetizers', 
        'Express Starters': 'Appetizers',
        
        # Entrees - will be further categorized as Veg/Non-Veg
        'Entrees': 'Entrées',
        'Main Entrées': 'Entrées',
        'Vegetarian Entrées': 'Veg Entrées',
        'Non-Veg Entrées': 'Non-Veg Entrées',
        'Veg': 'Veg Entrées',
        'Chicken': 'Non-Veg Entrées',
        'Lamb': 'Non-Veg Entrées',
        'Goat': 'Non-Veg Entrées',
        'Seafood': 'Non-Veg Entrées',
        'Non-Veg Delicacies': 'Non-Veg Entrées',
        'Veg Delicacies': 'Veg Entrées',
        'Vegetarian Specialties': 'Veg Entrées',
        
        # Bread
        'Breads': 'Bread',
        'Naans': 'Bread',
        'Indian Bread': 'Bread',
        
        # Beverages/Drinks
        'Drinks': 'Beverages',
        'Bevs': 'Beverages',
        
        # Biryani/Rice
        'Biryanis': 'Biryani',
        'Rice/Biryani': 'Biryani',
        'Biryani & Rice': 'Biryani',
        'Rice': 'Biryani',
        
        # Desserts
        'Desserts': 'Dessert',
        
        # Salads
        'Salads': 'Salad',
        'Soups & Salads': 'Salad',
        
        # Soups
        'Soup': 'Soups',
    }
    
    # Menu item standardization mappings
    menu_item_mappings = {
        # Butter Chicken variations
        'Chicken Butter Masala': 'Butter Chicken',
        'Purani Dilli Ka Butter Chicken': 'Butter Chicken',
        'Lunch Purani Dilli Ka Butter Chicken': 'Butter Chicken',
        
        # Paneer variations
        'Paneer Butter Masala': 'Paneer Tikka Masala',
        'Paneer Makhani': 'Paneer Tikka Masala',
        
        # Dal variations
        'Dal Makhni': 'Dal Makhani',
        'Daal Makhani': 'Dal Makhani',
        
        # Chana/Channa variations
        'Channa Masala': 'Chana Masala',
        'Chole Bhature': 'Chana Masala',
        
        # Remove size specifications
        'Dal Tadka (Regular)': 'Dal Tadka',
        'Dal Tadka (Large)': 'Dal Tadka',
        'Dal Makhani (Regular)': 'Dal Makhani',
        'Dal Makhani (Large)': 'Dal Makhani',
        'Steamed Rice (Regular)': 'Steamed Rice',
        'Steamed Rice (Large)': 'Steamed Rice',
        'Jeera Rice (Regular)': 'Jeera Rice',
        'Jeera Rice (Large)': 'Jeera Rice',
        'Chicken Biryani (Regular)': 'Chicken Biryani',
        'Chicken Biryani (Large)': 'Chicken Biryani',
        'Paneer Butter Masala (Regular)': 'Paneer Butter Masala',
        'Paneer Butter Masala (Large)': 'Paneer Butter Masala',
        'Channa Masala (Regular)': 'Chana Masala',
        'Channa Masala (Large)': 'Chana Masala',
        'Butter Chicken (Regular)': 'Butter Chicken',
        'Butter Chicken (Large)': 'Butter Chicken',
        'Chicken Tikka Masala (Regular)': 'Chicken Tikka Masala',
        'Chicken Tikka Masala (Large)': 'Chicken Tikka Masala',
        'Lamb Rogan Josh (Regular)': 'Lamb Rogan Josh',
        'Lamb Rogan Josh (Large)': 'Lamb Rogan Josh',
        'Prawns Curry (Regular)': 'Prawns Curry',
        'Prawns Curry (Large)': 'Prawns Curry',
        
        # Remove lunch prefix
        'Lunch Chicken Tikka Masala': 'Chicken Tikka Masala',
        'Lunch Shrimp Curry': 'Shrimp Curry',
        'Lunch Dhaba Chicken Curry': 'Dhaba Chicken Curry',
        'Lunch Rara Goat Curry': 'Rara Goat Curry',
        'Lunch Kadhai Chicken': 'Kadhai Chicken',
        'Lunch Navrantan Korma': 'Navrantan Korma',
        'Lunch Chana Masala': 'Chana Masala',
        'Lunch Lasooni Dal Tadka': 'Lasooni Dal Tadka',
        'Lunch Baingan Bharta': 'Baingan Bharta',
        'Lunch Palak Paneer': 'Palak Paneer',
        'Lunch Dal Makhani': 'Dal Makhani',
        'Lunch Chana Palak': 'Chana Palak',
        'Lunch Sarson Da Saag': 'Sarson Da Saag',
        'Lunch Vegetable Kofta': 'Vegetable Kofta',
        'Lunch Paneer Tikka Masala': 'Paneer Tikka Masala',
        
        # Standardize common variations
        'Chicken 65': 'Chicken 65',
        '65 Chicken': 'Chicken 65',
        'Prawns 65': 'Chicken 65',  # Similar concept
        'Chicken Pakoda': 'Chicken Pakora',
        'Paneer Pakoda': 'Paneer Pakora',
        'Gobi Pakoda': 'Gobi Pakora',
        'Onion Pakoda': 'Onion Pakora',
        'Mix Pakoda': 'Mix Pakora',
        'Cut Mirchi Pakoda': 'Cut Mirchi Pakora',
        'Veg Pakoda': 'Veg Pakora',
        'Gobi Pakoda': 'Gobi Pakora',
        'Paneer Pakoda': 'Paneer Pakora',
        'Chicken Pakoda': 'Chicken Pakora',
        'Mix Pakoda': 'Mix Pakora',
        'Cut Mirchi Pakoda': 'Cut Mirchi Pakora',
        
        # Standardize drink variations
        'Bottled Drinks': 'Soft Drinks',
        'Canned Drinks': 'Soft Drinks',
        'Pit. Drinks': 'Soft Drinks',
        'Fountain Soda': 'Soft Drinks',
        
        # Standardize bread variations
        'Roti/Chapati': 'Roti',
        'Tawa Roti': 'Roti',
        'Plain Roti': 'Roti',
        'Tandoori Roti': 'Roti',
        
        # Standardize rice variations
        'Basmati Rice': 'Rice',
        'Steamed Rice': 'Rice',
        'Jeera Rice': 'Rice',
    }
    
    return category_mappings, menu_item_mappings

def get_regional_classification(menu_item):
    """Determine regional classification based on menu item name"""
    if pd.isna(menu_item):
        return 'General'
    
    menu_item_lower = str(menu_item).lower()
    
    # South Indian dishes
    south_indian_keywords = ['dosa', 'idli', 'vada', 'sambar', 'rasam', 'uttapam', 'pongal', 'bisi bele bath', 
                           'upma', 'medu vada', 'rava dosa', 'mysore', 'chettinad', 'andhra', 'karnataka',
                           'tamil', 'kerala', 'hyderabadi', 'biryani', 'pulao', 'curd rice', 'tamarind rice']
    
    # North Indian dishes
    north_indian_keywords = ['butter chicken', 'tandoori', 'naan', 'roti', 'paratha', 'kulcha', 'bhature',
                           'rajma', 'chole', 'paneer', 'dal makhani', 'palak paneer', 'aloo gobi',
                           'kadai', 'korma', 'rogan josh', 'vindaloo', 'jalfrezi', 'saag', 'mutter',
                           'punjabi', 'mughlai', 'dhaba', 'tawa', 'tandoor']
    
    # Bengali dishes
    bengali_keywords = ['fish curry', 'macher jhol', 'rosogolla', 'mishti doi', 'bengali', 'kolkata']
    
    # Gujarati dishes
    gujarati_keywords = ['dhokla', 'khandvi', 'thepla', 'gujarati', 'gujarat']
    
    # Maharashtrian dishes
    maharashtrian_keywords = ['vada pav', 'pav bhaji', 'misal pav', 'puran poli', 'maharashtrian', 'mumbai']
    
    # Check for regional keywords
    for keyword in south_indian_keywords:
        if keyword in menu_item_lower:
            return 'South Indian'
    
    for keyword in north_indian_keywords:
        if keyword in menu_item_lower:
            return 'North Indian'
    
    for keyword in bengali_keywords:
        if keyword in menu_item_lower:
            return 'Bengali'
    
    for keyword in gujarati_keywords:
        if keyword in menu_item_lower:
            return 'Gujarati'
    
    for keyword in maharashtrian_keywords:
        if keyword in menu_item_lower:
            return 'Maharashtrian'
    
    return 'General'

def get_veg_nonveg_classification(menu_item, category):
    """Determine if a dish is vegetarian or non-vegetarian"""
    if pd.isna(menu_item):
        return 'Unknown'
    
    menu_item_lower = str(menu_item).lower()
    
    # Non-vegetarian keywords
    non_veg_keywords = ['chicken', 'lamb', 'goat', 'mutton', 'beef', 'pork', 'fish', 'shrimp', 'prawn', 
                       'crab', 'lobster', 'egg', 'tandoori', 'kebab', 'biryani', 'curry', 'tikka',
                       'seekh', 'boti', 'malai', 'reshmi', 'hariyali', 'achari', '65', 'manchurian']
    
    # Vegetarian keywords that might be confused
    veg_keywords = ['paneer', 'aloo', 'gobi', 'mutter', 'chana', 'dal', 'rajma', 'palak', 'baingan',
                   'bhindi', 'kaddu', 'lauki', 'tinda', 'karela', 'dosa', 'idli', 'vada', 'sambar',
                   'rasam', 'uttapam', 'pongal', 'upma', 'bisi bele bath', 'curd rice', 'tamarind rice']
    
    # Check if it's clearly non-vegetarian
    for keyword in non_veg_keywords:
        if keyword in menu_item_lower:
            return 'Non-Vegetarian'
    
    # Check if it's clearly vegetarian
    for keyword in veg_keywords:
        if keyword in menu_item_lower:
            return 'Vegetarian'
    
    # Default based on category
    if 'veg' in str(category).lower() or 'vegetarian' in str(category).lower():
        return 'Vegetarian'
    elif 'non-veg' in str(category).lower() or 'chicken' in str(category).lower() or 'lamb' in str(category).lower():
        return 'Non-Vegetarian'
    
    return 'Unknown'

def cleanse_data(df, category_mappings, menu_item_mappings):
    """Apply standardization mappings to cleanse the data"""
    df_cleansed = df.copy()
    
    # Clean categories
    df_cleansed['Restaurant Menu Classification'] = df_cleansed['Restaurant Menu Classification'].str.strip()
    df_cleansed['Restaurant Menu Classification'] = df_cleansed['Restaurant Menu Classification'].replace(category_mappings)
    
    # Clean menu items
    df_cleansed['Item '] = df_cleansed['Item '].str.strip()
    df_cleansed['Item '] = df_cleansed['Item '].replace(menu_item_mappings)
    
    # Add regional classification
    df_cleansed['Regional Classification'] = df_cleansed['Item '].apply(get_regional_classification)
    
    # Add veg/non-veg classification
    df_cleansed['Veg/Non-Veg'] = df_cleansed.apply(
        lambda row: get_veg_nonveg_classification(row['Item '], row['Restaurant Menu Classification']), 
        axis=1
    )
    
    # Further categorize entrees based on veg/non-veg
    def categorize_entrees(row):
        category = str(row['Restaurant Menu Classification']).strip()
        veg_nonveg = str(row['Veg/Non-Veg']).strip()
        
        if category == 'Entrées':
            if veg_nonveg == 'Vegetarian':
                return 'Veg Entrées'
            elif veg_nonveg == 'Non-Vegetarian':
                return 'Non-Veg Entrées'
            else:
                return 'Entrées'  # Keep as is if unknown
        else:
            return category
    
    df_cleansed['Restaurant Menu Classification'] = df_cleansed.apply(categorize_entrees, axis=1)
    
    return df_cleansed

def analyze_cleansed_data(df_cleansed):
    """Analyze the cleansed data to show improvements"""
    print("\n" + "="*60)
    print("ENHANCED CLEANSED DATA ANALYSIS")
    print("="*60)
    
    # Categories after cleansing
    categories_cleansed = df_cleansed['Restaurant Menu Classification'].dropna().str.strip()
    print(f"\nCategories after cleansing: {categories_cleansed.nunique()}")
    print("Top categories:")
    print(categories_cleansed.value_counts().head(15))
    
    # Menu items after cleansing
    menu_items_cleansed = df_cleansed['Item '].dropna().str.strip()
    print(f"\nMenu items after cleansing: {menu_items_cleansed.nunique()}")
    print("Top menu items:")
    print(menu_items_cleansed.value_counts().head(15))
    
    # Regional classification analysis
    regional_classification = df_cleansed['Regional Classification'].dropna()
    print(f"\nRegional Classifications: {regional_classification.nunique()}")
    print("Regional distribution:")
    print(regional_classification.value_counts())
    
    # Veg/Non-Veg classification analysis
    veg_nonveg = df_cleansed['Veg/Non-Veg'].dropna()
    print(f"\nVeg/Non-Veg Classifications: {veg_nonveg.nunique()}")
    print("Veg/Non-Veg distribution:")
    print(veg_nonveg.value_counts())
    
    # Show some examples of the new classifications
    print(f"\n" + "="*60)
    print("SAMPLE CLASSIFICATIONS")
    print("="*60)
    
    sample_data = df_cleansed[['Item ', 'Restaurant Menu Classification', 'Regional Classification', 'Veg/Non-Veg']].head(20)
    print(sample_data.to_string(index=False))
    
    return categories_cleansed, menu_items_cleansed

if __name__ == "__main__":
    # Read and analyze the data
    df = read_and_analyze_data()
    
    if df is not None:
        # Analyze menu items and categories
        menu_items = analyze_menu_items(df)
        categories = analyze_categories(df)
        
        # Get detailed analysis
        menu_items_detailed, categories_detailed = get_detailed_analysis(df)
        
        # Create standardization mappings
        category_mappings, menu_item_mappings = create_standardization_mappings()
        
        # Cleanse the data
        df_cleansed = cleanse_data(df, category_mappings, menu_item_mappings)
        
        # Analyze cleansed data
        categories_cleansed, menu_items_cleansed = analyze_cleansed_data(df_cleansed)
        
        # Save enhanced cleansed data
        df_cleansed.to_excel('IndianRestaurantsGNV_Enhanced_Cleansed.xlsx', index=False)
        print(f"\nEnhanced cleansed data saved to 'IndianRestaurantsGNV_Enhanced_Cleansed.xlsx'")
        
        # Show summary of changes
        print(f"\n" + "="*60)
        print("SUMMARY OF ENHANCED CLEANSING")
        print("="*60)
        print(f"Original categories: {categories.nunique()}")
        print(f"Enhanced cleansed categories: {categories_cleansed.nunique()}")
        print(f"Categories reduced by: {categories.nunique() - categories_cleansed.nunique()}")
        
        print(f"\nOriginal menu items: {menu_items.nunique()}")
        print(f"Enhanced cleansed menu items: {menu_items_cleansed.nunique()}")
        print(f"Menu items reduced by: {menu_items.nunique() - menu_items_cleansed.nunique()}")
        
        # Show new features
        print(f"\nNEW FEATURES ADDED:")
        print(f"- Regional Classification: {df_cleansed['Regional Classification'].nunique()} categories")
        print(f"- Veg/Non-Veg Classification: {df_cleansed['Veg/Non-Veg'].nunique()} categories")
        print(f"- Enhanced Entrée categorization (Veg/Non-Veg separation)")
        
        # Show column information
        print(f"\nNEW COLUMNS ADDED:")
        print(f"- 'Regional Classification': Categorizes dishes by Indian regional cuisine")
        print(f"- 'Veg/Non-Veg': Classifies dishes as Vegetarian, Non-Vegetarian, or Unknown")
        print(f"- Enhanced 'Restaurant Menu Classification': Now includes Veg/Non-Veg Entrées")
