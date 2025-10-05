import pandas as pd
import re
import numpy as np

# --- File paths ---
input_path = r"C:\Users\anand\Desktop\PersonalPortfolio_Project\IndianRestaurantsGNV.xlsx"
output_path = r"C:\Users\anand\Desktop\PersonalPortfolio_Project\IndianRestaurantsGNV_Final.xlsx"

# --- Load dataset ---
df = pd.read_excel(input_path)

# --- Clean column names ---
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
print("🧭 Cleaned column names:", list(df.columns))

# Define columns
restaurant_col = "restaurant_name"
address_col = "address"
item_col = "item"
price_col = "price"

# --- Normalize item names ---
def normalize_item_name(name):
    s = str(name).lower().strip()
    s = re.sub(r'[^a-zA-Z0-9\s]', '', s)
    s = re.sub(r'\b(small|medium|large|extra large|xl|regular|combo|plate|half|full|portion)\b', '', s)
    s = re.sub(r'\bveg(etable)?\b', 'veg', s)
    s = re.sub(r'\bmurgh\b', 'chicken', s)
    s = re.sub(r'\bmutton\b', 'mutton', s)
    s = re.sub(r'\bgoat\b', 'mutton', s)
    s = re.sub(r'\bmakhni\b', 'butter masala', s)
    s = re.sub(r'\s+', ' ', s).strip()

    canonical_map = {
        "butter chicken": "Butter Chicken",
        "chicken butter masala": "Butter Chicken",
        "murgh makhani": "Butter Chicken",
        "chicken tikka masala": "Chicken Tikka Masala",
        "chicken kadhai": "Chicken Kadhai",
        "kadhai chicken": "Chicken Kadhai",
        "paneer butter masala": "Paneer Butter Masala",
        "paneer makhani": "Paneer Butter Masala",
        "veg fried rice": "Veg Fried Rice",
        "vegetable fried rice": "Veg Fried Rice",
        "chicken fried rice": "Chicken Fried Rice",
        "egg fried rice": "Egg Fried Rice",
        "mutton biryani": "Mutton Biryani",
        "goat biryani": "Mutton Biryani",
        "chicken biryani": "Chicken Biryani",
        "veg biryani": "Veg Biryani",
        "vegetable biryani": "Veg Biryani",
        "paneer tikka": "Paneer Tikka",
        "chicken tikka": "Chicken Tikka",
        "gobi manchurian": "Gobi Manchurian",
        "veg manchurian": "Veg Manchurian",
        "chicken manchurian": "Chicken Manchurian",
        "veg curry": "Mixed Veg Curry",
        "vegetable curry": "Mixed Veg Curry",
        "dal tadka": "Dal Tadka",
        "dal makhani": "Dal Makhani",
        "masala dosa": "Masala Dosa",
        "plain dosa": "Plain Dosa",
        "idli sambar": "Idli Sambar",
        "vada sambar": "Vada Sambar",
        "gulab jamun": "Gulab Jamun",
        "rasmalai": "Rasmalai",
        "palak paneer": "Palak Paneer",
        "chana masala": "Chana Masala",
    }

    for k, v in canonical_map.items():
        if k in s:
            return v.title()
    return s.title()

df["Normalized Item Name"] = df[item_col].apply(normalize_item_name)

# --- Category / Regional / Veg logic ---
def classify_category(x):
    s = str(x).lower()
    mapping = {
        "Appetizers": ["samosa","pakora","cutlet","chaat","spring roll","kebab","65","tikka","manchurian","mirchi"],
        "Biryani/Rice": ["biryani","rice","pulao","fried rice","polav"],
        "Beverages": ["lassi","chai","tea","coffee","milkshake","juice","soda","water","drink"],
        "Soup / Daal": ["dal","daal","soup","rasam","sambar"],
        "Entrée": ["masala","korma","butter","methi","palak","curry","chettinad","vindaloo","kadhai","do pyaza","kofta","gobi"],
        "Bread": ["naan","roti","paratha","kulcha","poori"],
        "Salad": ["salad"],
        "Sides": ["raita","pickle","papad","sauce"],
        "Dessert": ["gulab","kheer","halwa","rasmalai","ice cream","laddu","jalebi","kulfi","faluda"],
        "Tandoori": ["tandoori","seekh"],
        "Drinks (NA)": ["soda","mocktail","soft drink"],
        "Drinks (A)": ["beer","wine","vodka","rum","whiskey"]
    }
    for k, v in mapping.items():
        if any(w in s for w in v):
            return k
    return "Other"

def classify_regional(x):
    s = str(x).lower()
    if any(w in s for w in ["paneer","butter","tikka","paratha","korma","dal","rajma","tandoori","chaat","naan","kofta","masala"]): 
        return "Regional (North)"
    if any(w in s for w in ["dosa","idli","sambar","rasam","chettinad","uttapam","pongal","sukka","hyderabadi","mirchi"]): 
        return "Regional (South)"
    if any(w in s for w in ["manchurian","chili","schezwan","fried rice","hakka","noodles"]): 
        return "Indo-Chinese"
    if any(w in s for w in ["goan","bengali","kashmiri","gujarati"]): 
        return "Regional (Other)"
    return "General"

def classify_veg_nonveg(x, cat):
    s = str(x).lower()
    if cat in ["Dessert","Bread","Beverages","Drinks (NA)","Drinks (A)"]:
        return "N/A"
    if any(w in s for w in ["chicken","mutton","lamb","fish","shrimp","prawn","egg"]):
        return "Non-Veg"
    if any(w in s for w in ["paneer","aloo","mushroom","veg","vegetable","gobi","chana","dal","tofu","mirchi","chaat","kofta"]):
        return "Veg"
    return "Unknown"

df["Category"] = df[item_col].apply(classify_category)
df["Regional"] = df[item_col].apply(classify_regional)
df["Veg/Non-Veg"] = df.apply(lambda r: classify_veg_nonveg(r[item_col], r["Category"]), axis=1)

# --- Clean numeric price ---
df["Price"] = df["price"].astype(str).str.replace(r"[^0-9.]", "", regex=True)
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
df = df.dropna(subset=[restaurant_col, item_col, "Price"])

# --- Manzil Fine Dining deduplication ---
manzil = df[df[restaurant_col].str.contains("manzil", case=False, na=False)]
others = df[~df[restaurant_col].str.contains("manzil", case=False, na=False)]

manzil_deduped = (
    manzil.groupby(
        [restaurant_col, "Normalized Item Name", "Category", "Regional", "Veg/Non-Veg", address_col],
        as_index=False
    )["Price"].mean()
)

# Merge back with others
final_df = pd.concat([others, manzil_deduped], ignore_index=True)

# --- Select and rename final columns ---
final_df = final_df.rename(columns={
    restaurant_col: "Restaurant Name",
    address_col: "Address",
    item_col: "Item",
    "Normalized Item Name": "Normalized Item Name",
    "Category": "Category",
    "Regional": "Regional",
    "Veg/Non-Veg": "Veg/Non-Veg",
    "Price": "Price"
})

final_df = final_df[["Restaurant Name", "Address", "Item", "Normalized Item Name", "Category", "Regional", "Veg/Non-Veg", "Price"]]

# --- Save safely ---
try:
    final_df.to_excel(output_path, index=False)
    print(f"✅ Done! File created successfully at:\n{output_path}")
except PermissionError:
    alt_path = output_path.replace(".xlsx", "_new.xlsx")
    final_df.to_excel(alt_path, index=False)
    print(f"⚠️ Original file was open — saved new copy as:\n{alt_path}")

print(f"🍛 Total rows after cleaning: {len(final_df)}")
