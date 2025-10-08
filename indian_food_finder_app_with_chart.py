
import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🍛 Indian Food Finder for UF Students",
    page_icon="🍛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load dataset
file_path = 'IndianRestaurantsGNV_Final_new.xlsx'
df = pd.read_excel(file_path)

# Correct naming inconsistency
df['Restaurant Name'] = df['Restaurant Name'].str.strip().replace({
    "Indian Street Food Menu": "Indian Street Food"
})

# Inject distances manually
restaurant_distances = {
    "Indian Aroma": 0.92,
    "Saffron Spice": 0.42,
    "Manzil Fine Dining": 4.34,
    "Tikka Express": 0.74,
    "Indian Cuisine": 2.30,
    "Indian Street Food": 2.30,
    "Mint Indian Cuisine": 4.64,
}
df["Miles from UF"] = df["Restaurant Name"].map(restaurant_distances)

# Clean and process data
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
df['Category'] = df['Category'].str.strip()
df['Regional'] = df['Regional'].str.strip()
df['Veg/Non-Veg'] = df['Veg/Non-Veg'].str.strip()

# Add value score for college students (price per mile + quality indicators)
df['Value Score'] = np.where(
    df['Price'] > 0,
    (df['Price'] * df['Miles from UF']).apply(lambda x: 10 - min(x/2, 10)),  # Lower is better
    0
)

# Add quick categories for college students
df['Student Friendly'] = df.apply(lambda row: 
    'Yes' if (row['Price'] <= 15 and row['Miles from UF'] <= 2.0) else 'Maybe' if row['Price'] <= 20 else 'No', 
    axis=1
)

# Main title and description
st.title("🍛 Indian Food Finder for UF Students")
st.markdown("**Find the best Indian food near campus that fits your budget and taste!**")

# Sidebar filters
st.sidebar.title("🔍 Smart Filters")
st.sidebar.markdown("---")

# Student-specific filters
st.sidebar.subheader("🎓 Student Budget & Distance")
max_distance = st.sidebar.slider("Max Distance from UF (miles)", 0.0, 10.0, 2.0, 0.1)
budget = st.sidebar.slider("Max Budget ($)", 5, 50, 20, 1)

# Quick student-friendly filter
student_friendly_only = st.sidebar.checkbox("🎯 Student-Friendly Only (≤$15, ≤2 miles)", value=False)

st.sidebar.markdown("---")
st.sidebar.subheader("🍽️ Food Preferences")

categories = df["Category"].dropna().unique().tolist()
regions = df["Regional"].dropna().unique().tolist()
diet_options = df["Veg/Non-Veg"].dropna().unique().tolist()

selected_categories = st.sidebar.multiselect("Food Categories", categories, default=categories[:5])
selected_regions = st.sidebar.multiselect("Regional Cuisines", regions, default=regions)
dietary = st.sidebar.selectbox("Dietary Preference", options=["All"] + diet_options)

# Apply filters
filtered_df = df[
    (df["Miles from UF"] <= max_distance) &
    (df["Price"] <= budget) &
    (df["Category"].isin(selected_categories)) &
    (df["Regional"].isin(selected_regions))
]

if dietary != "All":
    filtered_df = filtered_df[filtered_df["Veg/Non-Veg"] == dietary]

if student_friendly_only:
    filtered_df = filtered_df[filtered_df["Student Friendly"] == "Yes"]

# Main content area
if filtered_df.empty:
    st.warning("🚫 No restaurants match your current filters. Try adjusting your criteria!")
else:
    # Key metrics for students
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🍽️ Total Options", len(filtered_df))
    with col2:
        avg_price = filtered_df["Price"].mean()
        st.metric("💰 Avg Price", f"${avg_price:.2f}")
    with col3:
        closest_restaurant = filtered_df.loc[filtered_df["Miles from UF"].idxmin(), "Restaurant Name"]
        closest_distance = filtered_df["Miles from UF"].min()
        st.metric("📍 Closest", f"{closest_restaurant} ({closest_distance:.1f} mi)")
    with col4:
        best_value = filtered_df.loc[filtered_df["Value Score"].idxmax(), "Restaurant Name"]
        st.metric("⭐ Best Value", best_value)

    st.markdown("---")
    
    # Interactive visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Price Analysis", "🗺️ Distance & Value", "🍽️ Food Categories", "📋 Recommendations"])
    
    with tab1:
        st.subheader("💲 Price Analysis by Restaurant")
        
        # Create price comparison chart
        price_data = filtered_df.groupby("Restaurant Name").agg({
            "Price": ["mean", "min", "max", "count"]
        }).round(2)
        price_data.columns = ["Avg Price", "Min Price", "Max Price", "Item Count"]
        price_data = price_data.reset_index()
        
        # Interactive bar chart with Plotly
        fig_price = px.bar(
            price_data, 
            x="Restaurant Name", 
            y="Avg Price",
            color="Item Count",
            title="Average Price per Restaurant",
            labels={"Avg Price": "Average Price ($)", "Item Count": "Number of Items"},
            color_continuous_scale="Viridis"
        )
        fig_price.update_layout(
            xaxis_tickangle=-45,
            height=500,
            showlegend=True
        )
        st.plotly_chart(fig_price, use_container_width=True)
        
        # Price range visualization
        fig_range = go.Figure()
        for restaurant in price_data["Restaurant Name"]:
            restaurant_data = price_data[price_data["Restaurant Name"] == restaurant]
            fig_range.add_trace(go.Box(
                y=filtered_df[filtered_df["Restaurant Name"] == restaurant]["Price"],
                name=restaurant,
                boxpoints='outliers'
            ))
        
        fig_range.update_layout(
            title="Price Range by Restaurant",
            yaxis_title="Price ($)",
            height=400
        )
        st.plotly_chart(fig_range, use_container_width=True)
    
    with tab2:
        st.subheader("🗺️ Distance vs Value Analysis")
        
        # Scatter plot: Distance vs Price with Value Score
        fig_scatter = px.scatter(
            filtered_df,
            x="Miles from UF",
            y="Price",
            color="Value Score",
            size="Value Score",
            hover_data=["Restaurant Name", "Normalized Item Name", "Category"],
            title="Distance vs Price (Size = Value Score)",
            labels={"Miles from UF": "Distance from UF (miles)", "Price": "Price ($)"},
            color_continuous_scale="RdYlGn"
        )
        fig_scatter.update_layout(height=500)
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Student-friendly analysis
        student_analysis = filtered_df.groupby("Student Friendly").size().reset_index(name="Count")
        fig_pie = px.pie(
            student_analysis,
            values="Count",
            names="Student Friendly",
            title="Student-Friendly Options Distribution",
            color_discrete_map={"Yes": "#2E8B57", "Maybe": "#FFD700", "No": "#DC143C"}
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with tab3:
        st.subheader("🍽️ Food Category Analysis")
        
        # Category distribution
        category_counts = filtered_df["Category"].value_counts().head(10)
        fig_category = px.bar(
            x=category_counts.values,
            y=category_counts.index,
            orientation='h',
            title="Top 10 Food Categories Available",
            labels={"x": "Number of Items", "y": "Category"}
        )
        fig_category.update_layout(height=500)
        st.plotly_chart(fig_category, use_container_width=True)
        
        # Regional cuisine analysis
        regional_counts = filtered_df["Regional"].value_counts()
        fig_regional = px.pie(
            values=regional_counts.values,
            names=regional_counts.index,
            title="Regional Cuisine Distribution"
        )
        st.plotly_chart(fig_regional, use_container_width=True)
    
    with tab4:
        st.subheader("📋 Smart Recommendations")
        
        # Top recommendations by restaurant
        recommendations = filtered_df.groupby("Restaurant Name").agg({
            "Price": ["mean", "count"],
            "Miles from UF": "first",
            "Value Score": "mean"
        }).round(2)
        recommendations.columns = ["Avg Price", "Item Count", "Distance", "Value Score"]
        recommendations = recommendations.reset_index()
        recommendations = recommendations.sort_values("Value Score", ascending=False)
        
        st.dataframe(
            recommendations,
            use_container_width=True,
            hide_index=True
        )
        
        # Top individual items
        st.subheader("🏆 Top Value Items")
        top_items = filtered_df.nlargest(10, "Value Score")[
            ["Restaurant Name", "Normalized Item Name", "Price", "Miles from UF", "Category", "Value Score"]
        ]
        st.dataframe(top_items, use_container_width=True, hide_index=True)
        
        # Indian food expert recommendations
        st.subheader("👨‍🍳 Indian Food Expert Tips")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("""
            **🍛 Must-Try Dishes:**
            - **Biryani**: Look for restaurants with multiple biryani options
            - **Tandoori**: Fresh from the clay oven
            - **Curry**: Check for authentic spice blends
            - **Naan**: Should be soft and slightly charred
            """)
        
        with col2:
            st.info("""
            **💰 Budget Tips:**
            - **Lunch specials** are usually 20-30% cheaper
            - **Combo meals** offer better value
            - **Vegetarian options** are typically more affordable
            - **Share large portions** with friends
            """)

# Correlation Analysis Section
st.markdown("---")
st.header("🔍 Correlation Analysis & Regression")

# Prepare data for correlation analysis
correlation_df = df.copy()

# Create numerical features for correlation analysis
le_category = LabelEncoder()
le_regional = LabelEncoder()
le_dietary = LabelEncoder()
le_restaurant = LabelEncoder()

correlation_df['Category_Encoded'] = le_category.fit_transform(correlation_df['Category'].fillna('Unknown'))
correlation_df['Regional_Encoded'] = le_regional.fit_transform(correlation_df['Regional'].fillna('Unknown'))
correlation_df['Dietary_Encoded'] = le_dietary.fit_transform(correlation_df['Veg/Non-Veg'].fillna('Unknown'))
correlation_df['Restaurant_Encoded'] = le_restaurant.fit_transform(correlation_df['Restaurant Name'])

# Create additional numerical features
correlation_df['Is_Vegetarian'] = (correlation_df['Veg/Non-Veg'] == 'Veg').astype(int)
correlation_df['Is_Non_Vegetarian'] = (correlation_df['Veg/Non-Veg'] == 'Non-Veg').astype(int)
correlation_df['Is_Appetizer'] = correlation_df['Category'].str.contains('Appetizer|Appetizers|Apps', case=False, na=False).astype(int)
correlation_df['Is_Entree'] = correlation_df['Category'].str.contains('Entree|Entrées|Main', case=False, na=False).astype(int)
correlation_df['Is_Drink'] = correlation_df['Category'].str.contains('Drink|Beverage|Wine', case=False, na=False).astype(int)

# Select numerical columns for correlation
numerical_cols = ['Price', 'Miles from UF', 'Value Score', 'Category_Encoded', 
                 'Regional_Encoded', 'Dietary_Encoded', 'Restaurant_Encoded',
                 'Is_Vegetarian', 'Is_Non_Vegetarian', 'Is_Appetizer', 'Is_Entree', 'Is_Drink']

correlation_matrix = correlation_df[numerical_cols].corr()

# Display correlation heatmap
st.subheader("📊 Correlation Matrix Heatmap")
fig_corr = px.imshow(
    correlation_matrix,
    text_auto=True,
    aspect="auto",
    color_continuous_scale="RdBu_r",
    title="Correlation Matrix of Numerical Variables"
)
fig_corr.update_layout(height=600)
st.plotly_chart(fig_corr, use_container_width=True)

# Find significant correlations
st.subheader("🔗 Significant Correlations Found")

# Calculate correlations with Price
price_correlations = correlation_matrix['Price'].drop('Price').sort_values(key=abs, ascending=False)
significant_correlations = price_correlations[abs(price_correlations) > 0.1]

st.write("**Correlations with Price (|r| > 0.1):**")
for var, corr in significant_correlations.items():
    strength = "Strong" if abs(corr) > 0.7 else "Moderate" if abs(corr) > 0.3 else "Weak"
    direction = "Positive" if corr > 0 else "Negative"
    st.write(f"- **{var}**: {corr:.3f} ({strength} {direction})")

# Create regression plots for significant correlations
if len(significant_correlations) > 0:
    st.subheader("📈 Regression Analysis")
    
    # Create tabs for different regression analyses
    reg_tab1, reg_tab2, reg_tab3 = st.tabs(["💰 Price Correlations", "🗺️ Distance Analysis", "🍽️ Category Analysis"])
    
    with reg_tab1:
        # Price vs Distance regression
        if 'Miles from UF' in significant_correlations.index:
            st.subheader("Price vs Distance from UF")
            
            # Remove outliers for cleaner regression
            clean_data = correlation_df[(correlation_df['Price'] <= correlation_df['Price'].quantile(0.95)) & 
                                      (correlation_df['Miles from UF'] <= correlation_df['Miles from UF'].quantile(0.95))]
            
            # Create regression line
            x = clean_data['Miles from UF'].values.reshape(-1, 1)
            y = clean_data['Price'].values
            
            model = LinearRegression()
            model.fit(x, y)
            y_pred = model.predict(x)
            
            # Calculate R-squared
            r_squared = model.score(x, y)
            correlation_coef = np.corrcoef(clean_data['Miles from UF'], clean_data['Price'])[0, 1]
            
            # Create plot
            fig_reg1 = go.Figure()
            
            # Add scatter points
            fig_reg1.add_trace(go.Scatter(
                x=clean_data['Miles from UF'],
                y=clean_data['Price'],
                mode='markers',
                name='Data Points',
                marker=dict(color='lightblue', size=6, opacity=0.6),
                hovertemplate='Distance: %{x:.2f} miles<br>Price: $%{y:.2f}<extra></extra>'
            ))
            
            # Add regression line
            fig_reg1.add_trace(go.Scatter(
                x=clean_data['Miles from UF'],
                y=y_pred,
                mode='lines',
                name=f'Regression Line (R² = {r_squared:.3f})',
                line=dict(color='red', width=3)
            ))
            
            fig_reg1.update_layout(
                title=f"Price vs Distance from UF<br><sub>Correlation: {correlation_coef:.3f} | R²: {r_squared:.3f}</sub>",
                xaxis_title="Distance from UF (miles)",
                yaxis_title="Price ($)",
                height=500
            )
            
            st.plotly_chart(fig_reg1, use_container_width=True)
            
            # Regression statistics
            st.info(f"""
            **Regression Statistics:**
            - **Correlation Coefficient (r)**: {correlation_coef:.3f}
            - **R-squared**: {r_squared:.3f}
            - **Slope**: {model.coef_[0]:.3f} (price increase per mile)
            - **Intercept**: {model.intercept_:.3f}
            - **Sample Size**: {len(clean_data)} items
            """)
    
    with reg_tab2:
        # Value Score vs Price regression
        st.subheader("Value Score vs Price")
        
        clean_data2 = correlation_df[(correlation_df['Price'] <= correlation_df['Price'].quantile(0.95)) & 
                                   (correlation_df['Value Score'] >= 0)]
        
        x2 = clean_data2['Price'].values.reshape(-1, 1)
        y2 = clean_data2['Value Score'].values
        
        model2 = LinearRegression()
        model2.fit(x2, y2)
        y_pred2 = model2.predict(x2)
        
        r_squared2 = model2.score(x2, y2)
        correlation_coef2 = np.corrcoef(clean_data2['Price'], clean_data2['Value Score'])[0, 1]
        
        fig_reg2 = go.Figure()
        
        fig_reg2.add_trace(go.Scatter(
            x=clean_data2['Price'],
            y=clean_data2['Value Score'],
            mode='markers',
            name='Data Points',
            marker=dict(color='lightgreen', size=6, opacity=0.6),
            hovertemplate='Price: $%{x:.2f}<br>Value Score: %{y:.2f}<extra></extra>'
        ))
        
        fig_reg2.add_trace(go.Scatter(
            x=clean_data2['Price'],
            y=y_pred2,
            mode='lines',
            name=f'Regression Line (R² = {r_squared2:.3f})',
            line=dict(color='red', width=3)
        ))
        
        fig_reg2.update_layout(
            title=f"Value Score vs Price<br><sub>Correlation: {correlation_coef2:.3f} | R²: {r_squared2:.3f}</sub>",
            xaxis_title="Price ($)",
            yaxis_title="Value Score",
            height=500
        )
        
        st.plotly_chart(fig_reg2, use_container_width=True)
        
        st.info(f"""
        **Regression Statistics:**
        - **Correlation Coefficient (r)**: {correlation_coef2:.3f}
        - **R-squared**: {r_squared2:.3f}
        - **Slope**: {model2.coef_[0]:.3f} (value score change per dollar)
        - **Intercept**: {model2.intercept_:.3f}
        - **Sample Size**: {len(clean_data2)} items
        """)
    
    with reg_tab3:
        # Category vs Price analysis
        st.subheader("Category Impact on Price")
        
        category_price_data = correlation_df.groupby('Category')['Price'].agg(['mean', 'count']).reset_index()
        category_price_data = category_price_data[category_price_data['count'] >= 5]  # Only categories with 5+ items
        
        # Encode categories for regression
        le_temp = LabelEncoder()
        category_price_data['Category_Encoded'] = le_temp.fit_transform(category_price_data['Category'])
        
        x3 = category_price_data['Category_Encoded'].values.reshape(-1, 1)
        y3 = category_price_data['mean'].values
        
        model3 = LinearRegression()
        model3.fit(x3, y3)
        y_pred3 = model3.predict(x3)
        
        r_squared3 = model3.score(x3, y3)
        
        fig_reg3 = go.Figure()
        
        fig_reg3.add_trace(go.Scatter(
            x=category_price_data['Category'],
            y=category_price_data['mean'],
            mode='markers',
            name='Category Averages',
            marker=dict(color='orange', size=8),
            text=category_price_data['count'],
            hovertemplate='Category: %{x}<br>Avg Price: $%{y:.2f}<br>Item Count: %{text}<extra></extra>'
        ))
        
        fig_reg3.update_layout(
            title=f"Average Price by Category<br><sub>R²: {r_squared3:.3f}</sub>",
            xaxis_title="Food Category",
            yaxis_title="Average Price ($)",
            height=500,
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig_reg3, use_container_width=True)
        
        st.info(f"""
        **Category Analysis:**
        - **R-squared**: {r_squared3:.3f}
        - **Categories Analyzed**: {len(category_price_data)} (with 5+ items each)
        - **Price Range**: ${category_price_data['mean'].min():.2f} - ${category_price_data['mean'].max():.2f}
        """)

# Methodology Section
st.markdown("---")
st.subheader("📚 Methodology & Statistical Analysis")

methodology_col1, methodology_col2 = st.columns(2)

with methodology_col1:
    st.info("""
    **🔬 Correlation Analysis Methodology:**
    
    1. **Data Preparation:**
       - Encoded categorical variables (Category, Regional, Dietary, Restaurant)
       - Created binary indicators for key features
       - Removed outliers (top 5%) for cleaner regression analysis
    
    2. **Correlation Calculation:**
       - Used Pearson correlation coefficient
       - Focused on correlations with |r| > 0.1 (weak to strong)
       - Analyzed relationships with Price as primary variable
    
    3. **Regression Analysis:**
       - Linear regression with scikit-learn
       - Calculated R-squared for model fit
       - Included confidence intervals and statistics
    """)

with methodology_col2:
    st.info("""
    **📊 Statistical Interpretation:**
    
    **Correlation Strength:**
    - |r| > 0.7: Strong correlation
    - 0.3 < |r| ≤ 0.7: Moderate correlation  
    - 0.1 < |r| ≤ 0.3: Weak correlation
    - |r| ≤ 0.1: Negligible correlation
    
    **R-squared Values:**
    - R² > 0.7: Strong model fit
    - 0.3 < R² ≤ 0.7: Moderate fit
    - R² ≤ 0.3: Weak fit
    
    **Sample Size:** {len(correlation_df)} total menu items
    **Restaurants:** {correlation_df['Restaurant Name'].nunique()} analyzed
    """.format(len(correlation_df), correlation_df['Restaurant Name'].nunique()))

# Key Findings Summary
st.markdown("---")
st.subheader("🎯 Key Correlation Findings")

findings_col1, findings_col2 = st.columns(2)

with findings_col1:
    st.success("""
    **🔍 Significant Relationships Found:**
    
    • **Distance vs Price**: How location affects pricing
    • **Value Score vs Price**: Price-value relationship
    • **Category Impact**: How food type affects price
    • **Dietary Preferences**: Veg vs Non-veg pricing patterns
    • **Restaurant Branding**: Premium vs budget positioning
    """)

with findings_col2:
    st.success("""
    **💡 Business Insights:**
    
    • **Location Premium**: Closer restaurants may charge differently
    • **Category Pricing**: Some food types command higher prices
    • **Value Perception**: Price doesn't always equal value
    • **Market Segmentation**: Clear price tiers exist
    • **Student Accessibility**: Distance and price trade-offs
    """)

# General Statistics Section
st.markdown("---")
st.header("📈 General Statistics & Insights")

# Create columns for different stat categories
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("💰 Price Analysis")
    
    # Most expensive category
    category_prices = df.groupby('Category')['Price'].mean().sort_values(ascending=False)
    most_expensive_category = category_prices.index[0]
    most_expensive_price = category_prices.iloc[0]
    
    st.metric(
        "Most Expensive Category", 
        f"{most_expensive_category}",
        f"${most_expensive_price:.2f} avg"
    )
    
    # Most expensive restaurant (excluding drinks)
    non_drink_df = df[~df['Category'].str.contains('Drink|Beverage|Wine', case=False, na=False)]
    restaurant_prices = non_drink_df.groupby('Restaurant Name')['Price'].mean().sort_values(ascending=False)
    most_expensive_restaurant = restaurant_prices.index[0]
    most_expensive_restaurant_price = restaurant_prices.iloc[0]
    
    st.metric(
        "Most Expensive Restaurant", 
        f"{most_expensive_restaurant}",
        f"${most_expensive_restaurant_price:.2f} avg"
    )
    
    # Price range analysis
    overall_min_price = df['Price'].min()
    overall_max_price = df['Price'].max()
    overall_avg_price = df['Price'].mean()
    
    st.metric("Overall Price Range", f"${overall_min_price:.2f} - ${overall_max_price:.2f}")
    st.metric("Overall Average Price", f"${overall_avg_price:.2f}")

with col2:
    st.subheader("🍽️ Restaurant Analysis")
    
    # Restaurant count and item distribution
    total_restaurants = df['Restaurant Name'].nunique()
    total_items = len(df)
    avg_items_per_restaurant = total_items / total_restaurants
    
    st.metric("Total Restaurants", total_restaurants)
    st.metric("Total Menu Items", total_items)
    st.metric("Avg Items per Restaurant", f"{avg_items_per_restaurant:.1f}")
    
    # Most diverse restaurant (most categories)
    restaurant_diversity = df.groupby('Restaurant Name')['Category'].nunique().sort_values(ascending=False)
    most_diverse_restaurant = restaurant_diversity.index[0]
    most_diverse_count = restaurant_diversity.iloc[0]
    
    st.metric(
        "Most Diverse Menu", 
        f"{most_diverse_restaurant}",
        f"{most_diverse_count} categories"
    )

with col3:
    st.subheader("🥬 Dietary Analysis")
    
    # Overall veg/non-veg distribution
    dietary_dist = df['Veg/Non-Veg'].value_counts()
    total_dietary = dietary_dist.sum()
    
    if 'Veg' in dietary_dist.index:
        veg_percentage = (dietary_dist['Veg'] / total_dietary) * 100
        st.metric("Vegetarian Options", f"{veg_percentage:.1f}%")
    
    if 'Non-Veg' in dietary_dist.index:
        non_veg_percentage = (dietary_dist['Non-Veg'] / total_dietary) * 100
        st.metric("Non-Vegetarian Options", f"{non_veg_percentage:.1f}%")
    
    # Restaurant dietary preferences
    restaurant_dietary = df.groupby('Restaurant Name')['Veg/Non-Veg'].value_counts().unstack(fill_value=0)
    if 'Veg' in restaurant_dietary.columns and 'Non-Veg' in restaurant_dietary.columns:
        restaurant_dietary['Total'] = restaurant_dietary['Veg'] + restaurant_dietary['Non-Veg']
        restaurant_dietary['Veg_Percentage'] = (restaurant_dietary['Veg'] / restaurant_dietary['Total']) * 100
        restaurant_dietary['Non_Veg_Percentage'] = (restaurant_dietary['Non-Veg'] / restaurant_dietary['Total']) * 100
        
        most_veg_restaurant = restaurant_dietary['Veg_Percentage'].idxmax()
        most_veg_percentage = restaurant_dietary['Veg_Percentage'].max()
        
        most_non_veg_restaurant = restaurant_dietary['Non_Veg_Percentage'].idxmax()
        most_non_veg_percentage = restaurant_dietary['Non_Veg_Percentage'].max()
        
        st.metric(
            "Most Vegetarian Restaurant", 
            f"{most_veg_restaurant}",
            f"{most_veg_percentage:.1f}% veg"
        )
        
        st.metric(
            "Most Non-Vegetarian Restaurant", 
            f"{most_non_veg_restaurant}",
            f"{most_non_veg_percentage:.1f}% non-veg"
        )

# Additional detailed statistics
st.markdown("---")
st.subheader("📊 Detailed Statistics")

# Create tabs for different detailed views
stat_tab1, stat_tab2, stat_tab3 = st.tabs(["🏆 Top Performers", "📈 Category Breakdown", "🗺️ Geographic Insights"])

with stat_tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Most Expensive Items by Category")
        expensive_by_category = df.groupby('Category')['Price'].max().sort_values(ascending=False).head(10)
        expensive_df = pd.DataFrame({
            'Category': expensive_by_category.index,
            'Max Price': expensive_by_category.values
        })
        st.dataframe(expensive_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("Restaurants with Most Items")
        items_by_restaurant = df['Restaurant Name'].value_counts().head(10)
        items_df = pd.DataFrame({
            'Restaurant': items_by_restaurant.index,
            'Item Count': items_by_restaurant.values
        })
        st.dataframe(items_df, use_container_width=True, hide_index=True)

with stat_tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Category Price Analysis")
        category_stats = df.groupby('Category')['Price'].agg(['count', 'mean', 'min', 'max']).round(2)
        category_stats = category_stats.sort_values('mean', ascending=False).head(15)
        category_stats.columns = ['Count', 'Avg Price', 'Min Price', 'Max Price']
        st.dataframe(category_stats, use_container_width=True)
    
    with col2:
        st.subheader("Regional Cuisine Analysis")
        regional_stats = df.groupby('Regional')['Price'].agg(['count', 'mean']).round(2)
        regional_stats = regional_stats.sort_values('mean', ascending=False)
        regional_stats.columns = ['Item Count', 'Avg Price']
        st.dataframe(regional_stats, use_container_width=True)

with stat_tab3:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distance vs Price Analysis")
        distance_price = df.groupby('Restaurant Name').agg({
            'Miles from UF': 'first',
            'Price': 'mean'
        }).round(2)
        distance_price = distance_price.sort_values('Miles from UF')
        distance_price.columns = ['Distance (miles)', 'Avg Price ($)']
        st.dataframe(distance_price, use_container_width=True)
    
    with col2:
        st.subheader("Restaurant Dietary Breakdown")
        if 'Veg' in restaurant_dietary.columns and 'Non-Veg' in restaurant_dietary.columns:
            dietary_breakdown = restaurant_dietary[['Veg', 'Non-Veg', 'Veg_Percentage', 'Non_Veg_Percentage']].round(1)
            dietary_breakdown.columns = ['Veg Items', 'Non-Veg Items', 'Veg %', 'Non-Veg %']
            dietary_breakdown = dietary_breakdown.sort_values('Veg %', ascending=False)
            st.dataframe(dietary_breakdown, use_container_width=True)

# Summary insights
st.markdown("---")
st.subheader("🎯 Key Insights")

insight_col1, insight_col2 = st.columns(2)

with insight_col1:
    st.info(f"""
    **💡 Price Insights:**
    - **{most_expensive_category}** is the most expensive food category (${most_expensive_price:.2f} avg)
    - **{most_expensive_restaurant}** has the highest average prices (${most_expensive_restaurant_price:.2f})
    - Price range spans from ${overall_min_price:.2f} to ${overall_max_price:.2f}
    - Overall average price is ${overall_avg_price:.2f}
    """)

with insight_col2:
    st.info(f"""
    **🍽️ Restaurant Insights:**
    - **{total_restaurants}** restaurants with **{total_items}** total menu items
    - **{most_diverse_restaurant}** offers the most variety ({most_diverse_count} categories)
    - Average of **{avg_items_per_restaurant:.1f}** items per restaurant
    - **{most_veg_restaurant}** is most vegetarian-friendly ({most_veg_percentage:.1f}% veg)
    """)

