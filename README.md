# Indian-Restaurants-GNV

# Indian Food Finder for UF Students

A comprehensive data-driven web application that helps University of Florida students discover the best Indian restaurants near campus based on budget, distance, and dietary preferences.

## 🎯 Project Overview

As a college student at UF, finding quality Indian food that fits both your budget and schedule can be challenging. This project addresses that problem by analyzing restaurant data from Gainesville's Indian food scene and creating an interactive tool that helps students make informed dining decisions.

## 📊 The Data Challenge

The project started with raw restaurant data that needed significant cleaning and analysis:

- **Multiple Excel files** with inconsistent formatting
- **Missing distance data** from campus
- **Inconsistent naming** across restaurants and menu items
- **Price variations** that needed normalization
- **Category classifications** that required standardization

## 🔧 Data Processing Pipeline

### 1. Data Cleaning (`DataCleanse.py`)
- Standardized restaurant names and menu items
- Normalized price formats and removed outliers
- Categorized food items consistently
- Handled missing values and data inconsistencies

### 2. Distance Analysis
Since distance data wasn't available in the original dataset, I manually researched and added:
- **Indian Aroma**: 0.92 miles
- **Saffron Spice**: 0.42 miles  
- **Manzil Fine Dining**: 4.34 miles
- **Tikka Express**: 0.74 miles
- **Indian Cuisine**: 2.30 miles
- **Indian Street Food**: 2.30 miles
- **Mint Indian Cuisine**: 4.64 miles

### 3. Value Score Algorithm
Created a custom scoring system that considers both price and distance:

```python
Value Score = 10 - min((Price × Distance)/2, 10)
```

This formula:
- **Rewards** cheap items close to campus (higher scores)
- **Penalizes** expensive items far from campus (lower scores)
- **Caps** extreme values to maintain interpretability
- **Prioritizes** student-friendly options

## 🎨 Application Development

### Core Features
- **Interactive filtering** by budget, distance, and dietary preferences
- **Student-friendly interface** with intuitive controls
- **Real-time recommendations** based on user criteria
- **Value scoring** to identify best deals

### Advanced Analytics
- **Correlation analysis** between price, distance, and value
- **Regression modeling** to understand pricing patterns
- **Statistical validation** of the value scoring system
- **Market segmentation** analysis

## 📈 Key Insights Discovered

### Price Patterns
- **Distance premium**: Restaurants closer to campus tend to charge more
- **Category hierarchy**: Tandoori and Biryani items command premium prices
- **Budget options**: Appetizers and basic curries offer better value

### Student Accessibility
- **Convenience trade-off**: Closer restaurants often cost 10-20% more
- **Value sweet spots**: Items under $15 within 2 miles provide best value
- **Dietary options**: Vegetarian items typically more affordable

### Market Analysis
- **Restaurant positioning**: Clear premium vs budget segments
- **Menu diversity**: More variety doesn't always mean better value
- **Regional preferences**: Some cuisines command higher prices

## 🛠️ Technical Implementation

### Technologies Used
- **Python**: Data processing and analysis
- **Streamlit**: Interactive web application
- **Plotly**: Advanced visualizations
- **Pandas**: Data manipulation
- **Scikit-learn**: Machine learning and regression
- **NumPy**: Numerical computations

### Statistical Methods
- **Pearson correlation** analysis
- **Linear regression** modeling
- **Outlier detection** and removal
- **Hypothesis testing** for value score validation

## 🎓 Student-Centric Design

The application was built with college students in mind:

### Budget Considerations
- Default budget filter set to $20 (realistic for students)
- "Student-friendly" filter for items under $15 within 2 miles
- Value scoring that prioritizes affordability

### Convenience Features
- Distance-based filtering with campus as reference point
- Quick access to closest restaurants
- Dietary preference filtering (vegetarian/vegan options)

### User Experience
- Clean, intuitive interface with emojis and clear sections
- Interactive charts with hover tooltips
- Tabbed navigation for organized information
- Mobile-responsive design

## 📊 Validation and Results

### Statistical Validation
- **Correlation analysis** confirmed value score effectiveness
- **Regression modeling** validated distance-price relationships
- **Outlier analysis** ensured data quality
- **Sample size adequacy** for reliable statistical conclusions

### Business Impact
- **Market insights** for restaurant owners
- **Student accessibility** improvements
- **Price transparency** in the local market
- **Data-driven recommendations** for dining choices

## 🚀 Future Enhancements

### Potential Improvements
- **Real-time pricing** updates
- **User reviews** integration
- **Delivery options** analysis
- **Nutritional information** inclusion
- **Social features** for group dining

### Technical Upgrades
- **Database integration** for dynamic updates
- **API connections** for real-time data
- **Machine learning** for personalized recommendations
- **Mobile app** development

## 📁 Project Structure

```
Indian-Restaurants-GNV/
├── indian_food_finder_app_with_chart.py    # Main application
├── DataCleanse.py                          # Data cleaning script
├── pricing_analysis.py                     # Price analysis
├── IndianRestaurantsGNV_Final_new.xlsx     # Cleaned dataset
├── detailed_analysis.txt                   # Analysis results
└── README.md                               # This file
```

## 🎯 Key Takeaways

This project demonstrates how data science can solve real-world problems for specific communities. By combining:

- **Thorough data cleaning** and preparation
- **Domain expertise** in Indian cuisine and student needs
- **Statistical analysis** for validation
- **User-centered design** for accessibility

I created a tool that genuinely helps UF students make better dining decisions while providing valuable insights into the local restaurant market.

## 🔍 Methodology Notes

The correlation analysis revealed that the value scoring system works as intended - items with higher scores (better value) are indeed more suitable for budget-conscious students. The regression analysis confirmed expected relationships between distance, price, and perceived value, validating the approach used in this project.

This project showcases the intersection of data science, user experience design, and domain knowledge to create practical solutions for real-world problems.
