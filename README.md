# AI-based Flood Prediction and Risk Alarm System 

##  Project Overview
This project predicts the risk of floods using historical and climatic data.  
It falls under the theme **Climate Risk and Disaster Management**.

## ⚙️ Steps Completed (Week 1)
- Imported necessary libraries  
- Loaded and explored dataset  
- Cleaned missing values and duplicates  
- Performed descriptive statistics and visualizations  

##  Files
- `flood_prediction.ipynb` → Jupyter Notebook with code  
- `dataset.csv` → Dataset used  

# 🌊 Flood Prediction – Week 2 Assessment

## 📌 Overview
This week’s assessment focuses on **Exploratory Data Analysis (EDA)**, **data transformation**, and **feature selection** for the flood prediction dataset.  
The goal was to preprocess the data and identify the most important features that influence flood probability.

---

## 🔹 Tasks Done
### 1. Exploratory Data Analysis (EDA)
- Checked for missing values and dataset summary statistics.
- Visualized feature distributions and target variable (`FloodProbability`).
- Plotted correlation heatmap to identify highly related features.

### 2. Data Transformation
- Scaled features using **StandardScaler** to bring them to the same scale.
- Structured dataset into input features (`X`) and target variable (`y`).
- Created a processed dataset `flood_processed.csv` for model training.

### 3. Feature Selection
- Used **correlation analysis** to find strong relationships with the target.
- Applied **Mutual Information (MI)** to capture non-linear dependencies.
- Implemented **Random Forest feature importance** to rank key factors.

---

## 🔹 Key Improvisations from Week 1
- Added **correlation heatmap** and improved visualization in EDA.
- Applied **scaling and normalization** for cleaner data.
- Introduced **multiple feature selection methods** (Correlation, MI, Random Forest).
- Saved **processed dataset** for reuse in modeling.

---

## 📂 Files
- `eda_flood.ipynb` → Jupyter Notebook with full analysis  
- `flood.csv` → Original dataset  
- `flood_processed.csv` → Transformed dataset after preprocessing  

---

## 🚀 Next Steps
- Build machine learning models on the processed dataset.  
- Compare performance using selected features vs. all features.  
- Optimize feature engineering and hyperparameters.

---

