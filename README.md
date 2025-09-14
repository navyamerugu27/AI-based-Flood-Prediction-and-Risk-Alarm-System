🌊 AI-based Flood Prediction and Risk Alarm System
📌 Overview

Floods are one of the most devastating natural disasters, causing loss of life, damage to infrastructure, and severe economic impacts. This project aims to develop an AI-based Flood Prediction and Risk Alarm System that predicts the probability of flooding based on multiple environmental, climatic, and anthropogenic factors. By leveraging machine learning models, the system provides timely alerts and insights for effective disaster management.

📊 Dataset

Source: Kaggle (Flood Prediction Dataset)

File: flood.csv (processed as flood_processed.csv)

Size: ~50,000 records, 21 features

Features include:

MonsoonIntensity

TopographyDrainage

RiverManagement

Deforestation

Urbanization

ClimateChange

DamsQuality

Siltation

AgriculturalPractices

Encroachments

IneffectiveDisasterPreparedness

DrainageSystems

CoastalVulnerability

Landslides

Watersheds

DeterioratingInfrastructure

PopulationScore

WetlandLoss

InadequatePlanning

PoliticalFactors

Target → FloodProbability

⚙️ Project Workflow

Data Preprocessing

Cleaned missing values & standardized the dataset.

Scaled numerical features for better model performance.

Created a processed version: flood_processed.csv.

Exploratory Data Analysis (EDA)

Univariate, Bivariate & Multivariate Analysis.

Correlation heatmaps to identify influential factors.

Visualizations for interpretability of risk factors.

Model Development

Trained multiple machine learning models:

Logistic Regression

Random Forest

XGBoost

Decision Tree, SVM, etc.

Compared all features vs. selected features.

Optimized hyperparameters for better accuracy.

Evaluation Metrics

Accuracy, Precision, Recall, F1-score.

ROC-AUC for probability-based performance.

Deployment

Built an interactive Streamlit web app (streamlit_app.py).

User inputs factor values → System predicts flood probability.

Visual dashboards for easy interpretation.

🚀 Results & Improvisations

Achieved improved model accuracy with feature selection & tuning.

Enhanced interpretability with clear factor-wise visualizations.

Converted raw dataset into a processed, model-ready version.

Built an interactive, user-friendly app for real-world usability.

Added a presentation (PPT) for project demonstration.

📂 Project Structure
├── flood.csv                # Original dataset
├── flood_processed.csv      # Cleaned & processed dataset
├── eda.ipynb                # Exploratory Data Analysis
├── model_training.ipynb     # Model building & evaluation
├── streamlit_app.py         # Web application for predictions
├── requirements.txt         # Dependencies
├── README.md                # Project documentation
└── presentation.pptx        # Project presentation
