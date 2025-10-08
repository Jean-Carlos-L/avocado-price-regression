# avocado-price-regression

## Project Overview
This repository contains a data science project developed for academic purposes.  
The main objective is to predict avocado prices based on their characteristics using regression models.  

The project follows a complete data analysis workflow, including exploratory analysis, data cleaning, feature engineering, model training, and performance evaluation.  
In this specific case, the dataset focuses on avocado prices, and the main algorithm used is Linear Regression.

---

## Dataset
**Source:** [Avocado Prices - Kaggle](https://www.kaggle.com/datasets/neuromusic/avocado-prices/data)

The dataset provides historical information about avocado sales, including average prices, types, regions, and dates.  
It contains both categorical and numerical variables, which allows for comprehensive analysis and modeling.

Alternatively, any dataset from [Kaggle Datasets (Prices)](https://www.kaggle.com/datasets?search=prices) may be used, provided it meets the following conditions:
- Includes both categorical and numerical variables.  
- Contains a clearly defined target variable.  
- Has more than 1,000 records.

---

## Project Development

### 1. Descriptive Data Analysis
The first stage focuses on understanding the structure and distribution of the dataset.  
This includes:
- Loading and inspecting the dataset using **Pandas**.  
- Performing **Exploratory Data Analysis (EDA)** through visualizations created with **Matplotlib** and **Seaborn**.  
- Generating histograms, boxplots, and heatmaps to identify patterns, outliers, and correlations among variables.  
- Extracting relevant insights to guide the modeling process.

---

### 2. Data Cleaning and Normalization
This stage aims to prepare the dataset for modeling by ensuring data consistency and quality.  
Key steps include:
- Handling missing or inconsistent data using appropriate imputation strategies.  
- Detecting and managing outliers.  
- Encoding categorical variables into numerical format (e.g., one-hot encoding).  
- Standardizing and normalizing numerical variables to improve model performance.

---

### 3. Predictive Modeling
Once the dataset is clean and structured, multiple machine learning models are implemented and compared.  
Depending on the type of problem, different approaches are applied:

- **Regression models** (for continuous target variables):  
  - Linear Regression  
  - Random Forest Regressor  
  - Neural Networks  

- **Classification models** (for categorical target variables):  
  - K-Nearest Neighbors (KNN)  
  - Decision Tree  
  - Support Vector Machine (SVM)  

- **Clustering models** (for unsupervised segmentation):  
  - K-Means Clustering  
  - Evaluation using Silhouette Score  

Model performance is evaluated using appropriate metrics such as:
- Mean Squared Error (MSE)  
- Coefficient of Determination (R²)  
- Accuracy  
- F1-Score  

---

### 4. Conclusions and Discussion
In the final stage, the results obtained from the data analysis and modeling process are summarized.  
The performance of the predictive models is analyzed, and potential improvements are discussed.  
Visualizations and interpretations of the results are presented to support the conclusions.

---

## Technologies Used
- **Python 3.x**  
- **Pandas**, **NumPy** — Data manipulation and preprocessing  
- **Matplotlib**, **Seaborn** — Data visualization  
- **Scikit-learn** — Machine learning modeling and evaluation  
- **Jupyter Notebook** — Development environment  

---

## Expected Outcome
The project aims to develop a predictive model capable of estimating avocado prices based on relevant features.  
The analysis will provide insights into price behavior, influential variables, and model performance, demonstrating the practical application of machine learning techniques to real-world data.

---

## Author
**Jean Carlos Lerma Rojas**  
**Juan Carmilo Garcia Saenz**  
**Sebastina**  
Universidad del Valle — Data Science Project (2025)
