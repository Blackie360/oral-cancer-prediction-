# Oral Cancer Prediction – Top 30 Countries

## Project Overview
Oral cancer is a major global health concern, with significant variations in incidence and survival rates across different countries. This project focuses on analyzing oral cancer prediction data for the top 30 countries based on prevalence and risk factors. The dataset includes demographic details, lifestyle habits, medical history, and genetic predisposition, providing valuable insights for early detection and prevention strategies.

### Goals of the Project
- Identify key risk factors contributing to oral cancer.
- Develop predictive models to assess individual risk levels.
- Compare trends across different countries to understand regional variations.
- Apply machine learning techniques for classification and survival analysis.
- Assist healthcare professionals in early diagnosis and personalized treatment planning.

### Potential Use Cases
- **Exploratory Data Analysis (EDA):** Uncover patterns and correlations in oral cancer risk factors.
- **Risk Prediction Models:** Train ML models to estimate the likelihood of developing oral cancer.
- **Public Health Insights:** Support policymakers in designing targeted awareness and prevention programs.
- **Medical Research:** Enhance understanding of genetic and environmental influences on oral cancer.

---

## Import Dependencies
```python
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
```

---

## Dataset Import
```python
df = pd.read_csv("/kaggle/input/oral-cancer-prediction-dataset-top-30-countries/oral_cancer_prediction_dataset.csv")
df.head()
```

### Sample Data
| ID | Country   | Gender | Age | Tobacco_Use | Alcohol_Use | Socioeconomic_Status | Diagnosis_Stage | Treatment_Type | Survival_Rate | HPV_Related |
|----|-----------|--------|-----|-------------|-------------|-----------------------|-----------------|----------------|---------------|-------------|
| 1  | Ethiopia  | Male   | 34  | 1           | 1           | High                  | Early           | Radiotherapy   | 0.826235      | 0           |
| 2  | Turkey    | Female | 84  | 1           | 1           | High                  | Moderate        | Radiotherapy   | 0.376607      | 0           |

### Dataset Information
```python
df.info()
```
- **Rows:** 160,292
- **Columns:** 11
- **Memory Usage:** 13.5 MB

### Dataset Statistics
```python
df.describe()
```
| Column                | Count    | Mean      | Std       | Min  | 25%  | 50%  | 75%  | Max |
|-----------------------|----------|-----------|-----------|------|------|------|------|-----|
| Age                  | 160,292  | 46.56     | 20.59     | 20   | 29   | 39   | 64   | 89  |
| Tobacco_Use          | 160,292  | 0.60      | 0.49      | 0    | 0    | 1    | 1    | 1   |
| Alcohol_Use          | 160,292  | 0.49      | 0.50      | 0    | 0    | 0    | 1    | 1   |
| Survival_Rate        | 160,292  | 0.60      | 0.17      | 0.30 | 0.45 | 0.59 | 0.74 | 0.90|

---

## Exploratory Data Analysis (EDA)

### HPV-Related Cases Across Countries
```python
plt.figure(figsize=(12, 6))
hpv_counts = df.groupby("Country")["HPV_Related"].sum().reset_index()
hpv_counts = hpv_counts.sort_values(by="HPV_Related", ascending=False)
sns.heatmap(hpv_counts.set_index("Country"), annot=True, cmap="Reds", linewidths=0.5, fmt=".0f")
plt.title("HPV-Related Cases Across Countries")
plt.xlabel("Country")
plt.show()
```

### Survival Rate by Diagnosis Stage
```python
plt.figure(figsize=(8, 5))
sns.violinplot(x="Diagnosis_Stage", y="Survival_Rate", data=df, palette="coolwarm")
plt.xlabel("Diagnosis Stage")
plt.ylabel("Survival Rate")
plt.title("Survival Rate by Diagnosis Stage")
plt.show()
```

### Treatment Type Distribution
```python
plt.figure(figsize=(10, 5))
sns.countplot(y=df['Treatment_Type'], order=df['Treatment_Type'].value_counts().index, palette="viridis")
plt.xlabel("Number of Cases")
plt.ylabel("Treatment Type")
plt.title("Treatment Type Distribution")
plt.show()
```

### Impact of Tobacco & Alcohol Use
```python
plt.figure(figsize=(10, 5))
sns.histplot(data=df, x="Tobacco_Use", hue="Alcohol_Use", multiple="stack", palette="coolwarm")
plt.xlabel("Tobacco Use")
plt.ylabel("Count")
plt.title("Impact of Tobacco & Alcohol Use")
plt.show()
```

### Age Distribution of Patients
```python
plt.figure(figsize=(8, 5))
sns.histplot(df['Age'], bins=30, kde=True, color="darkblue")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Age Distribution of Patients")
plt.show()
```

### Gender Distribution
```python
plt.figure(figsize=(6, 6))
df['Gender'].value_counts().plot.pie(autopct="%1.1f%%", colors=["skyblue", "lightcoral"])
plt.title("Gender Distribution")
plt.ylabel("")
plt.show()
```

### Distribution of Oral Cancer Cases by Country
```python
plt.figure(figsize=(12, 5))
sns.countplot(y=df['Country'], order=df['Country'].value_counts().index, palette="Blues_r")
plt.xlabel("Number of Cases")
plt.ylabel("Country")
plt.title("Distribution of Oral Cancer Cases by Country")
plt.show()
```

### Cancer Diagnosis Stage Distribution
```python
plt.figure(figsize=(7, 5))
sns.countplot(x="Diagnosis_Stage", data=df, order=["Early", "Moderate", "Late"], palette="Set2")
plt.title("Cancer Diagnosis Stage Distribution")
plt.show()
```

### Survival Rate by Socioeconomic Status
```python
plt.figure(figsize=(7, 5))
sns.boxplot(x="Socioeconomic_Status", y="Survival_Rate", data=df, palette="husl")
plt.title("Survival Rate by Economic Status")
plt.show()
