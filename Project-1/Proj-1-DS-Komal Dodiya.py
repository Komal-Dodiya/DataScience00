#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import sweetviz as sv

# Load dataset
df = pd.read_csv("Desktop/PACE-FALL-DATA/Thursday-IntroDS/telco-customer-churn.csv")

# Preview the data
df.head()


# In[3]:


# 1.1 Check for missing values
print("Missing Values:\n", df.isnull().sum())

# 1.2 Check for NaN and infinite values
print("NaN Values:\n", df.isna().sum())
#print("Infinite Values:\n", np.isinf(df).sum())

# 1.3 Convert non-numeric columns to numeric using Label Encoding
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# 1.4 Check for outliers using the Interquartile Range (IQR)
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()

print("Outliers in each column:\n", outliers)


# In[4]:


# 2.1 List all column data types
print("Data Types:\n", df.dtypes)

# 2.2 Identify numeric and categorical columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

print("Numeric Columns:\n", numeric_cols)
print("Categorical Columns:\n", categorical_cols)


# In[5]:


# 3.1 Correlation matrix for numeric features
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# 3.2 Pairplot for key features against the target (assuming 'Churn' is the target)
sns.pairplot(df, hue='Churn', diag_kind='kde')  # Replace 'Churn' with your target column name
plt.show()

# 3.3 Feature Importance using Random Forest
X = df.drop(columns=['Churn'])  # Replace 'Churn' with your target column name
y = df['Churn']

# Fit a RandomForestClassifier to find feature importance
model = RandomForestClassifier()
model.fit(X, y)

# Get feature importance
feature_importance = pd.Series(model.feature_importances_, index=X.columns)
feature_importance.nlargest(10).plot(kind='barh')
plt.title('Top 10 Important Features')
plt.show()


# In[6]:


# 4.1 Split the dataset (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4.2 Create Sweetviz comparison report
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

report = sv.compare([train_data, "Training Data"], [test_data, "Test Data"])
report.show_html("sweetviz_comparison_report.html")  # Save report as HTML


# In[7]:


# 5.1 Check class imbalance in the target variable (assuming 'Churn' is the target)
class_counts = df['Churn'].value_counts()
print("Churn Class Distribution:\n", class_counts)

# 5.2 Check skewness of numeric features
numeric_skew = df[numeric_cols].skew()
print("Skewness in Numeric Columns:\n", numeric_skew)

# 5.3 Check for highly correlated features
high_corr = df.corr().abs().unstack().sort_values(ascending=False)
high_corr = high_corr[high_corr > 0.8]
print("Highly Correlated Features (Correlation > 0.8):\n", high_corr)


# In[8]:


#Conclusion
#Step 1 prepares the data by handling missing values, converting to numeric data, and detecting outliers.
#Step 2 lists all data types in the dataset.
#Step 3 performs EDA by showing the correlation matrix and determining feature importance.
#Step 4 splits the data and uses SweetViz to compare the training and testing sets.
#Step 5 identifies limitations such as class imbalance, skewness, and multicollinearity.#


# In[9]:


#Additional Visualization for EDA using Violin plots(Individual plots)
plt.figure(figsize=(10, 6))
sns.violinplot(x='Churn', y='InternetService', data=df)
plt.title('Churn vs InternetService')
plt.show()


# In[23]:


plt.figure(figsize=(10, 6))
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
plt.title('Churn vs MonthlyCharges')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Churn', y='PaperlessBilling', data=df)
plt.title('Churn vs PaperlessBilling')
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(x='Churn', y='Partner', data=df)
plt.title('Churn vs Partner')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Churn', y='Dependents', data=df)
plt.title('Churn vs Dependents')
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(x='Churn', y='StreamingTV', data=df)
plt.title('Churn vs StreamingTV')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Churn', y='StreamingMovies', data=df)
plt.title('Churn vs StreamingMovies')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Churn', y='DeviceProtection', data=df)
plt.title('Churn vs DeviceProtection')
plt.show()




# In[18]:


plt.figure(figsize=(10, 6))
sns.boxplot(x='Churn', y='MultipleLines', data=df)
plt.title('Churn vs MultipleLines')
plt.show()


# In[17]:


plt.figure(figsize=(12, 8))
sns.boxplot(x='Churn', y='SeniorCitizen', data=df)
plt.title('Churn vs SeniorCitizen')
plt.show()



# In[22]:


plt.figure(figsize=(10, 6))
sns.boxplot(x='Churn', y='PhoneService', data=df)
plt.title('Churn vs PhoneService')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Churn', y='gender', data=df)
plt.title('Churn vs Gender')
plt.show()


# In[25]:


plt.figure(figsize=(10, 6))
sns.boxplot(x='Churn', y='OnlineSecurity', data=df)
plt.title('Churn vs OnlineSecurity')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Churn', y='tenure', data=df)
plt.title('Churn vs Tenure')
plt.show()


# In[26]:


plt.figure(figsize=(10, 6))
sns.boxplot(x='Churn', y='OnlineBackup', data=df)
plt.title('Churn vs OnlineBackup')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Churn', y='TechSupport', data=df)
plt.title('Churn vs TechSupport')
plt.show()


# In[ ]:




