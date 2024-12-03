#!/usr/bin/env python
# coding: utf-8

# Name : Komal Dodiya
# Students ID:
# 
# The goal of this assignment is to understand the logic and methods of exploratory data analysis (EDA).

# In[1]:


pip install ydata-profiling


# In[3]:


pip install sweetviz


# In[1]:


# Basic libraries
import pandas as pd
import numpy as np

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


# Profiling libraries


import sweetviz as sv


# In[9]:


# Load the dataset
df = pd.read_csv(r"C:\Users\nirma\Documents\Pace University\Intoduction to Data Science\Projects\Project-1\telco-customer-churn.csv")


# In[11]:


# Display first few rows
df.head()


# In[13]:


df.columns


# In[15]:


df.shape


# 2) List all data types

# In[17]:


df.info()


# 1) Check for missing value, Null, NaN records. Find Outliers. Transform all data to numeric. 

# In[19]:


# Check for missing values
df.isnull().sum()


# By Seeing the result we can conclude that, there no null values in any columns

# In[22]:


pd.isna(df)


# By seeing results we can conclude that, there are no Nan values

# In[25]:


# lets check for outlier

# Creating boxplots to detect outliers
plt.figure(figsize=(15, 5))  # Set the figure size for better readability

sns.boxplot(data=df, x=df['MonthlyCharges'])  # Boxplot for each feature
plt.title(f'Boxplot of MonthlyCharges (The amount charged to the customer monthly) ')  # Adding a title for clarity

plt.tight_layout()  # Adjust subplots to fit the figure area nicely


# In[27]:


plt.figure(figsize=(15, 5))  # Set the figure size for better readability

sns.boxplot(data=df, x=df['tenure'])  # Boxplot for each feature
plt.title(f'Boxplot of tenure(Number of months the customer has stayed with the company)')  # Adding a title for clarity

plt.tight_layout()  # Adjust subplots to fit the figure area nicely


# In[29]:


plt.figure(figsize=(15, 5))  # Set the figure size for better readability

sns.boxplot(data=df, x=df['TotalCharges'])  # Boxplot for each feature
plt.title(f'Boxplot of TotalCharges (The total amount charged to the customer)')  # Adding a title for clarity

plt.tight_layout()  # Adjust subplots to fit the figure area nicely


# By reviewing all these results we can see that there are no outliers

# In[31]:


# List the types of data before transformation
print("Data Types Before Transformation:")
print(df.dtypes)


# In[32]:


df_numeric = df.apply(pd.to_numeric, errors='coerce')
df_numeric = df_numeric.dropna(axis=1, how='all')


# In[36]:


df_numeric.head()


# In[38]:


corr = df_numeric.corr()
corr.style.background_gradient(cmap='coolwarm')


# In[40]:


# getting unique values from each column, so we know which values to convert to numerical
for col in df.columns:
    print(col,df[col].unique())


# We will assign each of these text values a number.
# We will use 0 and 1 to represent binary data (yes/no, male/female), and assign unique integer values to everything else.

# Rename the values such as
#     "Yes": 1, "No": 0,
#     "Female": 0, "Male": 1,
#     "No phone service": 2, "No internet service": 3,
#     "DSL": 4, "Fiber optic": 5,
#     "Month-to-month": 6, "One year": 7, "Two year": 8,
#     "Electronic check": 9, "Mailed check": 10,
#     "Bank transfer (automatic)": 11, "Credit card (automatic)": 12,
#     " ": -1  # for missing values
# 

# In[44]:


for col in df.columns:
    df.loc[df[col] == "No", col] = 0
    df.loc[df[col] == "Yes", col] = 1
    df.loc[df[col] == "Female", col] = 0
    df.loc[df[col] == "Male", col] = 1
    df.loc[df[col] == "No phone service", col] = 2
    df.loc[df[col] == "No internet service", col] = 3
    df.loc[df[col] == "DSL", col] = 4
    df.loc[df[col] == "Fiber optic", col] = 5
    df.loc[df[col] == "Month-to-month", col] = 6
    df.loc[df[col] == "One year", col] = 7
    df.loc[df[col] == "Two year", col] = 8
    df.loc[df[col] == "Electronic check", col] = 9
    df.loc[df[col] == "Mailed check", col] = 10
    df.loc[df[col] == "Bank transfer (automatic)", col] = 11
    df.loc[df[col] == "Credit card (automatic)", col] = 12
    df.loc[df[col] == " ", col] = -1


# In[46]:


# checking the unique values of our dataframe
for col in df.columns:
    print(col,df[col].unique())


# In[48]:


df.info()


# Upon reviewing the result we can see that, even after changing values data types have not changed yet, so we will change it now.

# In[51]:


cols_to_convert_to_int = df.columns[np.where(df.dtypes=="object")]
cols_to_convert_to_int = cols_to_convert_to_int[1:] # we are not conserding the customerID col because it has all the unique values 

# converting columns to integer type
for col in cols_to_convert_to_int:
    df[col] = pd.to_numeric(df[col])


# In[53]:


print("Altered datatypes:\n")
df.dtypes


# Here, we can see the updated datatypes.

# In[56]:


# getting correlation matrix of matrix
df = df.drop(columns=['customerID']) #Droping this column because it contains string values and its not that important
corr = df.corr()
corr.style.background_gradient('coolwarm')


# In[58]:


sns.heatmap(corr)


# 3) Perform EDA. Present dependencies and correlations among the various features in the data. 

# In[61]:


# generating a SweetViz report for the dataframe
analysis_eda = sv.analyze(df)
analysis_eda.show_html('Analysis_eda.html')


# In[62]:


import seaborn as sns
sns.violinplot(x='Churn', y='gender', data=df)


# Here we can see that churn ratio for "Male"=1 who churn/ do not churn and churn ration for "Female"=0 who churn/ do not churn is equal.  

# In[66]:


sns.violinplot(x='Churn',y='SeniorCitizen', data=df)


# Here we can see that there is a wide spread for non-SeniorCitizen who do not churn.

# In[69]:


sns.violinplot(x='Churn',y='Partner', data=df)


# Customers with partner churn less than customers with no partner churn. 

# In[72]:


sns.violinplot(x='Churn',y='Dependents', data=df)


# Customer with Dependent churn less than customer with no dependents.

# In[75]:


sns.violinplot(x='Churn',y='tenure', data=df)


# Customers with short tenure churn more.

# In[78]:


sns.violinplot(x='Churn',y='PhoneService', data=df)


# Slighty higher propertion of customers with PhoneService do not churn then customers with PhoneService.

# In[81]:


sns.violinplot(x='Churn',y='MultipleLines', data=df)


# Values for MultipleLines are 0="No", 1="Yes", 2="No Phone service". There is little more of a customers who do not churn for all the 3 options compared to customers who churn for all options.

# In[84]:


sns.violinplot(x='Churn',y='InternetService', data=df)


# Values of Internet Service are 0="No", 4="Dsl", 5="Fiber optic". Here we can see that a wide distribution of customers with fiber optic tend to churn the most among all 3 values.

# In[87]:


sns.violinplot(x='Churn',y='OnlineSecurity', data=df)


# Values for OnlineSecurity are 0="No", 1="Yes", 3="No Internet service". Customers with no online security churn more than other 2.

# In[90]:


sns.violinplot(x='Churn',y='OnlineBackup', data=df)


# Values for OnlineBackup are 0="No", 1="Yes", 3="No Internet service". Customer with no online backup churn more than other 2.

# In[93]:


sns.violinplot(x='Churn',y='DeviceProtection', data=df)


# Values for DeviceProtection are 0="No", 1="Yes", 3="No Internet service". Customer with no Device Protection churn more than other 2.

# In[96]:


sns.violinplot(x='Churn',y='TechSupport', data=df)


# Values for TechSupport are 0="No", 1="Yes", 3="No Internet service". Customer with no Tech Support churn more than other 2.

# In[99]:


sns.violinplot(x='Churn',y='StreamingTV', data=df)


# Values for StreamingTV are 0="No", 1="Yes", 3="No Internet service". Customers with no StreamingTv and StreamingTv churn more.

# In[102]:


sns.violinplot(x='Churn',y='StreamingMovies', data=df)


# Values for StreamingMovies are 0="No", 1="Yes", 3="No Internet service". Customers with no StreamingMovies and StreamingMovies churn more.

# In[105]:


sns.violinplot(x='Churn',y='Contract', data=df)


# Values for Contract 6="Month-to-month", 7="one year", and 8="two years". Customers with month-to-month contracts tend to churn the most.

# In[108]:


sns.violinplot(x='Churn',y='PaperlessBilling', data=df)


# There is a wide distribution for customer with paperlessBilling and they will churn.

# In[111]:


sns.violinplot(x='Churn',y='PaymentMethod', data=df)


# Values for PaymentMethod, 9="electronic check", 10="mailed check", 11="automatic bank transfer" and 12="automatic credit card". An even proportion of customers across all 4 payment types do not churn, but a higher proportion of customers who pay by electronic check tend to churn more than the other 3 payment types.

# In[114]:


sns.violinplot(x='Churn',y='MonthlyCharges', data=df)


# Here we can see that customers with lower charges each month churn less than customers with higher charges each month.

# In[117]:


sns.violinplot(x='Churn',y='TotalCharges', data=df)


# Customers with low total charges churn more than the customers with high total charges.

# 4) Split the dataset into training and test datasets (80/20 ratio).

# In[121]:


from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Churn'])


# In[123]:


train_df.shape


# In[125]:


test_df.shape


# In[127]:


# Compare training and test datasets on the target 'Churn'
report = sv.compare([train_df, "Training Set"], [test_df, "Test Set"], target_feat='Churn')

# Generate the report
report.show_html('train_test_comparison_report.html')


# 5) State limitations/issues (if any) with the given dataset.

# State Limitations/Issues with the Dataset:-
# 
# Imbalanced Target Feature: The 'Churn' feature might be imbalanced, which can affect the performance of classification models.
# 
# Outliers: Numerical features such as 'MonthlyCharges' and 'TotalCharges' may contain outliers that can skew analysis results.
# 
#  
# While performing the datatype conversion, if we go ahead with "to_numeric" function on object type it will be converted to NaN 
# as had text data in it. So we got 2 option, either use the labelEncoding or to manually fetch the unique values and assign them
# unique numerical value to differentiate it and then convert it into numeric datatype using "to_numeric" function. The only 
# problem with latter method is that it needs attention and manual work.

# Project - 2 Building Machine Learning Model

# In[150]:


# 1. Isolate X variables
X = df.drop(columns= ['Churn'])

# 2. Isolate y variable
y = df['Churn']

# 3. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.20, stratify = y , random_state=42)


# Now we will build the model

# In[142]:


pip install xgboost


# In[198]:


from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize models
models = {
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# Function to train, predict, and evaluate models
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred)
    }

# Evaluate each model and print results
results = {}
for model_name, model in models.items():
    results[model_name] = evaluate_model(model, X_train, y_train, X_test, y_test)
    print(f"Results for {model_name}:")
    for metric, score in results[model_name].items():
        print(f"  {metric}: {score:.4f}")
    print("\n")


# Now it time for Model Tunning

# Method - 1 :- GridSearch CV

# In[160]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Define model
model = RandomForestClassifier()

# Define parameters to search over
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)


# Method - 2 RandomizedSearch CV

# In[162]:


from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np

model = RandomForestClassifier()

# Define random search grid with distributions for sampling
param_dist = {
    'n_estimators': np.arange(50, 500, 50),
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': np.arange(2, 11)
}

# Set up RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=50, cv=5, scoring='accuracy')
random_search.fit(X_train, y_train)

# Get best parameters and score
print("Best Parameters:", random_search.best_params_)
print("Best Score:", random_search.best_score_)


# Now Applying SMOTE Technique and again feed the data to our models and tune the new models

# In[182]:


pip install SMOTE


# In[196]:


from imblearn.over_sampling import SMOTE

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Check the balance of classes after applying SMOTE
print("Class distribution before SMOTE:", y_train.value_counts())
print("Class distribution after SMOTE:", y_train_smote.value_counts())


# Building New Model and giving input After SMOTE Technique

# Now we will tune this new models using GridSearch CV

# In[200]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Define model
model = RandomForestClassifier()

# Define parameters to search over
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_smote, y_train_smote)

# Get best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)


# Now we will tune this new models using RandomizedSearch CV

# In[202]:


from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np

model = RandomForestClassifier()

# Define random search grid with distributions for sampling
param_dist = {
    'n_estimators': np.arange(50, 500, 50),
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': np.arange(2, 11)
}

# Set up RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=50, cv=5, scoring='accuracy')
random_search.fit(X_train_smote, y_train_smote)

# Get best parameters and score
print("Best Parameters:", random_search.best_params_)
print("Best Score:", random_search.best_score_)

