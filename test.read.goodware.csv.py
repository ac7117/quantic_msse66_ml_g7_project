import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer, KNNImputer

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decision_tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


gw = pd.read_csv('goodware.csv', header=None)
bm = pd.read_csv('brazilian-malware.csv', header=None)

# gw.shape
# gw.head()
# gw.info()
# gw.describe()
# gw.columns

bm.shape
bm.head()
bm.info()
bm.describe()
bm.columns

bm.iloc[:, 10:16].head()

bm[11].value_counts()

bm.unique().shape
bm.isnull().sum()
bm.duplicated().sum()

# preprocessing / imputation

# Identify numeric and non-numeric columns
numeric_columns = bm.select_dtypes(include=[np.number]).columns
non_numeric_columns = bm.select_dtypes(exclude=[np.number]).columns

print(f"Numeric columns: {len(numeric_columns)}")
print(f"Non-numeric columns: {len(non_numeric_columns)}")
print(f"Non-numeric columns: {list(non_numeric_columns)}")

# Separate imputation for numeric and non-numeric data
# For numeric columns - use mean strategy
numeric_imputer = SimpleImputer(strategy='mean')
bm_numeric_imputed = pd.DataFrame(
    numeric_imputer.fit_transform(bm[numeric_columns]), 
    columns=numeric_columns,
    index=bm.index
)

# For non-numeric columns - use most_frequent strategy
categorical_imputer = SimpleImputer(strategy='most_frequent')
bm_categorical_imputed = pd.DataFrame(
    categorical_imputer.fit_transform(bm[non_numeric_columns]), 
    columns=non_numeric_columns,
    index=bm.index
)

# Combine both imputed datasets
bm_imputed = pd.concat([bm_numeric_imputed, bm_categorical_imputed], axis=1)

# Reorder columns to match original order
bm_imputed = bm_imputed[bm.columns]

print(f"Original missing values: {bm.isnull().sum().sum()}")
print(f"After imputation: {bm_imputed.isnull().sum().sum()}")




