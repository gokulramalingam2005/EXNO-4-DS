# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:

import pandas as pd
from scipy import stats
import numpy as np

df=pd.read_csv("/content/bmi.csv")
df.head()

df.dropna()
![image](https://github.com/user-attachments/assets/21d44096-14b0-4def-b07a-1d6d32f4e702)

max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
df1=df
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])

df1.head(10)

from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])

df.head(10)
![image](https://github.com/user-attachments/assets/7a7c1db8-06a2-4be5-9dfe-2a13d4a5e76a)

from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df2=df
df2[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])

df2
![image](https://github.com/user-attachments/assets/1cc21e2a-0705-4f3a-a36e-6e155deea9e8)

df3=pd.read_csv("/content/bmi.csv")

from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])

df3

df4=pd.read_csv("/content/bmi.csv")

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df4[['Height','Weight']]=scaler.fit_transform(df4[['Height','Weight']])
df4.head()
![image](https://github.com/user-attachments/assets/b078ccb3-92dd-4679-aaad-3421e24b1377)

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV,Ridge,Lasso
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import chi2

df=pd.read_csv('/content/titanic_dataset (1).csv')
df.columns

df.shape

x = df.drop("Survived", axis=1)
y = df['Survived']

df1=df.drop(["Name","Sex","Ticket","Cabin","Embarked"],axis=1)

df1.columns

df1['Age'].isnull().sum()

df1['Age'].fillna(method='ffill')
![image](https://github.com/user-attachments/assets/3fb4a6fb-667b-4155-a340-531c3ea2c546)

df1['Age']=df1['Age'].fillna(method='ffill')

df1['Age'].isnull().sum()

feature=SelectKBest(mutual_info_classif,k=3)

df1.columns

cols=df1.columns.tolist()
cols[-1],cols[1]=cols[1],cols[-1]
df1=df1[cols]

df1.columns

X=df1.iloc[:,0:6]
y=df1.iloc[:,6]

X.columns

y=y.to_frame()

y.columns

"""Filter Method
Chi2 method
"""

data=pd.read_csv('/content/titanic_dataset (1).csv')

data=data.dropna()

X=data.drop(['Survived','Name','Ticket'],axis=1)
y=data['Survived']

X
![image](https://github.com/user-attachments/assets/4460abb5-6102-4a47-8a6f-92b302d7717b)


data["Sex"]=data["Sex"].astype("category")
data["Cabin"]=data["Cabin"].astype("category")
data["Embarked"]=data["Embarked"].astype("category")

data["Sex"]=data["Sex"].cat.codes
data["Cabin"]=data["Cabin"].cat.codes
data["Embarked"]=data["Embarked"].cat.codes

data

from sklearn.preprocessing import LabelEncoder

if 'gender' in X.columns:
    # Using Label Encoding
    le = LabelEncoder()
    X['gender'] = le.fit_transform(X['gender'])

# Alternatively, using One-Hot Encoding
X = pd.get_dummies(X)

# Now you can apply SelectKBest
k = 5
selector = SelectKBest(score_func=chi2, k=k)
X_new = selector.fit_transform(X, y)

selected_feature_indices=selector.get_support(indices=True)
selected_features=X.columns[selected_feature_indices]
print("Selected Features:")
print("selected_features")

X.info()

print(X.columns)

print(X.columns.tolist())
![image](https://github.com/user-attachments/assets/60e51158-0d00-4232-9532-b4659f304af0)


columns_to_drop = ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']
existing_columns_to_drop = [col for col in columns_to_drop if col in X.columns]
X = X.drop(columns=existing_columns_to_drop)

X = X.drop(columns=columns_to_drop, errors='ignore')

from sklearn.feature_selection import SelectKBest, f_regression

selector=SelectKBest(score_func=f_regression,k=5)
X_new=selector.fit_transform(X,y)

selected_feature_indices=selector.get_support(indices=True)
selected_features=X.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)

from scipy.stats import chi2_contingency

tips=sns.load_dataset('tips')
tips.head()

contingency_table=pd.crosstab(tips['sex'],tips['time'])

print(contingency_table)

chi2,p,_,_=chi2_contingency(contingency_table)

print(f"Statistic:{chi2}")
print(f"p-value:{p}")

import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif

# Create a sample dataset
data = {
    'Feature1': [1, 2, 3, 4, 5],
    'Feature2': ['A', 'B', 'C', 'A', 'B'],
    'Feature3': [0, 1, 1, 0, 1],
    'Target': [0, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

# Separate features and target
X = df[['Feature1', 'Feature3']]
y = df['Target']

# SelectKBest with mutual_info_classif for feature selection
selector = SelectKBest(score_func=mutual_info_classif, k=1)
X_new = selector.fit_transform(X, y)

# Get the selected feature indices
selected_feature_indices = selector.get_support(indices=True)

# Print the selected features
selected_features = X.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)




# RESULT:
 Feature Scaling and Feature Selection processes are successfully done for the given data
