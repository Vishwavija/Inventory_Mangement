#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install xgboost


# In[2]:


## data manupalitation and handling 
import pandas as pd
import numpy as np

# Data visualizatio libraries
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sci

# Multicollinearity test and treatment Libraries
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA

## data preprocessing and EDA libraries 
from collections import OrderedDict 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

#Model Selection Libraries
from sklearn.model_selection import train_test_split , cross_val_score, GridSearchCV
 
#ML Models
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
import xgboost
from xgboost import XGBRegressor

#Model Evaluation Libraries
from sklearn.metrics import r2_score, mean_squared_error


# Warning filter library
import warnings
warnings.filterwarnings('ignore')


# ## loding the data frame

# In[3]:


df=pd.read_csv("Supplychain train dataset.csv")


# In[4]:


df.head()


# #### The Objective of this exercise is to build a model , using historical data that will  determine an optimim weight of the product to be  shipped each time from the respective warehouse.

# ### exploratory data analysis (EDA)

# In[5]:


df.info()


# observation from df.info
# 1. Some of  features  have  null values 
# 2. 22150 rows & 24 coloums 
# 3. feature have  int, float,object type data set 
# 4. Target  variable is product_wg_ton and rest are independent variable.

# In[6]:


df.describe()


# ##### Analysis from descriptive statistics
# 
# 1. There might be skewness in the data in the columns.
# 2. There might be chance of outliers if we compare Quartiles of some of the columns.(transport_issue,Competitor_in_mkt,  retail_shop_num,workers_num)
# 3. Since minimum and Q1 values are same for transport_issue, flood_impacted,electric_supply and temp_reg_mach we do not have outliers in the Lower Whisker region for them.
# 4. The range of values in flood_proof is 0 and 1  we can not be  consider this feature for outlier treatment.
# 

# ## Bulding a custom summary function for EDA report

# In[7]:


def custom_summary(my_df):
    result = []
    for col in my_df.columns:
        if my_df[col].dtypes != 'object':
            stats = OrderedDict({
                'Feature Name': col , 
                'Count': my_df[col].count() ,
                'Minimum': my_df[col].min() ,
                'Quartile1': my_df[col].quantile(.25) ,
                'Quartile2': my_df[col].quantile(.50) ,
                'Mean': my_df[col].mean() ,
                'Quartile 3': my_df[col].quantile(.75) ,
                'Maximum': my_df[col].max() ,
                'Variance': round(my_df[col].var()) ,
                'Standard Deviation': my_df[col].std() ,
                'Skewness': my_df[col].skew() , 
                'Kurtosis': my_df[col].kurt()
                })
            result.append(stats)
    result_df = pd.DataFrame(result)
    # skewness type
    skewness_label = []
    for i in result_df["Skewness"]:
        if i <= -1:
            skewness_label.append('Highly Negatively Skewed')
        elif -1 < i <= -0.5:
            skewness_label.append('Moderately Negatively Skewed')
        elif -0.5 < i < 0:
            skewness_label.append('Fairly Negatively Skewed')
        elif 0 <= i < 0.5:
            skewness_label.append('Fairly Positively Skewed')
        elif 0.5 <= i < 1:
            skewness_label.append('Moderately Positively Skewed')
        elif i >= 1:
            skewness_label.append('Highly Positively Skewed')
    result_df['Skewness Comment'] = skewness_label
    
    kurtosis_label=[]
    for i in result_df['Kurtosis']:
        if i >= 1:
            kurtosis_label.append('Leptokurtic Curve')
        elif i <= -1:
            kurtosis_label.append('Platykurtic Curve')
        else:
            kurtosis_label.append('Mesokurtic Curve')
    result_df['Kurtosis Comment'] = kurtosis_label
    Outliers_label = []
    for col in my_df.columns:
        if my_df[col].dtypes != 'object':
            Q1 = my_df[col].quantile(0.25)
            Q2 = my_df[col].quantile(0.5)
            Q3 = my_df[col].quantile(0.75)
            IQR = Q3 - Q1
            LW = Q1 - 1.5*IQR
            UW = Q3 + 1.5*IQR
            if len(my_df[(my_df[col] < LW) | (my_df[col] > UW)]) > 0:
                Outliers_label.append('Have Outliers')
            else:
                Outliers_label.append('No Outliers')
    result_df['Outlier Comment'] = Outliers_label

            
    return result_df


    

    


# In[8]:


custom_summary(df)


#  ##### Analysis from customs summary. 
#  
#  1. Feature having Mesokurtic curve implies the data points are moderate in distance from the mean so mean and SD are   moderate .
#  2. Feature having Leptokurtic curve  implies data points are closer to the mean.
#  3. Features with Platykurtic curve  implies the mean doesnt represent the whole data properly so SD is high.
#  
#  Overall Analysis from from customs summary Conclude that we can use the data for concluding results without Performing  Outlier treatment . 

# ### Data Pre-procesing and Cleaning For Model Building 

# In[9]:


## Droping Ware_house_id and Wh_manager_id as it's indivisual values.
df.drop(['Ware_house_ID',"WH_Manager_ID"],axis=1,inplace=True) 


# In[10]:


df.head()


# In[11]:


df.isnull().sum()


# ##### Observation 
# 
# 1. It's  observed  that workers_num , Wh_est_year  & approved_with_govt_certificate have missing values .
# 
# 2. for this data if we  will replace null values for workers_num with median Value of data set & for 
# approved_wh_govt_certificate replace null values with mode.
# 
# 3. For wh_est_year Since missing value is more then aproximately 50% we will perform linear regression model to predict the 
# missing vlue and will replace the null values with predicted values.
# 
# 

# In[12]:


## Replacing null values for approved_wh_gov_certificate with mode value of feature.
df['approved_wh_govt_certificate']=df['approved_wh_govt_certificate'].fillna(df['approved_wh_govt_certificate'].mode()[0])


# In[13]:


## Replacing null values for worker_num with median value of feature.
df['workers_num'] = df['workers_num'].fillna(df['workers_num'].median())


# In[14]:


df.isnull().sum()


# #### Model for predicting and replacing the same to null value for Wh_est_year
# 1. Since missing value is almost more then 50% we will be predicting the null value with LinearRegression model.

# In[15]:


correlation=df.corr('pearson')
correlation["wh_est_year"].sort_values(ascending=False)

x =df.copy()
def est_dataset(my_df, columns=[]):
  x = my_df.copy()
  x["idxs"] = np.where(x["wh_est_year"].isnull(), 1, 0)
  x_train = x[x["idxs"]==0]
  x_test = x[x["idxs"]==1]

  #Droping tcol from x_test
  x_test.drop("wh_est_year", axis=1,inplace=True)

  # Imputing values.
  x_test["wh_est_year"] = regression(x_train, x_test, "wh_est_year")

  #x_test.drop("wh_est_year", axis=1,inplace=True)

  '''
  #Label Encoding object dtype
  s = x_train.select_dtypes(include="object")
  for col in s.
# In[16]:


col=["storage_issue_reported_l3m","wh_breakdown_l3m","temp_reg_mach","wh_est_year"]


# In[17]:


x = df[col]
x["idxs"] = np.where(x["wh_est_year"].isnull(), 1, 0)
x_train = x[x["idxs"]==0]
x_test = x[x["idxs"]==1]

x_test.drop("wh_est_year", axis=1,inplace=True)


# In[18]:


mo = LinearRegression()
mo.fit(x_train.drop("wh_est_year", axis=1), x_train["wh_est_year"])
y_pred = mo.predict(x_test)


# In[19]:


y_pred


# In[20]:


x_test["wh_est_year"] = np.ceil(y_pred)


# In[21]:


x_final = pd.concat([x_train, x_test], axis=0)


# In[22]:


df["wh_est_year"] = x_final["wh_est_year"]


# In[23]:


df.isnull().sum()


# In[24]:


df.head()


# ## ODT( outlier Detection  technique) 
# 
# ###### Using IQR Methode 

# In[25]:


import numpy as np

# Select the continuous variables
continuous_columns = ['num_refill_req_l3m', 'transport_issue_l1y', 'Competitor_in_mkt',
                      'retail_shop_num', 'distributor_num', 'flood_impacted',
                      'flood_proof', 'electric_supply', 'dist_from_hub',
                      'workers_num', 'wh_est_year', 'storage_issue_reported_l3m',
                      'temp_reg_mach', 'wh_breakdown_l3m', 'govt_check_l3m', 'product_wg_ton'
                      ]

# Apply IQR method to treat outliers
df_outliers_treated = df.copy()  # Create a copy of the original dataset

for column in continuous_columns:
    Q1 = np.percentile(df[column], 25)
    Q3 = np.percentile(df[column], 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - (1.5 * IQR)
    upper_bound = Q3 + (1.5 * IQR)

    df_outliers_treated[column] = np.where((df[column] < lower_bound) | (df[column] > upper_bound),
                                             df[column].median(), df[column])

# The outliers have been treated in the `data_outliers_treated` DataFrame

# Assuming `data` is your original DataFrame and `data_outliers_treated` is the DataFrame with treated outliers

for column in continuous_columns:
    df[column] = df_outliers_treated[column]


# ### Encoding 
# 
# Transforming  the categorical values of the relevant features into numerical ones.

# In[26]:


def transform_Variable1(x):
    if x == 'Small':
        return 0
    else:
        return 1

def transform_Variable2(x):
    if x == 'Zone 1':
        return 0
    elif x == 'Zone 2' or x == 'Zone 3':
        return 1
    elif x == 'Zone 4':
        return 2
    elif x == 'Zone 5':
        return 3
    else:
        return 4

def transform_Variable3(x):
    if x == 'A':
        return 0
    elif x == 'A+':
        return 1
    elif x == 'B' or x == 'B+':
        return 2
    else:
        return 3

df.WH_capacity_size = df.WH_capacity_size.apply(transform_Variable1)
df.WH_regional_zone = df.WH_regional_zone.apply(transform_Variable2)
df.approved_wh_govt_certificate = df.approved_wh_govt_certificate.apply(transform_Variable3)


# In[27]:


df.head()


# ###### Using LabelEncoder to convert  categorical values of the relevant features into numerical ones

# In[ ]:





# In[28]:


label_encoder = LabelEncoder()
df["Location_type"] = label_encoder.fit_transform(df["Location_type"])


# In[29]:


label_encoder = LabelEncoder()
df["WH_regional_zone"] = label_encoder.fit_transform(df["WH_regional_zone"])


# In[30]:


label_encoder = LabelEncoder()
df["zone"] = label_encoder.fit_transform(df["zone"])


# In[31]:


label_encoder = LabelEncoder()
df["wh_owner_type"] = label_encoder.fit_transform(df["wh_owner_type"])


# In[32]:


df.head()


# ### Feature Engineering (importing new feature such as Age,Issue_Reported,and Infrastructure of Wh)

# In[33]:


df['No_of_year_old_Wh'] = pd.Timestamp.now().year - df['wh_est_year']
df['Issue_Reported'] = (df['transport_issue_l1y'].astype(bool) |
                   df['storage_issue_reported_l3m'].astype(bool) |
                   df['wh_breakdown_l3m'].astype(bool))
df['Infrastructure'] = (df['flood_proof'].astype(bool) |
                   df['electric_supply'].astype(bool) |
                   df['temp_reg_mach'].astype(bool))


# In[34]:


df.head(10)


# ##### Plots of feature Engineering 

# In[35]:


sns.countplot(data=df, x='No_of_year_old_Wh',hue='Location_type')
plt.xlabel('Age_wh')
plt.ylabel('No_of_wh')
plt.title('WH_AGE_AS_PER_LOCATION')
plt.show()


# In[36]:


sns.countplot(data=df, x='Infrastructure',hue='Issue_Reported')
plt.xlabel('Infrastructure')
plt.ylabel('No_of_wh')
plt.title('Issue_Reported_as_per_Infrastructure')
plt.show()


# In[37]:


sns.countplot(data=df, x='Issue_Reported',hue='Location_type')
plt.xlabel('Issue_Reported')
plt.ylabel('No_of_wh')
plt.title('Issue_Reported_as_per_Location')
plt.show()


# ###### Using LabelEncoder to convert  categorical values of the relevant features into numerical ones

# In[38]:


from sklearn.preprocessing import LabelEncoder

# Select the variables to be label encoded
variables = ['Issue_Reported', 'Infrastructure']

# Create an instance of LabelEncoder
label_encoder = LabelEncoder()

# Apply label encoding to the variables in the data_encoded DataFrame
for variable in variables:
    df[variable] = label_encoder.fit_transform(df[variable])


# #### ODT( outlier Detection  technique) 
# 
# ###### Using IQR Methode For added feature engineering Features. 

# In[39]:


import numpy as np

# Specify the columns with the newly created features
columns_with_outliers = ['No_of_year_old_Wh', 'Issue_Reported',
                         'Infrastructure']

# Loop through the columns and perform outlier treatment
for column in columns_with_outliers:
    # Calculate the IQR
    Q1 = np.percentile(df[column], 25)
    Q3 = np.percentile(df[column], 75)
    IQR = Q3 - Q1

    # Define the outlier boundaries
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers
    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)

    # Print the number of outliers
    print(f"Number of outliers in {column}: {outliers.sum()}")

    # Replace outliers with the median value
    df.loc[outliers, column] = df[column].median()


# In[40]:


df.head()


# #### Spliting Data into Train and Test Data set  for Model Building and Evaluation  

# In[41]:


from sklearn.model_selection import train_test_split

# Split the dataset into features (X) and target variable (y)
X = df.drop(columns=['product_wg_ton'])
y = df['product_wg_ton']

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Print the shapes of the train and test sets
print("Training set shape:", X_train.shape, y_train.shape)
print("Test set shape:", X_test.shape, y_test.shape)


# ##### Correlation of target feature with independent feature .

# In[ ]:





# In[42]:


def correlation_with_target(df, t_col):
    independent_variable = df.drop(t_col , axis = 1).columns
    corr_result = []
    for col in independent_variable:
        corr_result.append(df[t_col].corr(df[col]))
    result = pd.DataFrame([independent_variable , corr_result], index = ['Independent vairbales' , 'Correlation']).T
    return result.sort_values('Correlation' , ascending = False)
    


# In[43]:


correlation_with_target(df, 'product_wg_ton')


# ##### From  above Observation it is concluded that storage_issue_reported has highest  & num_refill_req has lowest  
# ##### correlation with target feature.

# ### Model Building 

# In[44]:


def model_builder(model_name, model, df, t_col):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    result = [model_name, rmse, r2]
    return result


# In[45]:


def multiple_models(df, t_col):
    col_names = ['Model Name' , 'RMSE' , 'R2 Score']
    result = pd.DataFrame(columns = col_names)
    result.loc[len(result)] = model_builder('LinearRegression' , LinearRegression() , df , t_col)
    result.loc[len(result)] = model_builder('Lasso' , Lasso() , df , t_col)
    result.loc[len(result)] = model_builder('Ridge' , Ridge() , df , t_col)
    result.loc[len(result)] = model_builder('DTR' , DecisionTreeRegressor() , df , t_col)
    result.loc[len(result)] = model_builder('SVR' , SVR() , df , t_col)
    result.loc[len(result)] = model_builder('Random Forest' , RandomForestRegressor() , df , t_col)
    result.loc[len(result)] = model_builder('KNN' , KNeighborsRegressor() , df , t_col)
    result.loc[len(result)] = model_builder('Gboost' , GradientBoostingRegressor() , df , t_col)
    result.loc[len(result)] = model_builder('XGboost' , XGBRegressor() , df , t_col)
    result.loc[len(result)] = model_builder('Adaboost' , AdaBoostRegressor() , df , t_col)

    return result.sort_values('R2 Score' , ascending = False)
    


# In[46]:


multiple_models(df , 'product_wg_ton')


# #### From  above  Accuracy Score  it  is  concluded  that  Gboost   has  highest  R2 _score ,  then  XGboost  and  third  one is Random_Forest  We  will  use  these  three  model  to  solve  the  Problem  Statement.

# In[47]:


##  GradientBoostingRegressor


# In[48]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Create a Gradient Boosting regressor
model = GradientBoostingRegressor()

# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate evaluation metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Get feature importances
feature_importances = model.feature_importances_

# Create a DataFrame with feature names and their importances
feature_importances_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})

# Sort the features by importance in descending order
top_features = feature_importances_df.nlargest(3, 'Importance')

# Print the top 3 features
print(top_features)


# Print the evaluation metrics
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared:", r2)


# In[49]:


import matplotlib.pyplot as plt

# Plotting the actual values and predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, c='green', alpha=0.5, label='Actual vs. Predicted')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Ideal')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Gradient Boosting Regression: Actual vs. Predicted")
plt.legend()
plt.show()


# In[50]:


## XGBRegressor


# In[51]:


import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Create an XGBoost regressor
model = xgb.XGBRegressor()

# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate evaluation metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Get feature importances
feature_importances = model.feature_importances_

# Create a DataFrame with feature names and their importances
feature_importances_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})

# Sort the features by importance in descending order
top_features = feature_importances_df.nlargest(3, 'Importance')

# Print the top 3 features
print(top_features)

# Print the evaluation metrics
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared:", r2)


# In[52]:


import matplotlib.pyplot as plt

# Plotting the actual values and predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, c='yellow', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("XGBoost Regression: Actual vs. Predicted")
plt.show()


# In[53]:


## RandomForestRegressor


# In[54]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Create a Random Forest regressor
model = RandomForestRegressor()

# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate evaluation metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared:", r2)


# In[55]:


import matplotlib.pyplot as plt

# Plotting the actual values and predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, c='orange', alpha=0.5, label='Actual vs. Predicted')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Ideal')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Random Forest Regression: Actual vs. Predicted")
plt.legend()
plt.show()


# In[56]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Create a Gradient Boosting regressor
model = GradientBoostingRegressor()

# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate evaluation metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Get feature importances
feature_importances = model.feature_importances_

# Create a DataFrame with feature names and their importances
feature_importances_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})

# Sort the features by importance in descending order
top_features = feature_importances_df.nlargest(3, 'Importance')

# Print the top 3 features
print(top_features)


# Print the evaluation metrics
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared:", r2)


# 

# In[57]:


df=pd.read_csv("Supply chain test dataset.csv")


# In[58]:


df.head()


# In[59]:


df.info()


# In[60]:


df.describe()


# In[61]:


df.isnull().sum()


# In[62]:


custom_summary(df)


# In[63]:


df.drop(['Ware_house_ID',"WH_Manager_ID"],axis=1,inplace=True)


# In[64]:


df['approved_wh_govt_certificate']=df['approved_wh_govt_certificate'].fillna(df['approved_wh_govt_certificate'].mode()[0])


# In[65]:


df['workers_num'] = df['workers_num'].fillna(df['workers_num'].median())


# In[66]:


df.isnull().sum()


# In[67]:


correlation=df.corr('pearson')
correlation["wh_est_year"].sort_values(ascending=False)

x =df.copy()
def est_dataset(my_df, columns=[]):
  x = my_df.copy()
  x["idxs"] = np.where(x["wh_est_year"].isnull(), 1, 0)
  x_train = x[x["idxs"]==0]
  x_test = x[x["idxs"]==1]

  #Droping tcol from x_test
  x_test.drop("wh_est_year", axis=1,inplace=True)

  # Imputing values.
  x_test["wh_est_year"] = regression(x_train, x_test, "wh_est_year")

  #x_test.drop("wh_est_year", axis=1,inplace=True)

  '''
  #Label Encoding object dtype
  s = x_train.select_dtypes(include="object")
  for col in s.
# In[68]:


col=["storage_issue_reported_l3m","wh_breakdown_l3m","temp_reg_mach","wh_est_year"]


# In[69]:


x = df[col]
x["idxs"] = np.where(x["wh_est_year"].isnull(), 1, 0)
x_train = x[x["idxs"]==0]
x_test = x[x["idxs"]==1]

x_test.drop("wh_est_year", axis=1,inplace=True)


# In[70]:


mo = LinearRegression()
mo.fit(x_train.drop("wh_est_year", axis=1), x_train["wh_est_year"])
y_pred = mo.predict(x_test)


# In[71]:


y_pred


# In[72]:


x_test["wh_est_year"] = np.ceil(y_pred)


# In[73]:


x_final = pd.concat([x_train, x_test], axis=0)


# In[74]:


df["wh_est_year"] = x_final["wh_est_year"]


# In[75]:


df.isnull().sum()


# In[76]:


df.head()


# In[77]:


import numpy as np

# Select the continuous variables
continuous_columns = ['num_refill_req_l3m', 'transport_issue_l1y', 'Competitor_in_mkt',
                      'retail_shop_num', 'distributor_num', 'flood_impacted',
                      'flood_proof', 'electric_supply', 'dist_from_hub',
                      'workers_num', 'wh_est_year', 'storage_issue_reported_l3m',
                      'temp_reg_mach', 'wh_breakdown_l3m', 'govt_check_l3m'
                      ]

# Apply IQR method to treat outliers
df_outliers_treated = df.copy()  # Create a copy of the original dataset

for column in continuous_columns:
    Q1 = np.percentile(df[column], 25)
    Q3 = np.percentile(df[column], 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - (1.5 * IQR)
    upper_bound = Q3 + (1.5 * IQR)

    df_outliers_treated[column] = np.where((df[column] < lower_bound) | (df[column] > upper_bound),
                                             df[column].median(), df[column])

# The outliers have been treated in the `data_outliers_treated` DataFrame

# Assuming `data` is your original DataFrame and `data_outliers_treated` is the DataFrame with treated outliers

for column in continuous_columns:
    df[column] = df_outliers_treated[column]


# In[78]:


def transform_Variable1(x):
    if x == 'Small':
        return 0
    else:
        return 1

def transform_Variable2(x):
    if x == 'Zone 1':
        return 0
    elif x == 'Zone 2' or x == 'Zone 3':
        return 1
    elif x == 'Zone 4':
        return 2
    elif x == 'Zone 5':
        return 3
    else:
        return 4

def transform_Variable3(x):
    if x == 'A':
        return 0
    elif x == 'A+':
        return 1
    elif x == 'B' or x == 'B+':
        return 2
    else:
        return 3

df.WH_capacity_size = df.WH_capacity_size.apply(transform_Variable1)
df.WH_regional_zone = df.WH_regional_zone.apply(transform_Variable2)
df.approved_wh_govt_certificate = df.approved_wh_govt_certificate.apply(transform_Variable3)


# In[79]:


df['WH_capacity_size'].value_counts()


# In[80]:


df.head()


# In[81]:


label_encoder = LabelEncoder()
df["Location_type"] = label_encoder.fit_transform(df["Location_type"])


# In[82]:


label_encoder = LabelEncoder()
df["WH_regional_zone"] = label_encoder.fit_transform(df["WH_regional_zone"])


# In[83]:


label_encoder = LabelEncoder()
df["zone"] = label_encoder.fit_transform(df["zone"])


# In[84]:


label_encoder = LabelEncoder()
df["wh_owner_type"] = label_encoder.fit_transform(df["wh_owner_type"])


# In[85]:


df.head()


# In[86]:


df['No_of_year_old_Wh'] = pd.Timestamp.now().year - df['wh_est_year']
df['Issue_Reported'] = (df['transport_issue_l1y'].astype(bool) |
                   df['storage_issue_reported_l3m'].astype(bool) |
                   df['wh_breakdown_l3m'].astype(bool))
df['Infrastructure'] = (df['flood_proof'].astype(bool) |
                   df['electric_supply'].astype(bool) |
                   df['temp_reg_mach'].astype(bool))


# In[87]:


from sklearn.preprocessing import LabelEncoder

# Select the variables to be label encoded
variables = ['Issue_Reported', 'Infrastructure']

# Create an instance of LabelEncoder
label_encoder = LabelEncoder()

# Apply label encoding to the variables in the data_encoded DataFrame
for variable in variables:
    df[variable] = label_encoder.fit_transform(df[variable])


# In[88]:


import numpy as np

# Specify the columns with the newly created features
columns_with_outliers = ['No_of_year_old_Wh', 'Issue_Reported',
                         'Infrastructure']

# Loop through the columns and perform outlier treatment
for column in columns_with_outliers:
    # Calculate the IQR
    Q1 = np.percentile(df[column], 25)
    Q3 = np.percentile(df[column], 75)
    IQR = Q3 - Q1

    # Define the outlier boundaries
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers
    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)

    # Print the number of outliers
    print(f"Number of outliers in {column}: {outliers.sum()}")

    # Replace outliers with the median value
    df.loc[outliers, column] = df[column].median()


# In[89]:


df.head()


# In[90]:


predicted_values = model.predict(df)


# In[91]:


predicted_values


# In[ ]:


import csv

def save_values_to_csv(values, filename):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Predicted Values'])  # Optional: Add a header row
        writer.writerows([[value] for value in values])

# Example usage:
predicted_values = model.predict(df)
save_values_to_csv(predicted_values, 'predicted_values.csv')


# ### Completed M4

# In[ ]:




