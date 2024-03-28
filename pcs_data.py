#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 13:33:38 2023

@author: srilu
"""

"""
Data Pre-Processing

1.	Identify other datasets and Merge the files into one dataset
2.	Provide the summary descriptive statistics, graphics such as histograms/plots
a)	For nominal data, show the number of distinct values and frequency of each nominal value
b)	For ordinal data, show the distinct values and their frequencies 
3.	Removal of obviously irrelevant data/columns
4.	Null Processing
5.	Correlation Analysis for nominal data
6.	Converting nominal/ordinal values to numerical values  
7.	Feature Selection using Feature Importances using the Random Forest Model
8.	PCA (Primary Component Analysis)
9.	Naive Random Over-Sampling	
10. Random Forest Classification Model
11. Decision Tree Classification Model
12. Gradient Boosting Classification Model
13. Logistic Regression Model
14. Naive Bayesian Classification Model

"""

# ====================================
# Importing the relevant libaries
# ====================================

import pandas as pd, numpy as np
from scipy.stats import chi2_contingency
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics
from imblearn.over_sampling import RandomOverSampler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import time
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix
from sklearn.svm import LinearSVC


# ========================================================
# Loading the PCS 2019 dataset into a pandas DataFrame
# ========================================================

df_pcs = pd.read_csv('Patient_Characteristics_Survey__PCS___2019.csv')
print(df_pcs.shape)
print(df_pcs.info())  # Get the null count & data type of each attribute
print(df_pcs.nunique()) # Get the number of unique values in each attribute


# ==================================================================
# 1. Identify other datasets and Merge the files into one dataset
# ==================================================================

# Merging the population variable using zip code
df_pop = pd.read_csv('uszips.csv')
df_pop.info()

# Extracting first 3 digits of zip code
df_pop['Three Digit Residence Zip Code'] = df_pop['zip'] // 100  

# Filtering New York data
df_pop = df_pop[(df_pop['state_id'] == 'NY') & (df_pop['Three Digit Residence Zip Code'] > 99)] 

# Selecting only zip code and Population columns from the dataset 
df_pop = df_pop[['Three Digit Residence Zip Code', 'population']].groupby('Three Digit Residence Zip Code').sum() 

df1_merged = pd.merge(df_pcs, df_pop, on='Three Digit Residence Zip Code', how='left')


# Merging the hospital count per zip code variable
df_hsptl = pd.read_csv('Entity_Hospitals_Q1_2023.csv')
df_hsptl.info()

# Extracting first 3 digits of zip code
df_hsptl['Three Digit Residence Zip Code'] = df_hsptl['Zipcode'] // 100  

# Calculate the unique hospital count per zip code
hsptl_count = df_hsptl.groupby(['Three Digit Residence Zip Code'])['Name'].nunique().reset_index()  

# Renaming the column Name to Unique hospital Count 
hsptl_count.rename(columns={'Name': 'Unique Hospital Count'}, inplace=True)

# Selecting only zip code and Hospital count columns from the dataset 
hsptl_count = hsptl_count[['Three Digit Residence Zip Code', 'Unique Hospital Count']] 

df2_merged = pd.merge(df1_merged, hsptl_count, on='Three Digit Residence Zip Code', how='left')


# Merging the Park dataset using zipcode 
df_park = pd.read_csv('park_zip.csv', encoding='ISO-8859-1')
df_park.info() 

# Extracting first 3 digits of zip code
df_park['Three Digit Residence Zip Code'] = df_park['Zipcode'] // 100

# Calculate the unique park count per zip code 
park_count = df_park.groupby(['Three Digit Residence Zip Code'])['Name'].nunique().reset_index()   
 
# Renaming the column Name to Unique Park Count 
park_count.rename(columns={'Name': 'Unique Park Count'}, inplace=True) 
 
# Selecting only zip code and Park count columns from the dataset 
park_count = park_count[['Three Digit Residence Zip Code', 'Unique Park Count']]  

df3_merged = pd.merge(df2_merged, park_count, on='Three Digit Residence Zip Code', how='left') 

# Merging the Income dataset using Zip code 
df_income = pd.read_csv('Income.csv')
df_income.info()

# Extracting first 3 digits of zip code
df_income['Three Digit Residence Zip Code'] = df_income['ZIPCODE'] // 100

# Filtering New York data
df_income = df_income[(df_income['STATE'] == 'NY') & (df_income['Three Digit Residence Zip Code'] != 999)] 

# Calculate the median household income per zip code 
median_income = df_income.groupby(['Three Digit Residence Zip Code'])['A00100'].median().reset_index()  
 
# Renaming the column Name to Median Household Income
median_income.rename(columns={'A00100': 'Median Household Income'}, inplace=True)

# Selecting only zip code and Median Household Income columns from the dataset 
median_income = median_income[['Three Digit Residence Zip Code', 'Median Household Income']]

df_merged = pd.merge(df3_merged, median_income, on='Three Digit Residence Zip Code', how='left') 
# df_merged.to_csv('PCSdata_merged.csv', index=False)   
 


# ========================================================================================
# 2. Show the number of distinct values and frequency of each nominal and ordinal values
# ========================================================================================

print('Distinct values and their frequencies of each nominal and ordinal value\n' )

for col in df_merged.columns:
    print(f"Column Name: {col}")

    max_length = df_merged[col].astype(str).str.len().max()
    print(f"Max Length: {max_length}")
    
    distinct_count = df_merged[col].nunique()
    print(f"Distinct Count: {distinct_count}")
    
    value_counts = df_merged[col].value_counts()
    print(f"Frequencies:\n{value_counts}")
    
    print("\n")
 


# =======================================================
# 3. Removal of obviously irrelevant data/columns
# =======================================================

print('Irrelevant Columns:\n 1) Survey Year\n 2) Three Digit Residence Zip Code\n 3) Number Of Hours Worked Each Week \n')
    
df_merged.drop(columns=['Survey Year', 'Number Of Hours Worked Each Week','Three Digit Residence Zip Code'], inplace=True)



# =========================
# 4. Processing Null Values
# =========================

print('Processing null values \n')
df_merged = df_merged[df_merged['Age Group'] != "UNKNOWN"]  #dropping 80 rows with that are neither Adult nor child
df_merged = df_merged[df_merged.Sex != "UNKNOWN"]  #droping 395 rows with that are neither Male nor Female
df_merged = df_merged[df_merged['Hispanic Ethnicity'] != "UNKNOWN"]  #dropping 5,965 rows with unknown hispanic ethnicity 

#Replacing the text with yes or no
df_merged['Transgender'] = df_merged['Transgender'].replace({'YES, TRANSGENDER': 'YES',\
                                                       'NO, NOT TRANSGENDER': 'NO',\
                                                       "CLIENT DIDN'T ANSWER": 'NOT SHARED',\
                                                       'UNKNOWN':'NOT SHARED'})

#Replacing Unknown and Client did not answer with OTHER
df_merged['Sexual Orientation'] = df_merged['Sexual Orientation'].replace({'UNKNOWN': 'OTHER',\
                                                                     "CLIENT DID NOT ANSWER": 'OTHER'})

#Replacing the Unknown Race with OTHER
df_merged['Race'] = df_merged['Race'].replace({'UNKNOWN RACE':'OTHER'})

#Replacing Unknown living condition  with OTHER
df_merged['Living Situation'] = df_merged['Living Situation'].replace({'UNKNOWN': 'OTHER LIVING SITUATION'})

# Replace "unknown" with "cohabitates with other" where "Living situation" is "private residence"
df_merged.loc[(df_merged['Living Situation'] == 'PRIVATE RESIDENCE') & \
           (df_merged['Household Composition'] == 'UNKNOWN'),\
           'Household Composition'] = 'COHABITATES WITH OTHERS'

# Replace the remaining "unknown" with "not applicable"
df_merged.loc[df_pcs['Household Composition'] == 'UNKNOWN', 'Household Composition'] = 'NOT APPLICABLE'

#Replacing Unknown Prefered language with OTHER
df_merged['Preferred Language'] = df_merged['Preferred Language'].replace({'UNKNOWN': 'ALL OTHER LANGUAGES'})

#Replacing Data not available with Religion nt shared
df_merged['Religious Preference'] = df_merged['Religious Preference'].replace({'DATA NOT AVAILABLE': 'RELIGION NOT SHARED'})

#Replacing Unknown veteran satus with NO
df_merged['Veteran Status'] = df_merged['Veteran Status'].replace({'UNKNOWN': 'NO'})

#Replacing Unemployed-looking for work and not looking for work with Unemployed & Non-paid/Volunteer with unknown employment status 
df_merged['Employment Status'] = df_merged['Employment Status'].replace({'UNEMPLOYED, LOOKING FOR WORK':'UNEMPLOYED',\
                                                                   'NOT IN LABOR FORCE:UNEMPLOYED AND NOT LOOKING FOR WORK':'UNEMPLOYED',\
                                                                   'NON-PAID/VOLUNTEER': 'UNKNOWN EMPLOYMENT STATUS'})

#Replacing No formal education, other & unknown status with Unknown education
df_merged['Education Status'] = df_merged['Education Status'].replace({'NO FORMAL EDUCATION': 'UNKNOWN EDUCATION',\
                                                                 'OTHER': 'UNKNOWN EDUCATION',\
                                                                 'UNKNOWN': 'UNKNOWN EDUCATION'})

#Replacing Unknown status with NO
df_merged['Special Education Services'] = df_merged['Special Education Services'].replace({'NOT APPLICABLE': 'NO', 'UNKNOWN': 'NO'})

#Replacing Unknown status with NOT MI - Other
df_merged['Principal Diagnosis Class'] = df_merged['Principal Diagnosis Class'].replace({'UNKNOWN': 'NOT MI - OTHER'})

#Replacing Unknown status with NOT MI - Other
df_merged['Additional Diagnosis Class'] = df_merged['Additional Diagnosis Class'].replace({'UNKNOWN': 'NOT MI - OTHER'})

#Replacing Not Applicable status with NO
df_merged['Medicaid Managed Insurance'] = df_merged['Medicaid Managed Insurance'].replace({'NOT APPLICABLE': 'NO'})

#Replacing Null values with 0 in 'Population', 'Hospital Count', 'Park Count', and 'Median Income' columns
df_merged['population'] = df_merged['population'].fillna(0)
df_merged['Unique Hospital Count'] = df_merged['Unique Hospital Count'].fillna(0)
df_merged['Unique Park Count'] = df_merged['Unique Park Count'].fillna(0)
df_merged['Median Household Income'] = df_merged['Median Household Income'].fillna(0)


# ================================================================================
# 5. Correlation Analysis for nominal data using Cramer's V and chi-square values
# ================================================================================
print('Correlation Analysis \n')

# This function generates the Cramer's V value
def cramer_v(x, y):
    n = len(x)
    ct = pd.crosstab(x, y) # crosstab
    chi2 = chi2_contingency(ct)[0]
    v = np.sqrt(chi2 / (n * (np.min(ct.shape) - 1)))
    return v

# This function returns a dataframe with Cramer's V values.
def cramer_values (df):
    '''Parameters:DataFrame; Returns: DataFrame
    Takes a DataFrame with nominal attributes and returns a DataFrame with
    Cramer's V values between all pairs of those attributes
    Required libraries:
        import pandas as pd, numpy as np
        from scipy.stats import chi2_contingency'''
    
    cramer_table = pd.DataFrame(columns=['col1','col2','Cramers V'])
    for i in df.columns:
        for j in df.columns:
            if i != j:
                v = cramer_v(df[i],df[j])
                row = pd.DataFrame({'col1':[i],'col2':[j],'Cramers V':[v]})
                cramer_table = pd.concat([cramer_table, row], ignore_index=True)
    return cramer_table.sort_values(by=['Cramers V'],ascending=False)

# cramer_values
pd.options.display.float_format = '{:.2f}'.format
c_results = cramer_values(df_merged)
c_results.to_csv('Cramer_Values.csv', index=False)

# Dropping highly correlated variables considering Cramer's value
df_merged.drop(columns=['Unknown Chronic Med Condition','No Chronic Med Condition',\
                     'Unknown Insurance Coverage','Medicare Insurance',\
                     'Other Chronic Med Condition', 'Veterans Cash Assistance'], inplace=True)

    

# ========================================================   
# 6. Converting nominal/ordinal values to numerical values 
# ========================================================

# Ordinal to numeric values
df_merged = df_merged.replace({'Education Status':{'PRE-K TO FIFTH GRADE':1,'MIDDLE SCHOOL TO HIGH SCHOOL':2,\
                               'SOME COLLEGE':3,'COLLEGE OR GRADUATE DEGREE':4,'UNKNOWN EDUCATION':5}})

# Replacing the text with Yes or No
df_merged['Hispanic Ethnicity'] = df_merged['Hispanic Ethnicity'].replace({'YES, HISPANIC/LATINO': 'YES',\
                                                               'NO, NOT HISPANIC/LATINO': 'NO'})

# Converting Nominal attribute to multiple binary attributes
cols_to_exclude = ['Program Category','Education Status','population','Unique Hospital Count','Unique Park Count','Median Household Income']
df_merged = pd.get_dummies(df_merged, columns=[col for col in df_merged.columns if col not in cols_to_exclude])

# Coverting categorical target variable to numerical values
le = LabelEncoder()
df_merged['Program Category'] = le.fit_transform(df_merged['Program Category'])
# df_merged.to_csv('PCS_converted.csv', index=False)

# Get the mapping of class names to numerical values
class_names = list(le.classes_)

# Generating correlation matrix for numerical data
corr_matrix = df_merged.corr()


# =============================================================================
# 7. Feature Selection using Feature Importances using the Random Forest Model
# =============================================================================

# Assigning target and feature variables
X = df_merged.iloc[:, 1: ]
y = df_merged.iloc[:, 0]
print(f'\nShape of the original feature data: {X.shape}')

fn = X.columns[0:]
print(f'Originally, we have {len(fn)} features.')

# Split the data into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size =.3,stratify=y)

# Create an instance (object) for classification and build a model.
rfcm = RandomForestClassifier().fit(X_train, y_train)

# Make predictions using the test data
y_pred = rfcm.predict(X_test)

# Show the Classification Report.
print('***Random Forest Model***')
print('\nClassification Report - Original data\n')
print(metrics.classification_report(y_test,y_pred))

# Find out the significant features for determining the Patient Care
importances = rfcm.feature_importances_
np.sum(importances)

# Draw a bar chart to see the sorted importance values with feature names.
df_importances = pd.DataFrame(data=importances, index=fn, 
                              columns=['importance_value'])
df_importances.sort_values(by = 'importance_value', ascending=False, 
                           inplace=True)
top_20_features = df_importances.head(20)

plt.barh(top_20_features.index,top_20_features.importance_value)
plt.xlabel('Importance Value')
plt.title('Top 20 Feature Importances using Random Forest Classifier')
plt.show()

# Set the threshold to the min importance value among the top 20 features
threshold = top_20_features['importance_value'].min()

# Build a model with a threshold based on the importance values of the top 20 features
selector = SelectFromModel(estimator=RandomForestClassifier(),threshold=threshold)
X_reduced = selector.fit_transform(X,y)
selected_TF = selector.get_support()
print(f'\nBy setting the threshold to be the min imporatnce of the top 20 features, {selected_TF.sum()} features\
      are selected from the original feature data.') 

# Show the first five names of those selected features.
selected_features = []
for i,j in zip(selected_TF, fn):
    if i: 
        selected_features.append(j)
print(f'The first five names of selected features are: \n{selected_features[:5]}') 

# Build a model using those reduced number of features.
X_reduced_train, X_reduced_test, y_reduced_train, y_reduced_test \
       = train_test_split(X_reduced,y,test_size =.3, stratify=y)

rfcm2 = RandomForestClassifier().fit(X_reduced_train, y_reduced_train)
y_reduced_pred = rfcm2.predict(X_reduced_test)
print('\nClassification Report - Reduced set of data\n')
print(metrics.classification_report(y_reduced_test,y_reduced_pred,target_names=class_names))



# =====================================
# 8. PCA (Primary Component Analysis)
# =====================================

# z_score normalize the data except the dummy variables
scaler = StandardScaler()
Xn = np.c_[scaler.fit_transform(X.iloc[:,:5].values), X.iloc[:, 5:].values] 

# Create an instance PCA and build the model using Xn
pca_prep = PCA().fit(Xn)

pca_prep.explained_variance_  #Eigen Values
pca_prep.explained_variance_ratio_

# Generating a scree plot to find an elbow or an inflection point on the plot
plt.plot(pca_prep.explained_variance_ratio_)
plt.xlabel('k number of components')
plt.ylabel('Explained variance')
plt.grid(True)
plt.show()

# Alternative plot using cumulative ratios
plt.plot(np.cumsum(pca_prep.explained_variance_ratio_))
plt.xlabel('k number of components')
plt.ylabel('cumulative explained variance')
plt.grid(True)
plt.show()

# From scree plot, we choose 30 components  
n_pc = 20
pca = PCA(n_components= n_pc).fit(Xn)

# X_pca has now 20 columns of primary components.
Xp = pca.transform(Xn)
print(f'After PCA, we use {pca.n_components_} components.\n')

# Split the data into training and testing subsets for PCA data
Xp_train, Xp_test, yp_train, yp_test = train_test_split(Xp,y,test_size =.3,
                                        random_state=1234,stratify=y)

# Create random forest model using the transformed data.
rfcm_pca = RandomForestClassifier().fit(Xp_train, yp_train)

# Predict the target values using the test data.
y_pred_pca = rfcm_pca.predict(Xp_test)

# Generate the Classification report
print('Classification Report - PCA\n')
print(metrics.classification_report(yp_test,y_pred_pca))



# =====================================
# 9. Naive Random Over Sampling
# =====================================

# Create an instance of RandomOverSampler
ros = RandomOverSampler(random_state=1234)
X_rs, y_rs = ros.fit_resample(Xn, y)
Xp_rs, yp_rs = ros.fit_resample(Xp, y)

X_rs.shape
y_rs.shape
Xp_rs.shape
yp_rs.shape
print(f'Over-sampled data: {np.unique(y_rs, return_counts=1)}')



# ==========================================================================================================
# Split the data into training and testing subsets for Original, Oversampled and PCA with Oversampled data
# ==========================================================================================================

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size =.3,
                                                    random_state=1234, stratify=y)

X_rs_train, X_rs_test, y_rs_train, y_rs_test = train_test_split(X_rs,y_rs,test_size =.3,
                                                    random_state=1234,stratify=y_rs)

Xp_rs_train, Xp_rs_test, yp_rs_train, yp_rs_test = train_test_split(Xp_rs,yp_rs,test_size =.3,
                                                    random_state=1234,stratify=yp_rs)



# ======================================
# 10. Random Forest Classification Model 
# ======================================

'''Create three models: first using the Original data, second using the Oversampled data 
and the third using the Oversampled transformed data'''

rfc_start = time.time()

rfc = RandomForestClassifier().fit(X_train, y_train)
rfc_rs = RandomForestClassifier().fit(X_rs_train, y_rs_train)
rfc_rs_pca = RandomForestClassifier().fit(Xp_rs_train, yp_rs_train)

# Predict the target values using the test data.
y_pred_rf = rfc.predict(X_test)
y_rs_pred_rf = rfc_rs.predict(X_test)
yp_rs_pred_rf = rfc_rs_pca.predict(Xp_test)

# Generate the Classification report
print('*** Random Forest Classification Model ***\n')
print('Classification Report - Original Data\n')
print(metrics.classification_report(y_test,y_pred_rf))

print('\nClassification Report - Oversampled Data\n')
print(metrics.classification_report(y_test,y_rs_pred_rf))

print('\nClassification Report - Oversampled PCA Data\n')
print(metrics.classification_report(yp_test,yp_rs_pred_rf))

rfc_end = time.time()
print(f'Random Forest Classification Model Execution time is {(rfc_end - rfc_start):.2f} seconds')

# Regularization and hyperparameter Tuning
rfc_clf = RandomForestClassifier(n_estimators = 100,
                                bootstrap = True,
                                max_samples = None, # int, float
                                oob_score = True,
                                criterion = "gini",
                                #splitter = "best", not available here, performs best
                                max_depth = 1,#decision stumps
                                min_samples_split = 2, 
                                min_samples_leaf = 1,
                                min_weight_fraction_leaf = 0,
                                max_features = None,
                                max_leaf_nodes = None,
                                random_state = None,
                                min_impurity_decrease = 0.0,
                                class_weight = None,
                                ccp_alpha = 0.0,
                                n_jobs = -1,
                                verbose=1
                               )
rfc_clf.fit(Xp_rs_train, yp_rs_train)
print("Test score:",rfc_clf.score(Xp_rs_train, yp_rs_train))
print("OOB score:",rfc_clf.oob_score_)


# =======================================
# 11. Decision Tree Classification Model 
# =======================================

'''Create three models: first using the Original data, second using the Oversampled data 
and the third using the Oversampled transformed data'''

dtc_start = time.time()

dtc = DecisionTreeClassifier().fit(X_train, y_train)
dtc_rs = DecisionTreeClassifier().fit(X_rs_train, y_rs_train)
dtc_rs_pca = DecisionTreeClassifier().fit(Xp_rs_train, yp_rs_train)

# Predict the target values using the test data
y_pred_dt = dtc.predict(X_test)
y_rs_pred_dt = dtc_rs.predict(X_test)
yp_rs_pred_dt = dtc_rs_pca.predict(Xp_test)

# Generate the Classification report
print('*** Decision Tree Classification Model ***\n')
print('Classification Report - Original Data\n')
print(metrics.classification_report(y_test,y_pred_dt))

print('\nClassification Report - Oversampled Data\n')
print(metrics.classification_report(y_test,y_rs_pred_dt))

print('\nClassification Report - Oversampled PCA Data\n')
print(metrics.classification_report(yp_test,yp_rs_pred_dt))

dtc_end = time.time()
print(f'Decision Tree Classification Model Execution time is {(dtc_end - dtc_start):.2f} seconds')

# Hyperparameter Tuning
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [25, 30, 35],
    'min_samples_split': [2,3],
    'min_samples_leaf': [2,3],
    'max_features': ['auto', 'sqrt', 'log2', None],
    'class_weight': ['balanced', None]
}

# Create a decision tree classifier object
dt = DecisionTreeClassifier(random_state=42)

# Create a GridSearchCV object with 10-fold cross-validation
gscv = GridSearchCV(dt, param_grid, cv=2, scoring='accuracy', verbose=1)
gscv_precision = GridSearchCV(dt, param_grid, cv=2, scoring='precision_macro', verbose=1)
gscv_recall = GridSearchCV(dt, param_grid, cv=2, scoring='recall_macro', verbose=1)

# Fit the GridSearchCV object on the training data
gscv.fit(Xp_rs_train, yp_rs_train)

# Print the best parameters and the best score
print('Best parameters:', gscv.best_params_)
print('Best score:', gscv.best_score_)


# ===========================================
# 12. Gradient Boosting Classification Model
# ===========================================

'''Create three models: first using the Original data, second using the Oversampled data 
and the third using the Oversampled transformed data'''

gbc_start = time.time()

gbc = GradientBoostingClassifier().fit(X_train, y_train)
gbc_rs = GradientBoostingClassifier().fit(X_rs_train, y_rs_train)
gbc_rs_pca = GradientBoostingClassifier().fit(Xp_rs_train, yp_rs_train)

# Predict the target values using the test data
y_pred_gb = gbc.predict(X_test)
y_rs_pred_gb = gbc_rs.predict(X_test)
yp_rs_pred_gb = gbc_rs_pca.predict(Xp_test)

# Show the Classification Report
print('*** Gradient Boosting Classification Model ***\n')

print('Classification Report - Original Data\n')
print(metrics.classification_report(y_test,y_pred_gb,target_names=class_names))

print('\nClassification Report - Oversampled Data\n')
print(metrics.classification_report(y_test,y_rs_pred_gb,target_names=class_names))

print('\nClassification Report - Oversampled PCA Data\n')
print(metrics.classification_report(yp_test,yp_rs_pred_gb,target_names=class_names))

gbc_end = time.time()
print(f'Gradient Boosting Classification Model Execution time is {(gbc_end - gbc_start):.2f} seconds')



# ==================================
# 13. Logistic Regression Model
# ==================================

'''Create three models: first using the Original data, second using the Oversampled data 
and the third using the Oversampled transformed data'''

clr_start = time.time()

clr = LogisticRegression(max_iter=1000).fit(X_train, y_train)
clr_rs = LogisticRegression(max_iter=1000).fit(X_rs_train, y_rs_train)
clr_rs_pca = LogisticRegression(max_iter=1000).fit(Xp_rs_train, yp_rs_train)

# Predict the target values using the test data
y_pred_lr = clr.predict(X_test)
y_rs_pred_lr = clr_rs.predict(X_test)
yp_rs_pred_lr = clr_rs_pca.predict(Xp_test)

# Show the Classification Report
print('*** Logistic Regression Model ***\n')

print('Classification Report - Original Data\n')
print(metrics.classification_report(y_test,y_pred_lr,target_names=class_names,zero_division=0))

print('\nClassification Report - Oversampled Data\n')
print(metrics.classification_report(y_test,y_rs_pred_lr,target_names=class_names,zero_division=0))

print('\nClassification Report - Oversampled PCA Data\n')
print(metrics.classification_report(yp_test,yp_rs_pred_lr,target_names=class_names,zero_division=0))

clr_end = time.time()
print(f'Logistic Regression Model Execution time is {(clr_end - clr_start):.2f} seconds')


# Hyperparameter tuning using GridSearchCV
clr_grid_start = time.time()

params_clr = {"penalty":["l2"],"C":[0.1,1,10]}, 
clr_grid = GridSearchCV(estimator=LogisticRegression(max_iter=1000),param_grid=params_clr, 
                        scoring = ["accuracy","roc_auc_ovr_weighted","f1_macro"], 
                        refit="roc_auc_ovr_weighted", #True 
                        cv = 3,  #If our estimator is classifier automatically do stratified CV 
                        n_jobs=-1,  #Num CPUs to use for calculation, -1 means all 
                        verbose = 1,  #Output status updates, higher number-> more messages 
                        return_train_score=True    #if false our results won't contain training scores 
                        )
clr_grid.fit(Xn, y)

clr_grid_end = time.time()

print('\n\n **Report**')
print(f'The best estimator: {clr_grid.best_estimator_}')
print(f'The best parameters:\n {clr_grid.best_params_}')
print(f'The best score: {clr_grid.best_score_:.4f}')
print(f'Total run time for GridSearchCV: {(clr_grid_end - clr_grid_start):.2f} seconds')


# =========================================
# 14. Naive Bayesian Classification Model
# =========================================

'''Create three models: first using the Original data, second using the Oversampled data 
and the third using the Oversampled transformed data'''

gnb_start = time.time()

gnb = GaussianNB().fit(X_train, y_train)
gnb_rs = GaussianNB().fit(X_rs_train, y_rs_train)
gnb_rs_pca = GaussianNB().fit(Xp_rs_train, yp_rs_train)

# Calculate the posteriori probabilities
p = gnb.predict_proba(X_test)
p_rs = gnb_rs.predict_proba(X_test)
p_rs_pca = gnb_rs_pca.predict_proba(Xp_test)

# Predict the target values using the test data
y_pred_gnb = gnb.predict(X_test)
y_rs_pred_gnb = gnb_rs.predict(X_test)
yp_rs_pred_gnb = gnb_rs_pca.predict(Xp_test)

# Show the Classification Report
print('*** Naive Bayesian Classification Model ***\n')

print('Classification Report - Original Data\n')
print(metrics.classification_report(y_test,y_pred_gnb,target_names=class_names,zero_division=0))

print('\nClassification Report - Oversampled Data\n')
print(metrics.classification_report(y_test,y_rs_pred_gnb,target_names=class_names,zero_division=0))

print('\nClassification Report - Oversampled PCA Data\n')
print(metrics.classification_report(yp_test,yp_rs_pred_gnb,target_names=class_names,zero_division=0))

gnb_end = time.time()
print(f'Gaussian Naive Bayesian Model Execution time is {(gnb_end - gnb_start):.2f} seconds')


# Hyperparameter tuning using GridSearchCV 
gnb_grid_start = time.time()

param_gnb = {'priors': [[0.2, 0.2, 0.2, 0.2, 0.2], [0.1, 0.15, 0.25, 0.3, 0.2]],  
             'var_smoothing': [1e-50, 1e-40]} 

 
# Create a Naive Bayesian classifier object 
gnb_grid = GaussianNB(priors = None,  # class priors, if defined priors won't be set from data 
                var_smoothing = 1e-9  # added this time maximum feature variance onto variance for smoothing 
                ) 
 
# Create a GridSearchCV object with 10-fold cross-validation 
gscv = GridSearchCV(gnb_grid, param_gnb, cv=10, scoring='accuracy', verbose=1) 
gscv_precision = GridSearchCV(gnb_grid, param_gnb, cv=10, scoring='precision_macro', verbose=1) 
gscv_recall = GridSearchCV(gnb_grid, param_gnb, cv=10, scoring='recall_macro', verbose=1) 
 
# Fit the GridSearchCV object on the training data 
gscv.fit(Xp_rs_train, yp_rs_train) 

gnb_grid_end = time.time()

print('\n\n **Report**')
print(f'The best estimator: {gscv.best_estimator_}')
print(f'The best parameters:\n {gscv.best_params_}')
print(f'The best score: {gscv.best_score_:.4f}')
print(f'Total run time for GridSearchCV: {(gnb_grid_end - gnb_grid_start):.2f} seconds')



# ================================================
# 16. Neural Network Classification Model
# ================================================

# Hyperparameter tuning using RandomizedSearchCV

nnm_rand = MLPClassifier()

params_rand = {
    'hidden_layer_sizes': [(160,), (180,), (200,), (220,), (160, 160), (180, 180), (200, 200), (220, 220)],
    'activation': ['logistic', 'tanh', 'relu'],
    'learning_rate': ['adaptive', 'constant'],
    'learning_rate_init': [0.001, 0.01, 0.1],
    'max_iter': [200, 500, 1000, 2000]
    } 
   
start_rand = time.time()

rand_src = RandomizedSearchCV(estimator= nnm_rand, param_distributions = params_rand,n_iter=10, random_state=1234, scoring='accuracy')
rand_src.fit(Xn,y)

end_rand = time.time()
print('\n\n **Report**')
print(f'The best estimator: {rand_src.best_estimator_}')
print(f'The best parameters:\n {rand_src.best_params_}')
print(f'The best score: {rand_src.best_score_:.4f}')
print(f'Total run time for RandomizedSearchCV: {(end_rand - start_rand):.2f} seconds')


# Hyperparameter tuning using GridSearchCV

nnm_grid = MLPClassifier()

params_grid = {'hidden_layer_sizes':[(20), (30)],
                    'activation':['logistic','relu','tanh'], 'max_iter': [4000,5000]}

start_grid = time.time()

grid_src = GridSearchCV(estimator= nnm_grid, param_grid= params_grid)
grid_src.fit(Xn, y)

end_grid = time.time()
print('\n\n **Report**')
print(f'The best estimator: {grid_src.best_estimator_}')
print(f'The best parameters:\n {grid_src.best_params_}')
print(f'The best score: {grid_src.best_score_:.4f}')
print(f'Total run time for GridSearchCV: {(end_grid - start_grid):.2f} seconds')


nnm_start = time.time()

'''Create three models: first using the Original data, second using the Oversampled data 
and the third using the Oversampled transformed data'''

nnm = MLPClassifier(hidden_layer_sizes=(200, 200), activation='relu',
                        max_iter=1000, learning_rate_init = 0.01, learning_rate = 'constant', random_state=1234)
nnm_rs = MLPClassifier(hidden_layer_sizes=(200, 200), activation='relu',
                        max_iter=1000, learning_rate_init = 0.01, learning_rate = 'constant', random_state=1234)
nnm_rs_pca = MLPClassifier(hidden_layer_sizes=(200, 200), activation='relu',
                        max_iter=1000, learning_rate_init = 0.01, learning_rate = 'constant', random_state=1234)


nnm.fit(X_train, y_train)
nnm_rs.fit(X_rs_train, y_rs_train)
nnm_rs_pca.fit(Xp_rs_train, yp_rs_train)

# Predict the target values using the test data
y_pred_nnm = nnm.predict(X_test)
y_rs_pred_nnm = nnm_rs.predict(X_test)
yp_rs_pred_nnm = nnm_rs_pca.predict(Xp_test)

# Show the Classification Report
print('*** Neural Networks Classification Model ***\n')

print('Classification Report - Original Data\n')
print(metrics.classification_report(y_test,y_pred_nnm,target_names=class_names))

print('Classification Report - Oversampled Data\n')
print(metrics.classification_report(y_test,y_rs_pred_nnm,target_names=class_names))

print('Classification Report - Oversampled PCA Data\n')
print(metrics.classification_report(yp_test,yp_rs_pred_nnm,target_names=class_names))

nnm_end = time.time()
print(f'Neural Networks Classification Model Execution time is {(nnm_end - nnm_start):.2f} seconds')



# ================================================
# 17. K-Fold Cross Validation for Classification
# ================================================

nnm_kf_start = time.time()

# Use Cross_val_score
nnm_mean_score = np.mean(cross_val_score(nnm,Xn,y,cv=5))
nnm_rs_mean_score = np.mean(cross_val_score(nnm_rs,X_rs,y_rs,cv=5))
nnm_rs_pca_mean_score = np.mean(cross_val_score(nnm_rs_pca,Xp_rs,yp_rs,cv=5))

# Print the scores
print('** Mean Scores (Accuracies) **')
print(f'Mean Score for Neural Network - Original Data: {nnm_mean_score:.4f}')
print(f'Mean Score for Neural Network - Oversampled Data: {nnm_rs_mean_score:.4f}')
print(f'Mean Score for Neural Network - Oversampled PCA Data: {nnm_rs_pca_mean_score:.4f}')

nnm_kf_end = time.time()
print(f'Neural Network Model with K-Fold Cross Validation Execution time is {(nnm_kf_end - nnm_kf_start):.2f} seconds')



# ================================================
# 18. Clustering
# ================================================

'''Finding the optimal K value using the elbow method on the inertia values. 
   Inertia shows the sum of squared distances of data points from their corresponding centroid.
   Usually, the lower the inertia the better model is. If the K value is large, inertia will decrease. 
   However, there will be costs involved such as overfitting, computation time, and others. 
   Also, we need to find an "optimal" value considering the domain problem'''

# Initialize the list for inertia values - sum of squared distances
inertia_list = []

# Calculate the inertia for the number of clusters.
for i in range(5,31):
    km = KMeans(n_clusters=i, random_state=1234)
    km.fit(Xn)
    inertia_list.append(km.inertia_)

# Check the inertia values.
for i in range(len(inertia_list)):
    print('{0}: {1:.2f}'.format(i+5, inertia_list[i]))

# Draw the plot to find the elbow
plt.plot(range(5,31), inertia_list)
plt.grid(True)
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()


'''Create three models: first using the Original data, second using the Oversampled data 
and the third using the Oversampled transformed data'''

# Using the elbow mwthod, we found optimal k value for k-means clustering model is 11.
km_start = time.time()

# Create an instance (object) of the KMeans class with the parameters initialized and build the model
km = KMeans(n_clusters=9, random_state=1234).fit(Xn)
km_rs = KMeans(n_clusters=9, random_state=1234).fit(X_rs)
km_rs_pca = KMeans(n_clusters=9, random_state=1234).fit(Xp_rs)

# Create clusters and assign them to a variable
labels_km = km.labels_
labels_km_rs = km_rs.labels_
labels_km_rs_pca = km_rs_pca.labels_ 

# Evaluation of K-means Clustering model using Silhouette score
print('Performance Evaluation using Silhouette score\n') 
print(f'K-Means Clustering - Original Data: {silhouette_score(Xn,labels_km):.2f}')
print(f'K-Means Clustering - Oversampled Data: {silhouette_score(X_rs,labels_km_rs):.2f}')
print(f'K-Means Clustering - Oversampled PCA Data: {silhouette_score(Xp_rs,labels_km_rs_pca):.2f}')

# Add the cluster numbers to the original data.
df_km = pd.DataFrame(data=X,columns=fn)
df_km['Cluster_No'] = labels_km
df_km.info()

# Divide the original data into the clusters.
Cluster_0 = df_km.loc[df_km.Cluster_No == 0]
Cluster_1 = df_km.loc[df_km.Cluster_No == 1]
Cluster_2 = df_km.loc[df_km.Cluster_No == 2]
Cluster_3 = df_km.loc[df_km.Cluster_No == 3]
Cluster_4 = df_km.loc[df_km.Cluster_No == 4]
Cluster_5 = df_km.loc[df_km.Cluster_No == 5]
Cluster_6 = df_km.loc[df_km.Cluster_No == 6]
Cluster_7 = df_km.loc[df_km.Cluster_No == 7]
Cluster_8 = df_km.loc[df_km.Cluster_No == 8]

km_end = time.time()
print(f'K-means Clustering Execution time is {(km_end - km_start):.2f} seconds')



# ================================================
# 16. Support Vector Machines
# ================================================

# # Hyperparameter tuning using RandomizedSearchCV 

params_SVM = {'C': (5, 10, 50),  # Regularization parameter
    'kernel': ['linear', 'rbf', 'poly'],  # Kernel type
    'penalty': ["l1", "l2"],  # The norm used in the penalization 
    'dual': [True, False],  # Whether to use the dual or primal formulation 
    'max_iter': [100000],  # The maximum number of iterations 
    'tol': (1e-8, 1e-10, 1e-15)}
    
start_svm = time.time()

# Create an instance of LinearSVC 
svm = LinearSVC()
rand_svm = RandomizedSearchCV(estimator= svm, param_distributions = params_SVM,
                              n_iter=10,cv=2, verbose=0, n_jobs=-1,random_state=42)
rand_svm.fit(Xn,y)

end_svm = time.time()

print('\n\n **Report**')
print(f'The best estimator: {rand_svm.best_estimator_}')
print(f'The best parameters:\n {rand_svm.best_params_}')
print(f'The best score: {rand_svm.best_score_:.4f}')
print(f'Total run time for RandomizedSearchCV: {(end_svm - start_svm):.2f} seconds')


linear_svm_start = time.time() 
  
# Specific linear SVC implementation that scales better to larger number of data 
lin_svc_clf = LinearSVC(penalty="l2", 
                       loss="squared_hinge", 
                       tol=1e-4, 
                       C = 1, 
                       max_iter=10000, 
                       verbose=True)  #ovr multiclass implementation 
 
lin_svc_clf.fit(Xp_train, yp_train) 
lin_svc_clf.coef_, lin_svc_clf.classes_   #weights for each feature 
y_pred_svc = lin_svc_clf.predict(Xp_test) 

print('*** Support Vector Machines Classification Model ***\n') 

lin_svc_cm = confusion_matrix(yp_test, y_pred_svc)  
print('Confusion Matrix\n')
print(lin_svc_cm) 
print(f'Accuracy from training data: {lin_svc_clf.score(Xp_train, yp_train):.2f}') 
print(f'Accuracy from testing data: {lin_svc_clf.score(Xp_test, yp_test):.2f}') 

linear_svm_end = time.time()   
print(f" Linear SVM RUN time: {linear_svm_end - linear_svm_start}s") 
