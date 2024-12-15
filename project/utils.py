# ----------------------------------------------------------------------------------------------------------------------
# Libraries
# ----------------------------------------------------------------------------------------------------------------------

# General Libraries
import pandas as pd
import numpy as np
import re
from math import ceil
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Machine Learning and Data Preprocessing
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder

# Feature Selection
from sklearn.feature_selection import RFE
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Machine Learning Models and Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from xgboost import XGBClassifier # pip install xgboost scikit-learn
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Model Evaluation
from sklearn.metrics import f1_score
import time

# ----------------------------------------------------------------------------------------------------------------------
# Helper Function to fix C-2 and C-3 Dates (from Notebook 2)
# ----------------------------------------------------------------------------------------------------------------------

def adjust_dates(row):
    # If Assembly Date is earlier than C-2 Date, adjust C-2 Date to match the Assembly Date
    if (row['C-2 Date'] > row['Assembly Date']) | (row['C-2 Date'] < row['Accident Date']):
        row['C-2 Date'] = row['Assembly Date']

    # If Assembly Date is earlier than C-3 Date
    if (row['C-3 Date'] > row['Assembly Date']) | (row['C-3 Date'] < row['C-2 Date']) | (row['C-3 Date'] < row['Accident Date']):
        if pd.notna(row['C-2 Date']): # adjust C-3 Date to match the C-2 Date
            row['C-3 Date'] = row['C-2 Date']
        else: # adjust C-3 Date to match the Assembly Date
            row['C-3 Date'] = row['Assembly Date']
    
    return row

# ----------------------------------------------------------------------------------------------------------------------
# Dictionary of features lists
# ----------------------------------------------------------------------------------------------------------------------

feats_dict = {
    # winsorization
    "winsorization": [
        "Age at Injury",
        "Average Weekly Wage"
    ],

    # missing values imputation
    "nums_imputation": [
        "Average Weekly Wage",
        "Male",
        "Alternative Dispute Resolution"
    ],
    "cats_imputation": [
        "Zip Code",
        "Medical Fee Region",
        "WCIO Part Of Body Code",
        "WCIO Nature of Injury Code",
        "WCIO Cause of Injury Code",
        "Industry Code",
        "Carrier Type",
        "County of Injury"
    ],
    
    # drop unnecessary columns
    "codes_drop": [
        "WCIO Part Of Body Code",
        "WCIO Nature of Injury Code",
        "WCIO Cause of Injury Code",
    ],
    "descriptions_drop": [
        "Carrier Name",
        "Industry Code Description",
        "WCIO Cause of Injury Description",
        "WCIO Nature of Injury Description",
        "WCIO Part Of Body Description"
    ],
    "dates_drop": [
        "Accident Date",
        "Assembly Date",
        "C-2 Date",
        "C-3 Date",
        "First Hearing Date"
    ],

    # categorical encoding
    "ordinal_features": [
        "Age Group at Injury",
        "Average Weekly Wage Category"
    ],
    "high_cardinality_features": [
        "Carrier Type",
        "County of Injury",
        "District Name",
        "Industry Code",
        "Medical Fee Region",
        "Zip Code",
        "Part of Body Category",
        "Nature of Injury Category",
        "Cause of Injury Category",
    ],

    # spearman selection (exclude binary features)
    "binary_features": [
        "Alternative Dispute Resolution",
        "Attorney/Representative",
        "COVID-19 Indicator",
        "Male",
    ],
}

# ----------------------------------------------------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------------------------------------------------

def manual_filter_outliers(train, age_threshold=100, avg_wage_threshold=8000):
    filter_age = train["Age at Injury"].isna() | (train["Age at Injury"] <= age_threshold)
    filter_wage = train["Average Weekly Wage"].isna() | (train["Average Weekly Wage"] <= avg_wage_threshold)
    
    return train[filter_age & filter_wage]

def winsorization(data, col, train=None, bounds=None):
    # Calculate bounds if training data is provided
    if train is not None:
        Q1 = train[col].quantile(0.25)
        Q3 = train[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        bounds = (lower_bound, upper_bound)

    # Apply Winsorization using the calculated or provided bounds
    data[col] = data[col].clip(lower=bounds[0], upper=bounds[1])

    return data, bounds

def impute_missing_values(data, feats_dict, train=None, imputers=None):
    # Convert all missing values to np.nan for consistency
    data.fillna(np.nan, inplace=True)
    
    # Define numeric and categorical columns
    numeric_columns = feats_dict["nums_imputation"]
    categorical_columns = feats_dict["cats_imputation"]
    
    # Create or use existing imputers
    if train is not None:
        numeric_imputer = SimpleImputer(strategy='median')
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        # Fit imputers on training data
        numeric_imputer.fit(train[numeric_columns])
        categorical_imputer.fit(train[categorical_columns])
        # Store imputers for future use
        imputers = (numeric_imputer, categorical_imputer)
    
    # Apply imputers to the data
    data[numeric_columns] = imputers[0].transform(data[numeric_columns])
    data[categorical_columns] = imputers[1].transform(data[categorical_columns])

    return data, imputers

# https://www.guarantysupport.com/wp-content/uploads/2024/02/WCIO-Legacy.pdf
# https://www.nycirb.org/digital-library/public/pdf_generate_paragraph?paragraph_id=A4E00D96-5539-4FD3-9489-41864CCC6503
def create_part_of_body_category(df, col='WCIO Part Of Body Code'):
    # Define mapping of categories to code ranges
    category_map = {
        'Head': range(10, 20),
        'Neck': range(20, 27),
        'Upper Extremities': range(30, 40),
        'Trunk': list(range(40, 50)) + list(range(60, 64)),
        'Lower Extremities': range(50, 59),
        'Multiple Body Parts': list(range(64, 67)) + [90, 91]
    }
    
    # Initialize the new column with the default value
    df['Part of Body Category'] = 'OTHER'
    
    # Assign categories based on the mapping
    for label, code_range in category_map.items():
        df.loc[df[col].isin(code_range), 'Part of Body Category'] = label
    
    return df

# https://www.guarantysupport.com/wp-content/uploads/2024/02/WCIO-Legacy.pdf
def create_nature_of_injury_category(df, col='WCIO Nature of Injury Code'):
    # Define mapping of categories to code ranges
    category_map = {
        'Specific Injury': [1, 2, 3, 4, 7, 10, 13, 16, 19, 22, 25, 28, 30, 31, 32, 34, 36, 37, 40, 41, 42, 43, 46, 47, 49, 52, 53, 54, 55, 58, 59],
        'Occupational Disease or Cumulative Injury': list(range(60, 81)),
        'Multiple Injury': [90, 91]
    }
    
    # Initialize the new column with the default value
    df['Nature of Injury Category'] = 'OTHER'
    
    # Assign categories based on the mapping
    for label, code_range in category_map.items():
        df.loc[df[col].isin(code_range), 'Nature of Injury Category'] = label
    
    return df

# https://www.guarantysupport.com/wp-content/uploads/2024/02/WCIO-Legacy.pdf
def create_cause_of_injury_category(df, col='WCIO Cause of Injury Code'):
    # Define mapping of categories to code ranges
    category_map = {
        'Burn or Scald': list(range(1, 10)) + [11, 14, 84],
        'Caught In, Under or Between': [10, 12, 13, 20],
        'Cut, Puncture, Scrape Injured By': range(15, 20),
        'Fall, Slip or Trip Injury': range(25, 34),
        'Motor Vehicle': [40, 41, 45, 46, 47, 48, 50],
        'Strain or Injury By': list(range(52, 62)) + [97],
        'Striking Against or Stepping On': range(65, 71),
        'Struck or Injured By': list(range(74, 82)) + [85, 86],
        'Rubbed or Abraded By': [94, 95],
        'Miscellaneous Causes': [82, 87, 88, 89, 90, 91, 93, 96, 98, 99],
    }

    # Initialize the new column with a default value
    df['Cause of Injury Category'] = 'Other'

    # Assign categories based on the mapping
    for label, code_range in category_map.items():
        df.loc[df[col].isin(code_range), 'Cause of Injury Category'] = label

    return df

def create_features(df):
    # -------------------- Date-based Features --------------------
    # Commented out as dates missing haven't been treated, so these features can't be reliably created for future datasets
    
    # # Extract injury year, quarter, and month from the 'Accident Date'
    # df['Injury Year'] = df['Accident Date'].dt.year
    # df['Injury Quarter'] = df['Accident Date'].dt.quarter
    # df['Injury Month'] = df['Accident Date'].dt.month
    # features_created.append('Injury Year')
    # features_created.append('Injury Quarter')
    # features_created.append('Injury Month')

    # # Calculate time-based features (in years)
    # df['Years Between Accident and Assembly'] = (df['Assembly Date'] - df['Accident Date']).dt.days / 365
    # df['Years Since Accident'] = (pd.to_datetime('today') - df['Accident Date']).dt.days / 365
    # features_created.append('Years Between Accident and Assembly')
    # features_created.append('Years Since Accident')

    # -------------------- Missing Data Flags --------------------
    
    # Create flags for missing dates (C-2 Date, C-3 Date, First Hearing Date)
    df['C-2 Date Missing'] = df['C-2 Date'].isna().astype(int)
    df['C-3 Date Missing'] = df['C-3 Date'].isna().astype(int)
    df['First Hearing Date Missing'] = df['First Hearing Date'].isna().astype(int)

    # -------------------- Age and Wage-based Features --------------------
    
    # Age Group at Injury (Categorizing based on age ranges)
    age_bins = [-float('inf'), 25, 35, 45, 60, float('inf')]
    age_labels = ['Young Adult', 'Adult', 'Mid-aged Adult', 'Older Adult', 'Senior']
    df['Age Group at Injury'] = pd.cut(df['Age at Injury'], bins=age_bins, labels=age_labels)

    # Average Weekly Wage Category (Categorizing into wage brackets)
    wage_bins = [-float('inf'), 50, 300, 1000, 2000, float('inf')]
    wage_labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
    df['Average Weekly Wage Category'] = pd.cut(df['Average Weekly Wage'], bins=wage_bins, labels=wage_labels)

    # Log transformation of Age at Injury (right-skewed distribution)
    df['Log Age at Injury'] = np.log(df['Age at Injury'])

    # Interaction term between Age and Average Weekly Wage
    df['Age * Avg Weekly Wage'] = df['Age at Injury'] * df['Average Weekly Wage']

    # -------------------- Aggregated Features --------------------
    
    # Average Weekly Wage by Medical Fee Region and by Gender
    df['Average Weekly Wage by Medical Fee Region'] = df.groupby('Medical Fee Region')['Average Weekly Wage'].transform('mean')
    df['Average Weekly Wage by Gender'] = df.groupby('Male')['Average Weekly Wage'].transform('mean')

    # -------------------- Category-based Features --------------------
    
    # Convert WCIO injury code columns to numeric for easier manipulation
    codes_to_numeric = ["WCIO Part Of Body Code", "WCIO Nature of Injury Code", "WCIO Cause of Injury Code"]
    df[codes_to_numeric] = df[codes_to_numeric].apply(pd.to_numeric)

    # Create categories for injury code columns based on reference documents:
    # https://www.guarantysupport.com/wp-content/uploads/2024/02/WCIO-Legacy.pdf
    # https://www.nycirb.org/digital-library/public/pdf_generate_paragraph?paragraph_id=A4E00D96-5539-4FD3-9489-41864CCC6503
    create_part_of_body_category(df)
    create_nature_of_injury_category(df)
    create_cause_of_injury_category(df)

    return df

def ordinal_encoder(feats_dict, train=None, data=None, encoders=None):
    # If training data is provided, create new encoders
    if train is not None:
        # Initialize a dictionary to store encoders
        encoders = {}
        for col in feats_dict["ordinal_features"]:
            # Create a new encoder for each column
            encoder = OrdinalEncoder()
            # Fit and transform the encoder on training data
            train[col] = encoder.fit_transform(train[[col]])
            # Store the encoder
            encoders[col] = encoder

    # Apply the encoders to the provided data (validation/test)
    if data is not None:
        for col in feats_dict["ordinal_features"]:
            # Apply the encoder for this column
            data[col] = encoders[col].transform(data[[col]])

    return train, data, encoders

def frequency_encoder(feats_dict, train=None, data=None, encoders=None, fill_values=None):
    # Initialize encoders and fill_values if training data is provided
    if train is not None:
        # Initialize dictionaries to store encoders and fill values
        encoders = {}
        fill_values = {}
        for col in feats_dict["high_cardinality_features"]:
            # Create a frequency map for the column based on the training data
            freq_map = train[col].value_counts().to_dict()
            # Store the frequency map in the encoders dictionary
            encoders[col] = freq_map
            # Apply the frequency map to the training data
            train[col] = train[col].map(freq_map)
            # Calculate and store the median of the column
            fill_values[col] = train[col].median()

    # Apply the encoders to the provided data (validation/test)
    if data is not None:
        for col in feats_dict["high_cardinality_features"]:
            # Retrieve the encoder (frequency map) and fill value for this column
            freq_map = encoders[col]
            fill_value = fill_values[col]
            # Apply the frequency map to the data and use the stored fill_value for unseen categories
            data[col] = data[col].map(freq_map).fillna(fill_value)

    return train, data, encoders, fill_values

def scale_data(X_train=None, data=None, scaler=None):
    # If training data is provided, fit a new scaler
    if X_train is not None:
        # Save original column names
        original_col_names = X_train.columns
        # Initialize the scaler
        scaler = StandardScaler()
        # Fit and transform training data
        X_train = scaler.fit_transform(X_train)
        # Convert back to DataFrame
        X_train = pd.DataFrame(X_train, columns=original_col_names)

    # If validation/test data is provided, apply the existing scaler
    if data is not None:        
        # Save original column names
        original_col_names = data.columns
        # Transform validation/test data
        data = scaler.transform(data)
        # Convert back to DataFrame
        data = pd.DataFrame(data, columns=original_col_names)

    return X_train, data, scaler

def label_encoder(y_train, y_val=None):
    # Initialize the label encoder
    lencoder = LabelEncoder()
    
    # Encode the target variable
    y_train = lencoder.fit_transform(y_train)
    # Convert the encoded labels back to a DataFrame
    y_train = pd.DataFrame(y_train, columns=['Claim Injury Type'])

    if y_val is not None:
        # Encode the target variable
        y_val = lencoder.transform(y_val)
        # Convert the encoded labels back to a DataFrame
        y_val = pd.DataFrame(y_val, columns=['Claim Injury Type'])
    
    return y_train, y_val, lencoder

def spearman_selection(X_train, feats_dict, threshold=0.8):
    # Exclude binary variables
    non_binary_features = [col for col in X_train.columns if col not in feats_dict["binary_features"]]

    # Calculate the correlation matrix for non-binary features
    correlation_matrix = X_train[non_binary_features].corr(method='spearman')

    # Create a set to track excluded features
    excluded_features = set()
    # Track pairs of highly correlated features
    highly_correlated_pairs = []

    # Check for pairs of highly correlated features
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            # If the absolute correlation is above the threshold
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                feature_i = correlation_matrix.columns[i]
                feature_j = correlation_matrix.columns[j]
                highly_correlated_pairs.append((feature_i, feature_j))

                # Decide which feature to exclude based on variance
                variance_i = X_train[feature_i].var()
                variance_j = X_train[feature_j].var()
                
                if variance_i >= variance_j:
                    excluded_features.add(feature_j)
                else:
                    excluded_features.add(feature_i)
    
    # Return a list of features that are not excluded
    selected_features = [col for col in non_binary_features if col not in excluded_features]

    # # Print results
    # print(f"Spearman Correlation - Number of Excluded Features: {len(excluded_features)} / {len(non_binary_features)}")
    # print("Spearman Correlation - Excluded Features:", excluded_features)
    # print("Spearman Correlation - Highly Correlated Pairs:", highly_correlated_pairs)
    # print("")
          
    return selected_features

def rfe_selection(X_train, y_train, model=None, percentile=25, random_state=42):
    # If no model is provided, use Logistic Regression
    if model is None:
        model = LogisticRegression(random_state=random_state)
    
    # Perform RFE
    rfe = RFE(model)
    rfe.fit(X_train, y_train)
    
    # Get feature importance from ranking
    importance_scores = 1 / (rfe.ranking_)
    importance_scores /= importance_scores.sum()  # Normalize to sum to 1
    
    # Determine the threshold
    threshold_value = np.percentile(importance_scores, percentile)
    
    # Select features that exceed the threshold
    selected_features = [col for col, importance in zip(X_train.columns, importance_scores) if importance > threshold_value]
    
    # # Print results
    # print(f"RFE - Number of Selected Features: {len(selected_features)} / {X_train.shape[1]}")
    # print("RFE - Selected Features:", selected_features)
    # print("")
    
    return selected_features

def dt_selection(X_train, y_train, random_state=42):
    # Initialize Decision Tree model
    tree = DecisionTreeClassifier(random_state=random_state)
    tree.fit(X_train, y_train)

    # Get feature importances
    importance = pd.Series(tree.feature_importances_, index=X_train.columns)

    # Determine the average expected contribution threshold
    threshold_value = 1 / X_train.shape[1]

    # Select features above the threshold
    selected_features = list(importance[importance > threshold_value].index)
    
    # # Print results
    # print(f"Decision Tree - Number of Selected Features: {len(selected_features)} / {X_train.shape[1]}")
    # print("Decision Tree - Selected Features:", selected_features)
    # print("")

    return selected_features

def lasso_selection(X_train, y_train, random_state=42):
    # Initialize Lasso Regression
    lasso = LassoCV(random_state=random_state)
    lasso.fit(X_train, y_train)

    # Extract features with non-zero coefficients
    lasso_coef = pd.Series(lasso.coef_, index=X_train.columns)
    non_zero_coef = lasso_coef[lasso_coef != 0]
    selected_features = list(non_zero_coef.index)

    # # Print results
    # print(f"Lasso Regression - Number of Selected Features: {len(selected_features)} / {X_train.shape[1]}")
    # print("Lasso Regression - Selected Features:", selected_features)
    # print("")

    return selected_features

def voting_feature_selection(X_train, selected_features_methods, vote_threshold=None):
    # Create a DataFrame to track votes for each feature
    feature_votes = pd.DataFrame(0, index=X_train.columns, columns=selected_features_methods.keys())

    # Fill the DataFrame with votes for each feature
    for method, features in selected_features_methods.items():
        feature_votes.loc[features, method] = 1

    # Calculate total votes for each feature
    feature_votes['total_votes'] = feature_votes.sum(axis=1)

    # Define default vote threshold if not provided
    if vote_threshold is None:
        vote_threshold = len(selected_features_methods) // 2  # Majority rule

    # Select features based on vote threshold
    final_selected_features = feature_votes[feature_votes['total_votes'] > vote_threshold].index.tolist()
    
    # # Print results
    # print(f"Majority Voting - Number of Selected Features: {len(final_selected_features)} / {X_train.shape[1]}")
    # print("Majority Voting - Selected Features:", final_selected_features)
    # print("")

    return final_selected_features

def define_models(models_parameters):
    models_and_parameters = {
        "LogisticRegression": (LogisticRegression(random_state=42), models_parameters["LogisticRegression"]),
        "RandomForest": (RandomForestClassifier(random_state=42), models_parameters["RandomForest"]),
        "XGBoost": (XGBClassifier(random_state=42),models_parameters["XGBoost"]),
        "SVM": (SVC(random_state=42), models_parameters["SVM"]),
        "Bagging": (BaggingClassifier(random_state=42), models_parameters["Bagging"]),
        "DecisionTree": (DecisionTreeClassifier(random_state=42), models_parameters["DecisionTree"]),
        "NeuralNetworks": (MLPClassifier(random_state=42), models_parameters["NeuralNetworks"]),
        "Adaboost": (AdaBoostClassifier(random_state=42), models_parameters["Adaboost"]),
        "GBM": (GradientBoostingClassifier(random_state=42), models_parameters["GBM"]),
     }

    return models_and_parameters

# ----------------------------------------------------------------------------------------------------------------------
# Cross Validation
# ----------------------------------------------------------------------------------------------------------------------

def cross_validation(models_and_parameters, df, feats_dict, n_splits=3, n_repeats=2):
    # Separate features and target
    X = df.drop(columns=["Claim Injury Type"])
    y = df["Claim Injury Type"]

    # Initialize lists to store F1 scores
    results = {
        model_name: {
            "f1_diff_list": [],
            "f1_val_list": [],
            "f1_train_list": [],
            "time_list": [],
            "n_iter_list": [],
            "best_params_list": []
        }
        for model_name in models_and_parameters.keys()
    }

    # Cross-validation strategy
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
   
    # Perform cross-validation
    for iter, (train_index, val_index) in enumerate(rskf.split(X, y)):
        print("-" * 80)
        print('CV Iteration', iter+1)
        print("-" * 80)
        print()

        # Split data 
        X_train, X_val = X.iloc[train_index].copy(), X.iloc[val_index].copy()
        y_train, y_val = y.iloc[train_index].copy(), y.iloc[val_index].copy()

        # -------------------- Treatment --------------------

        # Apply manual outlier removal
        X_train = manual_filter_outliers(X_train)
        y_train = y_train.loc[X_train.index]
        
        # Winsorization
        for col in feats_dict["winsorization"]:
            # Calculate bounds using training data and apply to training set
            X_train, winsorization_bounds = winsorization(X_train, col, train=X_train)
            # Apply the same bounds to the validation set
            X_val, _ = winsorization(X_val, col, bounds=winsorization_bounds)

        # Missing values imputation
        # Fit and apply imputation on the training set
        X_train, mv_imputers = impute_missing_values(X_train, feats_dict, train=X_train)
        # Apply the same imputers to the validation set
        X_val, _ = impute_missing_values(X_val, feats_dict, imputers=mv_imputers)

        # Feature engineering
        X_train = create_features(X_train)
        X_val = create_features(X_val)

        # Drop description and date columns
        X_train.drop(columns = feats_dict["codes_drop"] + feats_dict["descriptions_drop"] + feats_dict["dates_drop"], inplace=True)
        X_val.drop(columns = feats_dict["codes_drop"] + feats_dict["descriptions_drop"] + feats_dict["dates_drop"], inplace=True)

        # Ordinal encoding
        X_train, X_val, _ = ordinal_encoder(feats_dict, train=X_train, data=X_val)

        # Frequency encoding
        X_train, X_val, _, _ = frequency_encoder(feats_dict, train=X_train, data=X_val)

        # Data scaling
        X_train, X_val, _ = scale_data(X_train, X_val)
    
        # Label encoding
        y_train, y_val, _ = label_encoder(y_train, y_val)

        # Feature selection
        selected_feats_spearman = spearman_selection(X_train, feats_dict)
        selected_feats_rfe = rfe_selection(X_train, y_train)
        selected_feats_dt = dt_selection(X_train, y_train)
        selected_feats_lasso = lasso_selection(X_train, y_train)

        selected_features_methods = {
            'spearman': selected_feats_spearman,
            'rfe': selected_feats_rfe,
            'dt': selected_feats_dt,
            'lasso': selected_feats_lasso,
        }
        final_selected_features = voting_feature_selection(X_train, selected_features_methods)
        X_train = X_train[final_selected_features]
        X_val = X_val[final_selected_features]

        # -------------------- GridSearch --------------------

        print('End of Preprocessing; Starting GridSearch\n')

        # Combine train and validation datasets
        X_combined = np.concatenate([X_train, X_val])
        y_combined = np.concatenate([y_train, y_val])

        # Create a test fold index (-1 for train, 0 for validation)
        test_fold = [-1] * len(X_train) + [0] * len(X_val)
        
        # Define the PredefinedSplit
        ps = PredefinedSplit(test_fold=test_fold)

        for model_name, (model, param_grid) in models_and_parameters.items():
            # Define the GridSearch
            gridsearch = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                scoring="f1_macro", # Macro F1 scores to better align with Kaggle’s evaluation criteria
                cv=ps,
                refit=False
            )

            # Time the model training
            begin = time.perf_counter()
            gridsearch.fit(X_combined, y_combined)
            end = time.perf_counter()

            # Extract the best parameters and refit the model on the training set (refit is set to False)
            best_params = gridsearch.best_params_
            best_model = model.set_params(**best_params)
            best_model.fit(X_train, y_train)

            # Collect metrics
            print('Collecting Model Metrics')

            y_train_pred = best_model.predict(X_train)
            y_val_pred = best_model.predict(X_val)

            # Store metrics in the results dictionary
            results[model_name]['time_list'].append(end - begin)
            results[model_name]['best_params_list'].append(best_params)
            results[model_name]['f1_val_list'].append(f1_score(y_val, y_val_pred, average="macro")) # Macro F1 scores to better align with Kaggle’s evaluation criteria
            results[model_name]['f1_train_list'].append(f1_score(y_train, y_train_pred, average="macro")) # Macro F1 scores to better align with Kaggle’s evaluation criteria
            results[model_name]['f1_diff_list'].append(results[model_name]['f1_val_list'][-1] - results[model_name]['f1_train_list'][-1])
            n_iter = getattr(best_model, "n_iter_", [[0]])
            results[model_name]['n_iter_list'].append(n_iter[0] if isinstance(n_iter, list) else n_iter)
    
            # Print results
            print(
                f"{model_name}: "
                f"F1 Val = {results[model_name]['f1_val_list'][-1]:.3f}, "
                f"F1 Train = {results[model_name]['f1_train_list'][-1]:.3f}, "
                f"F1 Val-Train = {results[model_name]['f1_diff_list'][-1]:.3f}, "
                f"Time (s) = {results[model_name]['time_list'][-1]:.2f}, "
                f"Avg Iter = {np.mean(results[model_name]['n_iter_list'][-1]):.1f} "
                f"Best Params = {results[model_name]['best_params_list'][-1]}\n"
            )

    # Return results
    return results

# ----------------------------------------------------------------------------------------------------------------------
# Show CV Results
# ----------------------------------------------------------------------------------------------------------------------

def show_results(models_and_parameters, df, feats_dict):
    
    # Perform cross-validation
    results = cross_validation(
        models_and_parameters,
        df,
        feats_dict,
    )
    
    # Create an empty DataFrame to store the results
    results_df = pd.DataFrame(columns=["F1 Val", "F1 Train", "F1 Val-Train", "Time (s)", "Iterations", "Best Params"])
    
    for model_name in models_and_parameters.keys():
        # Compute average and standard deviation for each metric
        avg_val = f"{np.mean(results[model_name]['f1_val_list']):.3f} +/- {np.std(results[model_name]['f1_val_list']):.2f}"
        avg_train = f"{np.mean(results[model_name]['f1_train_list']):.3f} +/- {np.std(results[model_name]['f1_train_list']):.2f}"
        avg_diff = f"{np.mean(results[model_name]['f1_diff_list']):.3f} +/- {np.std(results[model_name]['f1_diff_list']):.2f}"
        avg_time = f"{np.mean(results[model_name]['time_list']):.3f} +/- {np.std(results[model_name]['time_list']):.2f}"
        avg_iter = f"{np.mean(results[model_name]['n_iter_list']):.1f} +/- {np.std(results[model_name]['n_iter_list']):.1f}"
    
        # Determine the mode of best parameters
        if all(not d for d in results[model_name]['best_params_list']):
            best_params_mode = None
        else:
            # Convert the list of dictionaries to a DataFrame
            best_params_df = pd.DataFrame(results[model_name]['best_params_list'])
            # Find the mode of parameter combinations
            best_params_mode = best_params_df.apply(lambda row: tuple(sorted(row.items())), axis=1).value_counts().idxmax()
            # Convert the tuple back to a dictionary
            best_params_mode = dict(best_params_mode)
    
        # Append results to the DataFrame, with model name as the index
        results_df.loc[model_name] = [avg_val, avg_train, avg_diff, avg_time, avg_iter, best_params_mode]
    
    # Extract the average values
    results_df["F1 Val_Sort"] = results_df["F1 Val"].str.extract(r'([-]?[0-9.]+)').astype(float)
    results_df["F1 Val-Train_Sort"] = results_df["F1 Val-Train"].str.extract(r'([-]?[0-9.]+)').astype(float)
    results_df["Time (s)_Sort"] = results_df["Time (s)"].str.extract(r'([-]?[0-9.]+)').astype(float)

    # Make the values absolute
    results_df["F1 Val_Sort"] = results_df["F1 Val_Sort"].abs()
    results_df["F1 Val-Train_Sort"] = results_df["F1 Val-Train_Sort"].abs()
    results_df["Time (s)_Sort"] = results_df["Time (s)_Sort"].abs()

    # Sort the DataFrame
    results_df = results_df.sort_values(
        by=["F1 Val_Sort", "F1 Val-Train_Sort", "Time (s)_Sort"],
        ascending=[False, True, True]
    )
    
    # Drop temporary sorting columns
    results_df = results_df.drop(columns=["F1 Val_Sort", "F1 Val-Train_Sort", "Time (s)_Sort"])
    
    return results, results_df

# ----------------------------------------------------------------------------------------------------------------------
# Final Predictions
# ----------------------------------------------------------------------------------------------------------------------

def final_predictions(df, df_test, best_model, feats_dict):
        # -------------------- Treatment --------------------

        # Separate features and target
        X_train = df.drop(['Claim Injury Type'],axis=1)
        y_train = df['Claim Injury Type']

        # Store original index of df_test because it will change after some operations
        original_index = df_test.index

        # Apply manual outlier removal
        X_train = manual_filter_outliers(X_train)
        y_train = y_train.loc[X_train.index]
    
        # Winsorization
        winsorization_bounds = {}
        for col in feats_dict["winsorization"]:
            # Calculate bounds using training data and apply to training set
            X_train, bounds = winsorization(X_train, col, train=X_train)
            # Apply the same bounds to the validation set
            df_test, _ = winsorization(df_test, col, bounds=bounds)
            # Store the calculated bounds
            winsorization_bounds[col] = bounds


        # Missing values imputation
        # Fit and apply imputation on the training set
        X_train, imputers = impute_missing_values(X_train, feats_dict, train=X_train)
        # Apply the same imputers to the validation set
        df_test, _ = impute_missing_values(df_test, feats_dict, imputers=imputers)

        # Feature engineering
        X_train = create_features(X_train)
        df_test = create_features(df_test)

        # Drop description and date columns
        X_train.drop(columns = feats_dict["codes_drop"] + feats_dict["descriptions_drop"] + feats_dict["dates_drop"], inplace=True)
        df_test.drop(columns = feats_dict["codes_drop"] + feats_dict["descriptions_drop"] + feats_dict["dates_drop"], inplace=True)
        
        # Ordinal encoding
        X_train, df_test, ordinal_encoders = ordinal_encoder(feats_dict, train=X_train, data=df_test)

        # Frequency encoding
        X_train, df_test, freq_encoders, fill_values_freq_encoding = frequency_encoder(feats_dict, train=X_train, data=df_test)

        # Data scaling
        X_train, df_test, scaler = scale_data(X_train, df_test)

        # Label encoding
        y_train, _, lencoder = label_encoder(y_train)

        # Feature selection
        selected_feats_spearman = spearman_selection(X_train, feats_dict)
        selected_feats_rfe = rfe_selection(X_train, y_train)
        selected_feats_dt = dt_selection(X_train, y_train)
        selected_feats_lasso = lasso_selection(X_train, y_train)

        selected_features_methods = {
            'spearman': selected_feats_spearman,
            'rfe': selected_feats_rfe,
            'dt': selected_feats_dt,
            'lasso': selected_feats_lasso,
        }
        final_selected_features = voting_feature_selection(X_train, selected_features_methods)
        X_train = X_train[final_selected_features]
        df_test = df_test[final_selected_features]

        # -------------------- Prediction --------------------

        # Train model
        best_model.fit(X_train, y_train)

        # Predict the 'Claim Injury Type' for the test dataset by using the trained model
        df_test['Claim Injury Type'] = best_model.predict(df_test)

        # Decode the predicted labels back to their original categorical values
        df_test['Claim Injury Type'] = lencoder.inverse_transform(df_test['Claim Injury Type'])

        # Restore the original index before preparing the submission DataFrame
        df_test.index = original_index
        
        # Prepare the submission DataFrame with the necessary columns (reset the index to treat 'Claim Identifier' as a column)
        df_test = df_test.reset_index()[['Claim Identifier', 'Claim Injury Type']]

        return df_test, {
            "winsorization_bounds": winsorization_bounds,
            "imputers": imputers,
            "ordinal_encoders": ordinal_encoders,
            "freq_encoders": freq_encoders,
            "fill_values_freq_encoding": fill_values_freq_encoding,
            "scaler": scaler,
            "lencoder": lencoder,
            "final_selected_features": final_selected_features
        }