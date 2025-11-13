# %% [markdown]
# # Load data

# %%
# Step 1: Import Libraries and Load Data  
from functools import partial
import json
import os
import sys
import pandas as pd
import numpy as np
# Import all necessary libraries
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
# import matplotlib.pyplot as plt
# import seaborn as sns

def loadData(file_path):
    # %%
    """
    Load the dataset from a CSV file.
    """
    try:
        cwd_path_os = os.getcwd()
        abspath = cwd_path_os +"/" + file_path
        print("abs file path:", abspath)
        df = pd.read_csv(abspath)
        print(f"Data loaded successfully from {file_path}")

        # Display the original shape of the dataframe
        print(f"Original dataframe shape (rows, columns): {df.shape}")

        return df
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    


# %% [markdown]
# # Data preprocessing

# %% [markdown]
# ## Handle missing values

# %%
#  Step 2: Handle Critical Missing Values
def cleanData(df, is_prediction=False, cols_to_drop = None):
    # %%
    # Create a copy to avoid SettingWithCopyWarning
    df_cleaned = df.copy()

    # Drop rows where the target variable 'CHURNED' is missing
    df_cleaned.dropna(subset=['CHURNED'], inplace=True)
    print(f"Shape after dropping rows with missing CHURNED: {df_cleaned.shape}")

    # Drop rows identified as "data ghosts" (where CUSTOMER_NUMBER or PRODUCT_TYPE is missing)

    df_cleaned.dropna(subset=['CUSTOMER_NUMBER', 'PRODUCT_TYPE'], inplace=True)
    print(f"Shape after dropping 'data ghosts': {df_cleaned.shape}")

    # Display the first few rows of the cleaned data
    print("\nPreview of the data after initial cleaning:")
    df_cleaned.head()

    # Step 3: Impute Remaining Missing Values 

    # Identify columns for different imputation strategies

    # Columns to be filled with 0 (usage-related features)
    fill_zero_cols = [col for col in df_cleaned.columns if 'IFP_' in col or \
                    'NUM_NZ_AUS_SMS_' in col or 'SUM_30D_' in col or '_COUNT' in col]

    # Columns to be filled with median (other numeric features)
    # We exclude columns we already handled or will handle differently
    numeric_cols = df_cleaned.select_dtypes(include=np.number).columns.tolist()
    fill_median_cols = [col for col in numeric_cols if col not in fill_zero_cols + ['CHURNED']]

    # Column to be filled with mode (categorical features)
    fill_mode_cols = ['COVERAGE_AREA']


    #   Apply the imputation  

    # Fill with 0
    for col in fill_zero_cols:
        df_cleaned[col] = df_cleaned[col].fillna(0)
    print(f"Filled {len(fill_zero_cols)} usage-related columns with 0.")


    # Fill with median
    for col in fill_median_cols:
        median_val = df_cleaned[col].median()
        df_cleaned[col] = df_cleaned[col].fillna(median_val)
    print(f"Filled {len(fill_median_cols)} numeric columns with their median.")

    # Fill with mode
    # .mode() returns a Series, so we take the first element [0]
    for col in fill_mode_cols:
        mode_val = df_cleaned[col].mode()[0]
        df_cleaned[col] = df_cleaned[col].fillna(mode_val)
    print(f"Filled {len(fill_mode_cols)} categorical column(s) with their mode.")


    #Final Verification 

    # Check if there are any missing values left
    missing_values_count = df_cleaned.isnull().sum().sum()
    print(f"\nTotal remaining missing values: {missing_values_count}")

    if missing_values_count == 0:
        print("\n All missing values have been successfully handled.")
    else:
        print("\nWarning: There are still some missing values left. Please check the columns.")
        print(df_cleaned.isnull().sum())

    # ## Customer-level aggregation

    # How many customers in the dataset hold multiple connections?

    # Group by customer ID and calculate the number of unique connections per customer
    connection_counts_per_customer = df_cleaned.groupby('CUSTOMER_NUMBER')['CONNECTION_NUMBER'].nunique()

    # Filter out customers who have more than one connection
    customers_with_multiple_connections = connection_counts_per_customer[connection_counts_per_customer > 1]

    #   Statistics and Output Results  

    # Calculate the number of multi-connection customers
    num_multi_connection = len(customers_with_multiple_connections)

    # Calculate the total number of unique customers
    total_customers = df_cleaned['CUSTOMER_NUMBER'].nunique()

    # Calculate the percentage of multi-connection customers
    percentage = (num_multi_connection / total_customers) * 100

    # Display formatted output
    print("\n" + "="*30)
    print("      Analysis Results ")
    print("="*30)
    print(f"Total number of unique customers : {total_customers}")
    print(f"Number of customers with multiple connections {num_multi_connection}")
    print(f"Percentage of multi-connection customers : {percentage:.2f}%")
    print("="*30)

    # How many customers in the dataset have two different churned values within the same month?”

    # We use the 'df_cleaned' dataframe, which is before customer-level aggregation

    # Group by month and customer, then count the number of unique CHURNED values for each group
    customer_churn_variance = df_cleaned.groupby(['SNAPSHOT_DATE', 'CUSTOMER_NUMBER'])['CHURNED'].nunique()
    conflicting_customers = customer_churn_variance[customer_churn_variance > 1]

    pd.set_option('display.max_rows', None)   
    pd.set_option('display.max_columns', None) 
    pd.set_option('display.width', None)     
    pd.set_option('display.max_colwidth', None) 

    if conflicting_customers.empty:
        print(" Verification Passed: No customer has conflicting churn statuses within the same month.")
    else:
        print(" Verification Failed: The following customers have conflicting churn statuses:")
        print(conflicting_customers)

    #  Step 1: Identify Conflicting Customers 
    print("Step 1: Identifying conflicting customers")

    # Group by month and customer, then count the number of unique churn values
    customer_churn_variance = df_cleaned.groupby(['SNAPSHOT_DATE', 'CUSTOMER_NUMBER'])['CHURNED'].nunique()

    # Select customers with more than one unique churn value (i.e., both 0 and 1)
    conflicting_customers = customer_churn_variance[customer_churn_variance > 1]


    #  Step 2: Separate Conflicting Customers 
    print("\n--- Step 2: Separating conflicting customers ---")

    # Extract conflicting customer IDs from the MultiIndex
    conflicting_customer_ids = conflicting_customers.index.get_level_values('CUSTOMER_NUMBER').unique()

    # Create two DataFrames: one for conflicting customers, one for non-conflicting customers
    df_conflicting = df_cleaned[df_cleaned['CUSTOMER_NUMBER'].isin(conflicting_customer_ids)].copy()
    df_non_conflicting = df_cleaned[~df_cleaned['CUSTOMER_NUMBER'].isin(conflicting_customer_ids)].copy()

    # Print summary of separation
    print(f"Successfully extracted all records of {df_conflicting['CUSTOMER_NUMBER'].nunique()} conflicting customers.")
    print(f"Shape of conflicting customer dataset: {df_conflicting.shape}")
    print(f"Shape of non-conflicting customer dataset: {df_non_conflicting.shape}")


    #  Step 3: Generate Statistical Comparison 
    print("\n--- Step 3: Generating statistical comparison ---")

    # Define columns to analyze
    analysis_cols = [
        'STANDARD_UNIT_PRICE',
        'SUM_30D_DATA_CHARGED_WEEKDAY_MB',
        'SUM_30D_DATA_CHARGED_WEEKNIGHT_MB',
        'SUM_30D_DATA_CHARGED_WEEKENDDAY_MB',
        'SUM_30D_DATA_CHARGED_WEEKENDNIGHT_MB',
        'SUM_30D_VOICE_CHARGED_MIN',
        'TENURE_IN_YRS_FROM_ACTIVATION_DATE',
        'POSTPAID_COUNT',
        'BROADBAND_COUNT'
    ]

    # Keep only columns that actually exist in the dataset
    analysis_cols_exist = [col for col in analysis_cols if col in df_conflicting.columns]

    # Generate descriptive statistics for both groups
    conflicting_stats = df_conflicting[analysis_cols_exist].describe()
    non_conflicting_stats = df_non_conflicting[analysis_cols_exist].describe()

    # Print the comparison tables
    print("\n" + "="*60)
    print("Statistical Summary for Conflicting Customers")
    print("="*60)
    print(conflicting_stats)

    print("\n" + "="*60)
    print("Statistical Summary for Non-Conflicting Customers")
    print("="*60)
    print(non_conflicting_stats)


    #  Step 4: Export Conflicting Customer Data 
    print("\n--- Step 4: Exporting conflicting customer data to CSV ---")

    # Set output file name
    output_filename = 'conflicting_churn_customers.csv'

    # Export to CSV file
    df_conflicting.to_csv(output_filename, index=False, encoding='utf-8-sig')

    # Confirm export success
    print(f"\nSuccess! Exported {df_conflicting.shape[0]} rows of conflicting customer data to file: {output_filename}")

    #   Step 1: Aggregate data from connection-level to customer-level  
    # Objective: Create a single summary record for each customer's activity per month for customer-level churn prediction.
    # Method: Group by 'SNAPSHOT_DATE' and 'CUSTOMER_NUMBER', and apply aggregation functions to all connection records for each customer.

    print("Starting customer-level aggregation")
    print(f"Shape before aggregation (connection-level): {df_cleaned.shape}")

    # Define the aggregation rules in a dictionary
    # - For numeric features like usage or product counts, use 'sum' to get the total.
    # - For customer tenure, use 'max' to reflect the longest service period.
    # - For categorical features, use 'first'（'PRODUCT_TYPE'） or 'mode'（COVERAGE_AREA）.
    # - For the target variable 'CHURNED', use 'max'. This ensures that if any connection for a customer has churned, the customer is marked as churned.
    aggregation_logic = {
        # Target and Categorical Features
        'CHURNED': 'max',
        'COVERAGE_AREA': lambda x: x.mode()[0] if not x.mode().empty else None, # Fill with the first mode, 'None' if there's no mode
        'PRODUCT_TYPE': 'first',

        # Customer Attribute Features
        'TENURE_IN_YRS_FROM_ACTIVATION_DATE': 'max',
        'STANDARD_UNIT_PRICE': 'sum',

        # Product Holding Count Features
        'FIBRE_COUNT': 'sum',
        'BROADBAND_COUNT': 'sum',
        'VDSL_COUNT': 'sum',
        'POSTPAID_COUNT': 'sum',
        'VOICEONLY_COUNT': 'sum',
        'COPPER_COUNT': 'sum',
        'PREPAID_COUNT': 'sum',
        'ADSL_COUNT': 'sum',

        # Installment Plan Features
        'IFP_COMPLETED_MONTHS': 'sum',
        'IFP_REMAINING_MONTHS': 'sum',

        # 30-Day Usage Features
        'SUM_30D_DATA_CHARGED_WEEKDAY_MB': 'sum',
        'SUM_30D_DATA_CHARGED_WEEKNIGHT_MB': 'sum',
        'SUM_30D_DATA_CHARGED_WEEKENDDAY_MB': 'sum',
        'SUM_30D_DATA_CHARGED_WEEKENDNIGHT_MB': 'sum',
        'SUM_30D_VOICE_CHARGED_MIN': 'sum',
        'NUM_NZ_AUS_SMS_0W_4W': 'sum'
    }

    # Perform the groupby and aggregation operation
    df_customer_level = df_cleaned.groupby(['SNAPSHOT_DATE', 'CUSTOMER_NUMBER']).agg(aggregation_logic).reset_index()

    print(f"\nShape after aggregation (customer-level): {df_customer_level.shape}")
    print(f"\nShape after aggregation (customer-level.head()): {df_customer_level.head()}")
    print("Customer-level aggregation complete.")


    # ## Split data by months

    #   Step 2 : Split dataset directly based on month labels  
    # Strategy: The 'SNAPSHOT_DATE' column already contains labels like 'Month 1', 'Month 2'. We will use these labels directly for splitting.

    print("\n\nSplitting dataset based on month labels")

    try:
        # Find all unique month labels in the column and sort them to ensure 'Month 1' comes first.
        unique_month_labels = sorted(df_customer_level['SNAPSHOT_DATE'].unique())

        # Check if there are at least two month labels for splitting
        if len(unique_month_labels) >= 2 and is_prediction == False:
            month_label_1 = unique_month_labels[0]
            month_label_2 = unique_month_labels[1]
            month_label_3 = unique_month_labels[2] if len(unique_month_labels) > 2 else None

            # Directly filter the data based on the string labels to create train and test sets
            # train_df = df_customer_level[((df_customer_level['SNAPSHOT_DATE'] == month_label_1)
            #                                 | (df_customer_level['SNAPSHOT_DATE'] == month_label_2))].copy()

            train_df = df_customer_level[((df_customer_level['SNAPSHOT_DATE'] == month_label_1))].copy()

            test_df = df_customer_level[df_customer_level['SNAPSHOT_DATE'] == month_label_2].copy()

            predict_df = df_customer_level[df_customer_level['SNAPSHOT_DATE'] == month_label_3].copy() if month_label_3 else None

            print(f"Training set month label: {month_label_3}, shape: {predict_df.shape}")
            filtered_df = df[df['SNAPSHOT_DATE'] == month_label_3].copy() if month_label_3 else None

            # --- 4. Save to CSV file ---
            output_filename = 'predict_customers.csv'
            filtered_df.to_csv(output_filename, index=False)

            print(f"filtered_predict_df shape: {filtered_df.shape}")

            print(f"Training set ({month_label_1}): {train_df.shape}")
            print(f"Testing set ({month_label_2}): {test_df.shape}")
            print("Time-based dataset split complete.")
        elif is_prediction == True:
            train_df = df_customer_level.copy()
            test_df = df_customer_level.copy()
            print(f"Testing set (all months): {test_df.shape}")
        else:
            print("Error: Could not find at least two unique month labels in the 'SNAPSHOT_DATE' column to split the data.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    # ## Feature engineering

    # Step 3 : Advanced Feature Engineering 
    # Objective: Create new, potentially more predictive variables from existing features to enhance model performance.

    print("Starting advanced feature engineering")


    # Apply the feature engineering function to the training and testing sets separately
    train_df_eng = create_basic_features(train_df)
    test_df_eng = create_basic_features(test_df)

    # Non-linear Features
    # Binning must be learned from the training set and then applied to both  

    # 1. Binning for Customer Tenure
    # Use qcut (quantile-based cut) to divide the tenure in the training set into 4 tiers
    try:
        tenure_bins = pd.qcut(train_df_eng['TENURE_IN_YRS_FROM_ACTIVATION_DATE'], q=4, retbins=True, duplicates='drop')[1]
        tenure_labels = [f'Tenure_Group_{i}' for i in range(len(tenure_bins)-1)]
        train_df_eng['Tenure_Group'] = pd.cut(train_df_eng['TENURE_IN_YRS_FROM_ACTIVATION_DATE'], bins=tenure_bins, labels=tenure_labels, include_lowest=True)
        print("\nVerifying Tenure Group counts in Training Data (Month 1) AFTER original binning:")
        print(train_df_eng['Tenure_Group'].value_counts())

        test_df_eng['Tenure_Group'] = pd.cut(test_df_eng['TENURE_IN_YRS_FROM_ACTIVATION_DATE'], bins=tenure_bins, labels=tenure_labels, include_lowest=True)
        print("Tenure binned successfully.")
    except Exception as e:
        print(f"Could not bin tenure: {e}")

    # 2. Binning for Standard Unit Price (Robust Version) 
    try:
        # --- Step 2a: Calculate Quantile Boundaries from Training Data (>0) ---
        # Ensure we only use non-zero prices for quantile calculation, as in EDA
        non_zero_prices_train = train_df_eng.loc[train_df_eng['STANDARD_UNIT_PRICE'] > 0, 'STANDARD_UNIT_PRICE']

        if not non_zero_prices_train.empty:
            # Calculate 25th, 50th, and 75th percentiles
            p25 = non_zero_prices_train.quantile(0.25)
            p50 = non_zero_prices_train.quantile(0.50)
            p75 = non_zero_prices_train.quantile(0.75)
            # We also need min and max for defining bins
            min_price = non_zero_prices_train.min() # Should be > 0
            max_price = non_zero_prices_train.max()

            print(f"Calculated quantiles on non-zero prices (Month 1): min={min_price:.4f}, p25={p25:.4f}, p50={p50:.4f}, p75={p75:.4f}, max={max_price:.4f}")

            # Step 2b: Define Bin Edges Manually
            # Use slightly adjusted bounds to handle potential overlaps and include all values
            # Include -inf to capture 0 prices if any exist after aggregation, though ideally shouldn't be many
            bin_edges = [-np.inf, p25, p50, p75, np.inf]
            # Ensure edges are unique; if quantiles are identical, pd.cut might behave unexpectedly otherwise.
            # For simplicity here, we assume p25 < p50 < p75. If they can be equal, more complex logic needed.
            print(f"Using bin edges: {bin_edges}")


            # --- Step 2c: Define Bin Labels ---
            bin_labels = ['Lowest (0-25%)', 'Low (25-50%)', 'Medium (50-75%)', 'Highest (75-100%)']

            # Step 2d: Apply pd.cut to Train and Test Sets ----
            # Apply to training set
            train_df_eng['Price_Tier'] = pd.cut(train_df_eng['STANDARD_UNIT_PRICE'],
                                                bins=bin_edges,
                                                labels=bin_labels,
                                                right=True, # Intervals are (edge1, edge2]
                                                include_lowest=True) # Include the lowest edge

            # Apply the SAME BINS learned from training set to the test set
            test_df_eng['Price_Tier'] = pd.cut(test_df_eng['STANDARD_UNIT_PRICE'],
                                            bins=bin_edges,
                                            labels=bin_labels,
                                            right=True,
                                            include_lowest=True)

            print("Price binned successfully using manual edges.")

            # --- Step 2e: Verify Bin Counts on Training Data ---
            print("\nVerifying Price Tier counts in Training Data (Month 1):")
            print(train_df_eng['Price_Tier'].value_counts())

        else:
            print("Warning: No non-zero prices found in training data to calculate quantiles. Price tier not created.")
            # Create dummy columns to avoid errors later
            train_df_eng['Price_Tier'] = 'Undefined'
            test_df_eng['Price_Tier'] = 'Undefined'

    except Exception as e:
        print(f"An error occurred during robust price binning: {e}")
        # Create dummy columns to avoid errors later
        train_df_eng['Price_Tier'] = 'Error'
        test_df_eng['Price_Tier'] = 'Error'


    print(f"\nShape of training data after feature engineering: {train_df_eng.shape}")
    print(f"Shape of testing data after feature engineering: {test_df_eng.shape}")
    print("\nAdvanced feature engineering complete.")

    # Step_additional: Handle Right-Skewed Distributions and Outliers
    # This section caps extreme outliers and applies log1p transformation
    # to stabilize numeric scales and reduce skewness for usage-related variables.

    print("\n Handling right-skewed usage features and outliers")

    # Identify usage-related numeric features
    usage_cols = [c for c in train_df_eng.columns 
                if c.startswith("SUM_30D_DATA") or c.startswith("IFP_")]

    print(f"Targeted columns ({len(usage_cols)}): {usage_cols}")

    # Keep a copy of pre-transform data for comparison (optional)
    train_df_raw = train_df_eng.copy()

    # Quantify baseline info
    n_before = len(train_df_eng)
    print(f"\nTotal records before processing: {n_before:,}")

    # Calculate Winsorization thresholds (1st–99th percentiles on training data)
    q_low = train_df_eng[usage_cols].quantile(0.01)
    q_hi  = train_df_eng[usage_cols].quantile(0.99)

    # Count how many values are outside this range before clipping
    outlier_counts = {}
    for c in usage_cols:
        count_low = (train_df_eng[c] < q_low[c]).sum()
        count_hi  = (train_df_eng[c] > q_hi[c]).sum()
        outlier_counts[c] = count_low + count_hi

    print("\nApprox. number of values to be clipped by Winsorization:")
    for c, v in outlier_counts.items():
        print(f"  {c}: {v:,} values ({v / n_before:.2%} of total)")

    # Apply Winsorization (cap at 1%–99%)
    train_df_eng[usage_cols] = train_df_eng[usage_cols].clip(q_low, q_hi, axis=1)
    test_df_eng[usage_cols]  = test_df_eng[usage_cols].clip(q_low, q_hi, axis=1)

    # Apply log1p transform (ensure non-negative values first)
    train_df_eng[usage_cols] = np.log1p(train_df_eng[usage_cols].clip(lower=0))
    test_df_eng[usage_cols]  = np.log1p(test_df_eng[usage_cols].clip(lower=0))

    # Verify data consistency after processing
    n_after = len(train_df_eng)
    nan_counts = train_df_eng[usage_cols].isna().sum().sum()
    inf_counts = np.isinf(train_df_eng[usage_cols]).sum().sum()

    print(f"\nTotal records after processing: {n_after:,}")
    if n_before == n_after:
        print(" No customer rows were removed during processing.")
    else:
        print(f"⚠️ {n_before - n_after:,} rows were dropped (please verify cause).")

    print(f"NaN introduced by log1p: {nan_counts}, Infinite values: {inf_counts}")

    # # Optional: visualize post-processing distribution
    # import matplotlib.pyplot as plt
    # train_df_eng[usage_cols].hist(bins=40, figsize=(12, 8))
    # plt.suptitle("After Winsorization + Log1p", fontsize=14)
    # plt.tight_layout()
    # plt.show()

    # print("\n Winsorization + Log1p completed successfully")


    # Step 4: Final Data Preparation (Handling New Features) 
    # Objective: Process the feature-engineered dataframes into the final format ready for model input.

    print("\n\nPreparing final data for modeling (incorporating new features)")

    # We now use the feature-engineered dataframes
    target = 'CHURNED'

    X_train = train_df_eng.drop(columns=[target])
    y_train = train_df_eng[target]
    X_test = test_df_eng.drop(columns=[target])
    y_test = test_df_eng[target]

    # Identify categorical columns for one-hot encoding (including our newly created ones)
    categorical_cols_to_encode = ['COVERAGE_AREA', 'Tenure_Group', 'Price_Tier']

    # Robustly check which columns actually exist, then encode them
    actual_categorical_cols = [col for col in categorical_cols_to_encode if col in X_train.columns]
    print(f"Columns identified for one-hot encoding: {actual_categorical_cols}")

    X_train = pd.get_dummies(X_train, columns=actual_categorical_cols, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=actual_categorical_cols, drop_first=True)

    # Remove identifier columns AND the constant column 'PRODUCT_TYPE'
    if cols_to_drop is not None:
        # Use errors='ignore' to ensure no error is raised if a column doesn't exist
        X_train = X_train.drop(columns=cols_to_drop, errors='ignore')
        X_test = X_test.drop(columns=cols_to_drop, errors='ignore')

    # Align the feature columns of the training and testing sets
    train_cols = X_train.columns
    test_cols = X_test.columns
    missing_in_test = set(train_cols) - set(test_cols)
    for c in missing_in_test:
        X_test[c] = 0
    missing_in_train = set(test_cols) - set(train_cols)
    for c in missing_in_train:
        X_train[c] = 0
    X_test = X_test[train_cols]

    print("\nFinal data dimensions ready for modeling:")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_test: {y_test.shape}")
    print("\nData preparation complete. Final datasets for model training are now generated.")

    return (train_df, test_df, X_train, y_train, X_test, y_test)

# %%
# Create a function to generate features that can be safely applied to any dataset
def create_basic_features(df):

    if df is None or df.empty:
        print("Warning: Input dataframe is None or empty. Returning empty dataframe.")
        return pd.DataFrame()
    # %%
    df_eng = df.copy()

    # Business-driven Features
    # Calculate total IFP (Installment Plan) contract duration
    df_eng['Total_IFP_Months'] = df_eng['IFP_COMPLETED_MONTHS'] + df_eng['IFP_REMAINING_MONTHS']

    # Calculate total data usage in the last 30 days
    data_usage_cols = ['SUM_30D_DATA_CHARGED_WEEKDAY_MB', 'SUM_30D_DATA_CHARGED_WEEKNIGHT_MB',
                       'SUM_30D_DATA_CHARGED_WEEKENDDAY_MB', 'SUM_30D_DATA_CHARGED_WEEKENDNIGHT_MB']
    df_eng['Total_30D_Data_Usage'] = df_eng[data_usage_cols].sum(axis=1)

    # Deeper Signal Features
    # Calculate the completion progress ratio of the IFP contract
    # Use .fillna(0) to handle cases where the denominator is 0 (i.e., no IFP contract)
    df_eng['IFP_Progress_Ratio'] = (df_eng['IFP_COMPLETED_MONTHS'] / df_eng['Total_IFP_Months']).fillna(0)
    
    # Calculate the interaction between price and remaining IFP months, potentially reflecting remaining contract value
    df_eng['Price_x_Remaining_Months'] = df_eng['STANDARD_UNIT_PRICE'] * df_eng['IFP_REMAINING_MONTHS']

    return df_eng

# %% [markdown]
# # Modelling

# %%
#   Step 6 : Complete Model Comparison Experiment (Basic vs. Engineered Features)  



def prepare_datasets(train_df, test_df, X_train, y_train, X_test, y_test):
    #   1. Prepare Both Datasets  

    print("  Preparing Basic and Advanced Feature Sets  ")

    # A. Prepare Basic Feature Set (without advanced feature engineering)
    target = 'CHURNED'
    X_train_basic = train_df.drop(columns=[target])
    y_train_basic = train_df[target]
    X_test_basic = test_df.drop(columns=[target])
    y_test_basic = test_df[target]

    # Apply necessary one-hot encoding and column cleanup for the basic set
    categorical_cols_basic = ['COVERAGE_AREA', 'PRODUCT_TYPE']
    actual_categorical_basic = [col for col in categorical_cols_basic if col in X_train_basic.columns]
    X_train_basic = pd.get_dummies(X_train_basic, columns=actual_categorical_basic, drop_first=True)
    X_test_basic = pd.get_dummies(X_test_basic, columns=actual_categorical_basic, drop_first=True)
    cols_to_drop_basic = ['SNAPSHOT_DATE', 'CUSTOMER_NUMBER']
    X_train_basic = X_train_basic.drop(columns=cols_to_drop_basic, errors='ignore')
    X_test_basic = X_test_basic.drop(columns=cols_to_drop_basic, errors='ignore')
    X_train_basic, X_test_basic = X_train_basic.align(X_test_basic, join='inner', axis=1, fill_value=0)

    # B. Prepare Advanced Feature Set (already prepared)
    # Variables already exist, just renaming for clarity
    X_train_advanced = X_train
    y_train_advanced = y_train
    X_test_advanced = X_test
    y_test_advanced = y_test

    print(f"Basic feature set shape: {X_train_basic.shape}")
    print(f"Advanced feature set shape: {X_train_advanced.shape}")

    #   3. Run the 6 Experiments  

    datasets = {
        "Basic Features": (X_train_basic, y_train_basic, X_test_basic, y_test_basic),
        "Advanced Features": (X_train_advanced, y_train_advanced, X_test_advanced, y_test_advanced)
    }

    for name, (X_train_data, y_train_data, X_test_data, y_test_data) in datasets.items():
        print(f"\n\n{'='*25} Running Experiments on: {name} {'='*25}")
        
        # Strategy A: No Handling (Baseline)
        model_base = LogisticRegression(solver='liblinear', random_state=42)
        model_base.fit(X_train_data, y_train_data)
        y_pred_base = model_base.predict(X_test_data)
        y_proba_base = model_base.predict_proba(X_test_data)[:, 1]
        evaluate_model(y_test_data, y_pred_base, y_proba_base, model_name=f'{name} - A. Baseline')

        # Strategy B: Class Weighting
        model_weighted = LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42)
        model_weighted.fit(X_train_data, y_train_data)
        y_pred_weighted = model_weighted.predict(X_test_data)
        y_proba_weighted = model_weighted.predict_proba(X_test_data)[:, 1]
        evaluate_model(y_test_data, y_pred_weighted, y_proba_weighted, model_name=f'{name} - B. Class Weighting')

        # Strategy C: SMOTE Oversampling
        pipeline = Pipeline([
            ('smote', SMOTE(random_state=42)),
            ('classifier', LogisticRegression(solver='liblinear', random_state=42))
        ])
        pipeline.fit(X_train_data, y_train_data)
        y_pred_smote = pipeline.predict(X_test_data)
        y_proba_smote = pipeline.predict_proba(X_test_data)[:, 1]
        evaluate_model(y_test_data, y_pred_smote, y_proba_smote, model_name=f'{name} - C. SMOTE')

    return (train_df, test_df, X_train, y_train, X_test, y_test)
        

#   2. Define Evaluation Function  
def evaluate_model(y_true, y_pred, y_proba, model_name=''):
    print(f"\n  {model_name}  ")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, digits=4))
    auc_score = roc_auc_score(y_true, y_proba)
    print(f"ROC AUC Score: {auc_score:.4f}")
    # cm = confusion_matrix(y_true, y_pred)
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Churned', 'Churned'], yticklabels=['Not Churned', 'Churned'])
    # plt.ylabel('Actual')
    # plt.xlabel('Predicted')
    # plt.title(f'{model_name} Confusion Matrix')
    # plt.show()



# %%
#   Step 7: Train and Evaluate an Advanced Model (LightGBM)  

# Import the LightGBM model
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def train_models(X_train, y_train, X_test, y_test):
    #   1. Train the LightGBM Model with Advanced Features
    print("  1. Training LightGBM Model with Advanced Features  ")
    # Handling Class Imbalance for LightGBM
    # For LightGBM, we use the 'scale_pos_weight' parameter instead of 'class_weight'.
    # It is calculated as: (count of negative class / count of positive class)
    neg_count = y_train.value_counts()[0]
    pos_count = y_train.value_counts()[1]
    scale_pos_weight_value = neg_count / pos_count
    print(f"Calculated scale_pos_weight for LightGBM: {scale_pos_weight_value:.2f}")

    # Initialize the LightGBM Classifier
    lgb_clf = lgb.LGBMClassifier(objective='binary',
                                scale_pos_weight=scale_pos_weight_value,
                                random_state=42)

    # Train the model on the enhanced training data
    lgb_clf.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred_lgb = lgb_clf.predict(X_test)
    y_pred_proba_lgb = lgb_clf.predict_proba(X_test)[:, 1]

    #   2. Evaluate the LightGBM Model  
    print("\n  Evaluation for LightGBM with Advanced Features  ")

    # Print Classification Report
    print("Classification Report:")
    print(sklearn.metrics.classification_report(y_test, y_pred_lgb, digits=4))

    # Calculate and Print ROC AUC Score
    auc_lgb = sklearn.metrics.roc_auc_score(y_test, y_pred_proba_lgb)
    print(f"ROC AUC Score: {auc_lgb:.4f}")

    # Display Confusion Matrix
    cm_lgb = sklearn.metrics.confusion_matrix(y_test, y_pred_lgb)
    print("\nConfusion Matrix:", cm_lgb)
    # sns.heatmap(cm_lgb, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Churned', 'Churned'], yticklabels=['Not Churned', 'Churned'])
    # plt.ylabel('Actual')
    # plt.xlabel('Predicted')
    # plt.title('LightGBM Confusion Matrix (Advanced Features)')
    # plt.show()

    # %%
    #   Step 8: Train and Evaluate XGBoost and Decision Tree Models  

    # Import the XGBoost and Decision Tree models
    from xgboost import XGBClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

    # We will continue to use the evaluate_model function defined previously

    #   8.1: Train and Evaluate the XGBoost Model  
    print("== 1. Training XGBoost Model ==")
    print("== 1. XGBoost ==")

    # XGBoost also uses the 'scale_pos_weight' parameter to handle imbalance
    # We use the same value calculated earlier for LightGBM
    xgb_clf = XGBClassifier(objective='binary:logistic',
                            scale_pos_weight=scale_pos_weight_value,
                            use_label_encoder=False,
                            eval_metric='logloss',
                            random_state=42)

    # Train the model
    xgb_clf.fit(X_train, y_train)

    # Make predictions
    y_pred_xgb = xgb_clf.predict(X_test)
    y_proba_xgb = xgb_clf.predict_proba(X_test)[:, 1]

    # Evaluate the model
    evaluate_model(y_test, y_pred_xgb, y_proba_xgb, model_name='XGBoost')


    #   8.2: Train and Evaluate the Decision Tree Model  
    print("\n\n== 2. Training Decision Tree Model ==")

    # The Decision Tree uses the 'class_weight' parameter, same as Logistic Regression
    dt_clf = DecisionTreeClassifier(class_weight='balanced',
                                    random_state=42)

    # Train the model
    dt_clf.fit(X_train, y_train)

    # Make predictions
    y_pred_dt = dt_clf.predict(X_test)
    y_proba_dt = dt_clf.predict_proba(X_test)[:, 1]

    # Evaluate the model
    evaluate_model(y_test, y_pred_dt, y_proba_dt, model_name='Decision Tree')

    # %%
    #   Step 9: Final Performance Comparison of All Models  

    # Import necessary libraries
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score
    from sklearn.linear_model import LogisticRegression

    #   1. Re-gather predictions from all models  
    # To ensure a fair comparison, we re-run Logistic Regression to get results on the latest test set
    model_lr_final = LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42)
    model_lr_final.fit(X_train, y_train)
    y_pred_lr_final = model_lr_final.predict(X_test)
    y_proba_lr_final = model_lr_final.predict_proba(X_test)[:, 1]

    # y_pred_lgb, y_proba_lgb, y_pred_xgb, y_proba_xgb, y_pred_dt, y_proba_dt
    # Predictions from other models already exist in variables

    #   2. Calculate key metrics for each model  

    models = {
        'Logistic Regression': (y_pred_lr_final, y_proba_lr_final),
        'Decision Tree': (y_pred_dt, y_proba_dt),
        'XGBoost': (y_pred_xgb, y_proba_xgb),
        'LightGBM': (y_pred_lgb, y_pred_proba_lgb)
    }

    results = []
    for model_name, (y_pred, y_proba) in models.items():
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        results.append({
            'Model': model_name,
            'Recall': recall,
            'Precision': precision,
            'F1-Score': f1,
            'ROC AUC': auc
        })

    #   3. Create and display the comparison table  

    results_df = pd.DataFrame(results).set_index('Model').sort_values(by='Recall', ascending=False)
    print("  Model Performance Comparison Table  ")
    # print(json.dumps(results_df.round(4).to_dict(), indent=4))
    # display(results_df.round(4))
    print(results_df.round(4))

#   4. Create comparison plots  

# fig, axes = plt.subplots(2, 1, figsize=(12, 10))
# fig.suptitle('Model Performance Comparison', fontsize=16)

# # Plot Recall comparison
# sns.barplot(x=results_df.index, y='Recall', data=results_df, ax=axes[0], palette='viridis')
# axes[0].set_title('Recall Score Comparison', fontsize=14)
# axes[0].set_ylabel('Recall')
# axes[0].bar_label(axes[0].containers[0], fmt='%.4f')

# # Plot ROC AUC comparison
# sns.barplot(x=results_df.index, y='ROC AUC', data=results_df, ax=axes[1], palette='plasma')
# axes[1].set_title('ROC AUC Score Comparison', fontsize=14)
# axes[1].set_ylabel('ROC AUC Score')
# axes[1].set_yticklabels([f'{x:.2f}' for x in axes[1].get_yticks()])
# axes[1].bar_label(axes[1].containers[0], fmt='%.4f')


# plt.tight_layout(rect=[0, 0, 1, 0.96])
# # Save the comparison plot to a file
# plt.savefig('model_comparison.png')
# print("\nComparison plot saved as 'model_comparison.png'")
# plt.show()

# %%
#   Final Step (Forced Training): Using Fixed Iterations to Ensure Full Training  

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

def final_model_training(X_train, y_train, X_test, y_test, dir = "./", model1_name = 'lgbm_churn_model.joblib', model2_name = 'lgbm_churn_model_optuna-tuned.joblib'):
    #   1. Data Preparation: Create Final Training and Test Sets
    print("  1. Data Preparation: Creating Training, Validation, and Final Test Sets  ")

    # We no longer need a validation set for early stopping, but splitting the data is still good practice.
    X_train_final = X_train
    y_train_final = y_train
    X_test_final = X_test
    y_test_final = y_test

    print("  2. Final Model Training with a Fixed Number of Iterations  ")

    scale_pos_weight_value = y_train_final.value_counts()[0] / y_train_final.value_counts()[1]

    # Initialize the model, using the default number of estimators (100) which worked well before.
    final_model = lgb.LGBMClassifier(objective='binary',
                                    scale_pos_weight=scale_pos_weight_value,
                                    random_state=42,
                                    n_estimators=100,  
                                    learning_rate=0.1,
                                    num_leaves=31)

    print("Starting model training for a fixed 100 iterations")

    # Train directly on the full Month 1 data, without early stopping
    final_model.fit(X_train_final, y_train_final)

    print("Training complete.")

    # ----------------------------------------------------
    # 3. Save the Final Model
    # ----------------------------------------------------
    print("  4. Saving the Model  ")
    model_filename = dir + model1_name
    joblib.dump(final_model, model_filename)
    print(f"Model successfully saved to: {model_filename}")

    print("  3. Evaluating the Final Model on the Test Set  ")

    y_pred_final = final_model.predict(X_test_final)
    y_proba_final = final_model.predict_proba(X_test_final)[:, 1]

    # Evaluate the final model
    evaluate_model(y_test_final, y_pred_final, y_proba_final, model_name='Final LightGBM (100 Iterations)')

    # %% [markdown]
    # ## Feature importance

    # %%
    # Analyze Feature Importance of the Champion Model  

    print("  Analyzing Feature Importance from the Final LightGBM Model  ")

    # Our final model is stored in the 'final_model' variable
    # X_train contains all our feature column names

    try:
        # Create a DataFrame to store feature names and their importance scores
        feature_importance_df = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': final_model.feature_importances_
        })

        # Sort the DataFrame by importance in descending order
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

        # Print the top 15 most important features
        print("\nTop 15 Most Important Features:")
        print(feature_importance_df.head(15))

        # #   Plot Feature Importance  
        # plt.figure(figsize=(12, 8))
        # sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(15), palette='rocket')
        # plt.title('Top 15 Feature Importances in LightGBM Model', fontsize=16)
        # plt.xlabel('Importance Score')
        # plt.ylabel('Feature')
        # plt.grid(axis='x')
        
        # # Save the plot to a file
        # plt.savefig('feature_importance.png', bbox_inches='tight')
        # print("\nFeature importance plot saved as 'feature_importance.png'")
        
        # plt.show()

    except NameError as e:
        print(f"\nAn error occurred: {e}")
        print("Please ensure you have successfully run the previous model training code so that the 'final_model' and 'X_train' variables are available.")

    # %% [markdown]
    # # Further hyperparameter tuning: Optuna

    # %%
    # Advanced Tuning: Hyperparameter Search for LightGBM using Optuna  

    # Import all necessary libraries
    import optuna
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    import random
    import os


    #   1. Data Preparation: Create Training and Validation sets  
    # We continue to use the Month 1 data for tuning, split into 80% for training and 20% for validation.
    X_train_tune, X_val, y_train_tune, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    # The final test set remains the untouched Month 2 data
    X_test_final = X_test
    y_test_final = y_test

    objective_with_data = partial(
        objective, 
        X_train_tune=X_train_tune, 
        y_train_tune=y_train_tune,
        X_val=X_val,
        y_val=y_val
    )

    #   3. Create and Run the Optuna Study  
    # We tell Optuna our goal is to maximize the return value of the objective function
    study = optuna.create_study(direction='maximize')
    # study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
    print("Starting Optuna hyperparameter search for 50 trials this will take a while.")
    # Run 50 trials. You can increase or decrease this number as needed.
    study.optimize(objective_with_data, n_trials=1)

    #   4. Print the Best Results  
    print("\nOptuna search finished.")
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (AUC): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    #   5. Evaluate the Best Model on the Final Test Set  
    print("\n  Training final model with best params and evaluating on test set  ")
    best_params = trial.params
    # Ensure we add our fixed parameters
    best_params['objective'] = 'binary'
    best_params['metric'] = 'auc'
    best_params['verbosity'] = -1
    best_params['boosting_type'] = 'gbdt'
    best_params['scale_pos_weight'] = y_train.value_counts()[0] / y_train.value_counts()[1]
    best_params['random_state'] = 42

    # Retrain the final model on the **ENTIRE Month 1 data** using the best parameters
    final_optuna_model = lgb.LGBMClassifier(**best_params)
    final_optuna_model.fit(X_train, y_train)

    final_optuna_model_filename = dir + model2_name
    joblib.dump(final_optuna_model, final_optuna_model_filename)

    print(f"Optuna-tuned model successfully saved to: {final_optuna_model_filename}")

    # Make predictions on the final test set
    y_pred_optuna = final_optuna_model.predict(X_test)
    y_proba_optuna = final_optuna_model.predict_proba(X_test)[:, 1]

    # Evaluate the final tuned model
    evaluate_model(y_test, y_pred_optuna, y_proba_optuna, model_name='Final Tuned LightGBM (Optuna)')

    # (A) Calculate Precision and Recall
    precision = precision_score(y_test, y_pred_optuna)
    recall = recall_score(y_test, y_pred_optuna)

    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")

    # (B) Calculate AUC Score
    auc = roc_auc_score(y_test, y_proba_optuna)
    print(f"AUC Score:         {auc:.4f}\n")

    return (precision, recall, auc)

#   2. Define the "Objective Function"  
# This function is the core of Optuna. Optuna repeatedly calls this function with different parameter combinations, trying to maximize its return value (in our case, the AUC score).
def objective(trial, X_train_tune, y_train_tune, X_val, y_val):
    # Define the hyperparameter search space
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'scale_pos_weight': y_train_tune.value_counts()[0] / y_train_tune.value_counts()[1],
        'random_state': 42,
        'n_estimators': 1000, 
        'learning_rate': trial.suggest_float('learning_rate', 0.010187155669571828, 0.010187155669571828, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 45, 45),
        'max_depth': trial.suggest_int('max_depth', 12, 12),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0961137078732296, 0.0961137078732296, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 4.213965653875635e-06, 4.213965653875635e-06, log=True),
        'colsample_bytree':trial.suggest_float('colsample_bytree',  0.626931092112369,  0.626931092112369),
        'subsample':trial.suggest_float('subsample', 0.7697265059520819, 0.7697265059520819) ,
        'min_child_samples': trial.suggest_int('min_child_samples', 52, 52),
    }

      # Train the model
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train_tune, y_train_tune,
              eval_set=[(X_val, y_val)],
              eval_metric='auc',
              callbacks=[lgb.early_stopping(50, verbose=False)])

    # Evaluate on the validation set
    preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, preds)

    return auc

def loadModel(model_path):
    print(f"  A. Loading model from {model_path} ...")
    try:
        loaded_model = joblib.load(model_path)
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
    return loaded_model

def predictChurn(loaded_model, X_new):
    print("  B. Making predictions on new data ...")
    try:
        predictions = loaded_model.predict(X_new)
        prediction_probabilities = loaded_model.predict_proba(X_new)[:, 1]
        print("Predictions made successfully.")

        print("\n--- Prediction Results (first 10) ---")
        print("Predictions:", predictions[:10])
        print("Probabilities:", np.round(prediction_probabilities[:10], 2))
        return predictions, prediction_probabilities
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None
    
def predict(file_path, model_path="lgbm_churn_model.joblib", model_path_optuna="lgbm_churn_model_optuna-tuned.joblib"):
    # Load new data
    df_new = loadData(file_path)
    cols_to_drop = ['SNAPSHOT_DATE', 'CUSTOMER_NUMBER', 'PRODUCT_TYPE']

    _, Test_df_, _, _, X_new, _ = cleanData(df_new, is_prediction = True)

    print("\nNew data prepared for prediction.")
    print(f"Shape of new data features: {X_new.shape}")
    print(f"X_new: {X_new.head()}")

    # Load the trained model
    loaded_model = loadModel(model_path)

     # Load the trained model
    loaded_oputna_model = loadModel(model_path_optuna)

    print("\n--- Using Default Model ---")

    X_new_predict = X_new.drop(columns=cols_to_drop, errors='ignore')

    # Make predictions
    predictions1, prediction_probabilities1 = predictChurn(loaded_model, X_new_predict)

    # results_df_model1 = X_new
    # results_df_model1['Predicted_Churn'] = predictions1
    # results_df_model1['Churn_Probability'] = prediction_probabilities1
    # print("predictions1 - 1:", predictions1.head())
    # churning_customers1 = results_df_model1[results_df_model1['Prediction_Churn'] == 1]
    # churning_customers1 = results_df_model1[['CUSTOMER_NUMBER', 'Predicted_Churn', 'Churn_Probability']]
    # print("predictions1 shape:", churning_customers1.shape)  
    X_new['Predicted_Churn'] = predictions1
    X_new['Churn_Probability'] = prediction_probabilities1
    print("predictions1 - 1:")
    # churning_customers1 = X_new[(X_new['Predicted_Churn'] == 1.0)]
    print("predictions1 - 2:")
    # churning_customers1 = churning_customers1[['CUSTOMER_NUMBER', 'Predicted_Churn', 'Churn_Probability']]
    print("predictions1 len:", len(churning_customers1))  


    print("\n\n--- Using Optuna-Tuned Model ---")
    # Make predictions
    predictions2, prediction_probabilities2 = predictChurn(loaded_oputna_model, X_new_predict)
    # results_df_model2 = X_new
    # results_df_model2['Predicted_Churn'] = predictions1
    # results_df_model2['Churn_Probability'] = prediction_probabilities1
    # churning_customers2 = results_df_model2[results_df_model2['Prediction_Churn'] == 1]
    # churning_customers2 = results_df_model2[['CUSTOMER_NUMBER', 'Predicted_Churn', 'Churn_Probability']]
    # print("predictions2 shape:", churning_customers2.shape)
    # results_df_model2 = X_new
    X_new['Predicted_Churn_Optuna'] = predictions2
    X_new['Churn_Probability_Optuna'] = prediction_probabilities2
    # churning_customers2 = X_new[X_new['Predicted_Churn_Optuna'] == 1.0]
    # churning_customers2 = churning_customers2[['CUSTOMER_NUMBER', 'Predicted_Churn_Optuna', 'Churn_Probability_Optuna']]
    print("predictions2 len:", len(churning_customers2))

    churning_customers = X_new[['CUSTOMER_NUMBER', 'Predicted_Churn', 'Churn_Probability', 'Predicted_Churn_Optuna', 'Churn_Probability_Optuna']]
  

    return churning_customers

if __name__ == "__main__":
    # 从命令行获取参数
    file_path = sys.argv[1] if len(sys.argv) > 1 else 'customer_churn_data.csv'
    task_name = sys.argv[2] if len(sys.argv) > 2 else "DefaultTask"

    # # Load the dataset
    # df = loadData(file_path)
    # cols_to_drop = ['SNAPSHOT_DATE', 'CUSTOMER_NUMBER', 'PRODUCT_TYPE']
    # (train_df, test_df, X_train, y_train, X_test, y_test) = cleanData(df, cols_to_drop = cols_to_drop)

    # train_models(X_train, y_train, X_test, y_test)

    # final_model_training(X_train, y_train, X_test, y_test)

    df1 = predict("predict_customers.csv")




