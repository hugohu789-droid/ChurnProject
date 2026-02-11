import pandas as pd
import numpy as np
import joblib
import os
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_score, recall_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants for default paths
DEFAULT_MODEL_PATH = "churn_model_rf.joblib"
DEFAULT_OUTPUT_PATH = "prediction_results.csv"

def load_data(file_path):
    """
    Load data from a CSV file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    df = pd.read_csv(file_path)
    logger.info(f"Data loaded successfully from {file_path}. Shape: {df.shape}")
    return df

def clean_data(df, is_training=True):
    """
    Clean and preprocess the Telco Customer Churn dataset.
    
    Args:
        df (pd.DataFrame): Raw dataframe.
        is_training (bool): If True, processes the target 'Churn' column.
        
    Returns:
        pd.DataFrame: Cleaned dataframe ready for encoding.
    """
    df_clean = df.copy()

    # 1. Handle TotalCharges: Convert to numeric, coerce errors (empty strings) to NaN
    # The dataset contains " " which causes issues.
    if 'TotalCharges' in df_clean.columns:
        df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
        # Fill NaN values (created by empty strings) with 0 or median
        df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(0)

    # 2. Drop customerID as it's not a feature
    if 'customerID' in df_clean.columns:
        df_clean = df_clean.drop('customerID', axis=1)

    # 3. Process Target Variable 'Churn' if in training mode
    if is_training and 'Churn' in df_clean.columns:
        df_clean['Churn'] = df_clean['Churn'].map({'Yes': 1, 'No': 0})
    
    return df_clean

def perform_eda(df):
    """
    Perform basic Exploratory Data Analysis.
    """
    logger.info("--- EDA Summary ---")
    logger.info(f"Missing Values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    
    if 'Churn' in df.columns:
        churn_rate = df['Churn'].value_counts(normalize=True)
        logger.info(f"Churn Distribution:\n{churn_rate}")
    
    desc = df.describe()
    logger.info(f"Statistical Summary:\n{desc}")
    return desc

def train_model(file_path, model_save_path=DEFAULT_MODEL_PATH):
    """
    Train a Random Forest model to predict Churn.
    
    1. Loads and cleans data.
    2. Performs One-Hot Encoding.
    3. Trains RandomForestClassifier.
    4. Evaluates metrics.
    5. Saves the model and feature names to disk.
    """
    logger.info(f"Starting training pipeline with file: {file_path}")
    
    # 1. Load
    df = load_data(file_path)
    
    # 2. Clean
    df_clean = clean_data(df, is_training=True)
    
    # 3. EDA
    perform_eda(df_clean)
    
    # 4. Prepare Features (X) and Target (y)
    X = df_clean.drop('Churn', axis=1)
    y = df_clean['Churn']
    
    # 5. Encoding Categorical Variables
    # Using get_dummies is robust for this dataset. 
    # We must save the column names to ensure alignment during prediction.
    X_encoded = pd.get_dummies(X, drop_first=True)
    feature_names = X_encoded.columns.tolist()
    
    # 6. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 7. Train Model
    # RandomForest is chosen for its robustness and ability to handle non-linear data without scaling.
    # class_weight='balanced' helps with the class imbalance in Churn datasets.
    clf = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        class_weight='balanced',
        max_depth=10
    )
    clf.fit(X_train, y_train)
    
    # 8. Evaluate
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba))
    }
    
    logger.info("--- Model Evaluation ---")
    logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"ROC AUC:   {metrics['roc_auc']:.4f}")
    logger.info(f"Recall:    {metrics['recall']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    
    # 9. Save Model and Feature Metadata
    # We save a dictionary containing the model and the list of features it expects.
    model_data = {
        'model': clf,
        'features': feature_names
    }
    joblib.dump(model_data, model_save_path)
    logger.info(f"Model and feature metadata saved to: {model_save_path}")
    
    return metrics

def predict(model_path, data_path, output_path=DEFAULT_OUTPUT_PATH):
    """
    Load a trained model and predict churn on a new dataset.
    
    Args:
        model_path (str): Path to the .joblib model file.
        data_path (str): Path to the CSV file containing new data.
        output_path (str): Path where the results CSV will be saved.
        
    Returns:
        str: The absolute path to the saved results file.
    """
    logger.info(f"Starting prediction pipeline using model: {model_path}")
    
    # 1. Load Model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    loaded_data = joblib.load(model_path)
    model = loaded_data['model']
    expected_features = loaded_data['features']
    
    # 2. Load New Data
    df_new = load_data(data_path)
    
    # 3. Clean Data (Same steps as training, but is_training=False)
    df_clean = clean_data(df_new, is_training=False)
    
    # 4. Encode
    # We use get_dummies again
    df_encoded = pd.get_dummies(df_clean, drop_first=True)
    
    # 5. Align Features
    # Ensure the new data has exactly the same columns as the training data
    # Add missing columns with 0
    for col in expected_features:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
            
    # Reorder columns to match training order and drop any extra columns not seen in training
    df_final = df_encoded[expected_features]
    
    # 6. Predict
    predictions = model.predict(df_final)
    probabilities = model.predict_proba(df_final)[:, 1]
    
    # 7. Save Results
    # We append predictions to the original dataframe for context
    results_df = df_new.copy()
    results_df['Predicted_Churn'] = predictions
    results_df['Churn_Probability'] = probabilities
    
    # Map 1/0 back to Yes/No for readability if desired, but keeping numeric is usually better for systems
    # results_df['Predicted_Churn_Label'] = results_df['Predicted_Churn'].map({1: 'Yes', 0: 'No'})
    
    results_df.to_csv(output_path, index=False)
    
    abs_path = os.path.abspath(output_path)
    logger.info(f"Predictions saved successfully to: {abs_path}")
    
    return abs_path

if __name__ == "__main__":
    # Example usage for testing
    # You can run this file directly to test: python modeltrain.py
    
    # Assuming the dataset is in the same directory or you provide a path
    # For demonstration, we'll try to find the file in the context path provided
    dataset_path = "dataset/Telco_Cusomer_Churn.csv" 
    
    # Check if file exists in current dir or specific path
    if not os.path.exists(dataset_path):
        # Fallback for the user's specific environment path if running locally there
        dataset_path = r"c:\\Users\\Hugo\Documents\\ChurnProject\\backend\\app\dataset\\Telco_Cusomer_Churn.csv"

    if os.path.exists(dataset_path):
        # 1. Train
        metrics = train_model(dataset_path)
        
        # 2. Predict (using the same file as a dummy test for prediction)
        predict(DEFAULT_MODEL_PATH, dataset_path)
    else:
        logger.error("Dataset not found. Please check the path.")
