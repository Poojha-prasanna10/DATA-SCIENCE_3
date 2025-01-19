import pandas as pd
from sklearn.model_selection import train_test_split

# Load the raw data
raw_data_path = 'data/raw/raw_data.csv'
cleaned_data_path = 'data/processed/cleaned_data.csv'
train_data_path = 'data/processed/train.csv'
test_data_path = 'data/processed/test.csv'

try:
    # Read the raw dataset
    df = pd.read_csv(raw_data_path)
    print("Raw data loaded successfully.")

    # Step 1: Handle missing values (Example: Drop rows with missing values)
    df_cleaned = df.dropna()  # Drop rows with missing values (customize as needed)
    print(f"Data after handling missing values: {df_cleaned.shape[0]} rows, {df_cleaned.shape[1]} columns")

    # Step 2: Feature Engineering (Example: Convert categorical columns to numerical, if applicable)
    if 'Category' in df_cleaned.columns:
        df_cleaned['Category'] = pd.factorize(df_cleaned['Category'])[0]

    # Step 3: Split the data into training and testing sets (80% for training, 20% for testing)
    X = df_cleaned.drop(columns=['target'])  # Features
    y = df_cleaned['target']  # Target variable
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Combine features and target for training data
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = X_test  # No target for test data
    
    # Save the train and test data
    train_data.to_csv(train_data_path, index=False)
    test_data.to_csv(test_data_path, index=False)
    print(f"Train data saved as {train_data_path}")
    print(f"Test data saved as {test_data_path}")

    # Step 4: Save the cleaned data (optional)
    df_cleaned.to_csv(cleaned_data_path, index=False)
    print(f"Cleaned data saved as {cleaned_data_path}")

except FileNotFoundError:
    print(f"Error: The file '{raw_data_path}' was not found.")
except Exception as e:
    print(f"An error occurred during preprocessing: {e}")
