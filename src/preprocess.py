import pandas as pd

raw_data_path = 'data/raw/raw_data.csv'
cleaned_data_path = 'data/processed/cleaned_data.csv'

try:
    df = pd.read_csv(raw_data_path)
    print("Raw data loaded successfully.")

    df_cleaned = df.dropna()
    print(f"Data after handling missing values: {df_cleaned.shape[0]} rows, {df_cleaned.shape[1]} columns")

    if 'Category' in df_cleaned.columns:
        df_cleaned['Category'] = pd.factorize(df_cleaned['Category'])[0]

    df_cleaned.to_csv(cleaned_data_path, index=False)
    print(f"Cleaned data saved as {cleaned_data_path}")

except FileNotFoundError:
    print(f"Error: The file '{raw_data_path}' was not found.")
except Exception as e:
    print(f"An error occurred during preprocessing: {e}")

