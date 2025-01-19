import pandas as pd
import joblib

# Load the trained model
model_path = 'model.pkl'
cleaned_data_path = 'data/processed/cleaned_data.csv'

try:
    # Load the trained model using joblib
    model = joblib.load(model_path)
    print("Model loaded successfully.")

    # Load the cleaned data (assuming the target column is 'target' and features are the rest)
    df_cleaned = pd.read_csv(cleaned_data_path)
    
    # Assuming the target column is 'target' (replace it with your actual target column name)
    X = df_cleaned.drop(columns=['target'])  # Features
    y = df_cleaned['target']  # Target
    
    # Make predictions using the model
    predictions = model.predict(X)
    print(f"Predictions: {predictions[:5]}")  # Show the first 5 predictions

    # Optionally, save the predictions to a CSV file
    predictions_df = pd.DataFrame(predictions, columns=['Predicted_Target'])
    predictions_df.to_csv('predictions.csv', index=False)
    print("Predictions saved to 'predictions.csv'")

except FileNotFoundError:
    print(f"Error: The file '{model_path}' or '{cleaned_data_path}' was not found.")
except Exception as e:
    print(f"An error occurred during prediction: {e}")
