import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # You can change the model based on your problem
from sklearn.metrics import accuracy_score
import joblib

# Load the cleaned data
cleaned_data_path = 'data/processed/cleaned_data.csv'

try:
    df = pd.read_csv(cleaned_data_path)
    print("Cleaned data loaded successfully.")

    # Assuming the target variable is 'target' and all other columns are features
    # Adjust the column name 'target' as per your dataset
    X = df.drop(columns=['target'])
    y = df['target']

    # Split the data into training and testing sets (80% for training, 20% for testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the model (using RandomForest here as an example)
    model = RandomForestClassifier(random_state=42)

    # Train the model
    model.fit(X_train, y_train)
    print("Model trained successfully.")

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy * 100:.2f}%")

    # Save the trained model
    model_path = 'model.pkl'
    joblib.dump(model, model_path)
    print(f"Trained model saved to {model_path}")

except FileNotFoundError:
    print(f"Error: The file '{cleaned_data_path}' was not found.")
except Exception as e:
    print(f"An error occurred during training: {e}")
