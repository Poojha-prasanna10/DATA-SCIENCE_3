import pandas as pd
from sklearn.ensemble import RandomForestClassifier  # Example model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load the cleaned data
cleaned_data_path = 'data/processed/cleaned_data.csv'
df = pd.read_csv(cleaned_data_path)

# Assuming 'target' is the column we want to predict
X = df.drop('target', axis=1)
y = df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestClassifier()

# Train the model
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'model.pkl')
print("Model trained and saved as 'model.pkl'.")

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy}")
