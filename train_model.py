import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("Loading dataset...")
# Load the dataset using Pandas
df = pd.read_csv('heart.csv')

# Display the first five records
print("\n--- First Five Records ---")
print(df.head())

# Identify Numerical features and Target variable
target_variable = 'target'
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_features.remove(target_variable)

print(f"\nNumerical Features: {numerical_features}")
print(f"Target Variable: {target_variable}")

# Check for missing values
missing_values = df.isnull().sum().sum()
print(f"\nTotal Missing Values: {missing_values}")

# Prepare features and target
X = df.drop(columns=[target_variable])
y = df[target_variable]

# Split the dataset into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print("\nDataset split into 80% training and 20% testing sets.")

# Build a classification model (Random Forest)
print("\nTraining Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model using Accuracy Score
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy Score: {accuracy:.4f}")

# Save the trained model using Pickle
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
print("Model saved successfully as 'model.pkl'.")