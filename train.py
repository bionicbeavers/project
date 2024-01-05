import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

# Read the CSV file into a DataFrame
file_path = 'dummy_sensor_data.csv'
data = pd.read_csv(file_path)

# Encode categorical columns (Machine_ID and Sensor_ID)
label_encoders = {}
for col in ['Machine_ID', 'Sensor_ID']:
    label_encoders[col] = LabelEncoder()
    data[col] = label_encoders[col].fit_transform(data[col])

# Convert 'Timestamp' column to datetime and extract hour
data['Timestamp'] = pd.to_datetime(data['Timestamp'])
data['Hour'] = data['Timestamp'].dt.hour  # Extract hour from timestamp
data.set_index('Timestamp', inplace=True)

# Create features and target variable
X = data[['Machine_ID', 'Sensor_ID', 'Hour']]
y = data['Reading']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Print shapes of training and validation sets
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

# Create RandomForestRegressor model
model = RandomForestRegressor(random_state=42)

# Define hyperparameters to tune
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=2)
grid_search.fit(X_train, y_train)

# Get the best model from the grid search
best_model = grid_search.best_estimator_

# Train the best model on the entire training set
best_model.fit(X_train, y_train)

# Predict on the validation set
y_pred = best_model.predict(X_val)

# Evaluate model performance on validation set
mse = mean_squared_error(y_val, y_pred)
print(f"Mean Squared Error on Validation Set: {mse}")

# Get the best hyperparameters found by GridSearchCV
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)