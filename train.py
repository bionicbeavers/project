import mlflow
from mlflow.models import infer_signature
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import pickle
import time
import os
# Read the CSV file into a DataFrame
file_path = './data/dummy_sensor_data.csv'
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

# Set our tracking server uri for logging
mlflow.set_tracking_uri(uri="https://dagshub.com/bionicbeavers/project.mlflow")

mlflow.set_registry_uri(uri="https://dagshub.com/bionicbeavers/project.mlflow")
# Create a new MLflow Experiment
mlflow.set_experiment("MLflow Machine Sensor")

# Start an MLflow run
with mlflow.start_run():
    # Log the hyperparameters
    mlflow.log_params(best_params)

    # Log the loss metric
    mlflow.log_metric("msevalue", mse)

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", "dummy_data")

    # Infer the model signature
    signature = infer_signature(X_train, best_model.predict(X_train))

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="machine_sensor_model",
        signature=signature,
        input_example=X_train,
        registered_model_name="tracking-machine-sensing",
    )


# Retrieve the run ID of the run with the lowest MSE value
best_run = mlflow.search_runs(order_by=["metrics.msevalue ASC"]).iloc[0]
best_run_id = best_run.run_id

print("Best Run ID:", best_run_id)

# Retry mechanism
max_retries = 5
retry_delay = 10  # seconds
retry_count = 0

while retry_count < max_retries:
    try:
        # Attempt to load the model associated with the best run
        loaded_model = mlflow.sklearn.load_model("runs:/{}/machine_sensor_model".format(best_run_id))
        print("Model loaded successfully.")
        print(loaded_model)
        break  # Break out of the loop if successful
    except Exception as e:
        print(f"Error loading the model (retry {retry_count + 1}): {e}")
        retry_count += 1
        time.sleep(retry_delay)

# Save the loaded model as a pickle file
if loaded_model:
    try:
     os.remove("best_model.pkl")
    except Exception as e:
        print(e)
    with open('best_model.pkl', 'wb') as file:
        pickle.dump(loaded_model, file)
   
    registered_model_name = "my-best-model"  # Replace with your desired model name

    model_version = mlflow.register_model(
        model_uri= "runs:/{}/machine_sensor_model".format(best_run_id),
        name=registered_model_name,
        # description='Best model for production',
    )

    print("Best model saved as 'best_model.pkl'")
else:
    print("Failed to load the model after maximum retries.")


bestMode = mlflow.sklearn.load_model("models:/my-best-model/latest")
print(bestMode)
    
