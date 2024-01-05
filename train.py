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

