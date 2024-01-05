import pandas as pd
import joblib  # This library is used to load the model from a .pkl file

# Load the saved model
loaded_model = joblib.load('best_model.pkl')

# Define sample data for prediction (similar to the format used during training)
sample_data = pd.DataFrame({
    'Machine_ID': [2],  # Replace with appropriate values for your dataset
    'Sensor_ID': [2],   # Replace with appropriate values for your dataset
    'Hour': [4]         # Replace with appropriate values for your dataset
})

# Predict using the loaded model
prediction = loaded_model.predict(sample_data)

# Print the parameters used by the model
model_params = loaded_model.get_params()
print("Model Parameters:")
print(model_params)

# Print the prediction
print("\nPrediction:")
print(prediction)
