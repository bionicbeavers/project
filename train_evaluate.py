from flask import Flask, render_template
import mlflow
import pandas as pd

# app = Flask(__name__)

# @app.route('/')
# def index():
    # Connect to the MLflow server
mlflow.set_tracking_uri(uri="http://127.0.0.1:8081")

mlflow.set_experiment("MLflow Machine Sensor")

# Get the best run with the lowest MSE value
best_run = mlflow.search_runs(order_by=["metrics.msevalue DESC"]).iloc[0]

print(best_run )


# Specify the run ID or the artifact URI of the model you want to load
run_id = best_run['run_id']  # Replace with the actual run ID or artifact URI
artifact_url = best_run['artifact_uri']  # Replace with the actual run ID or artifact URI

val = (f"runs:/{run_id}/{artifact_url}")
print(val)
# Load the model
loaded_model = mlflow.sklearn.load_model(val)

# Now, you can use the loaded_model for predictions

print(loaded_model)
# Get the best model information
# model_info = mlflow.sklearn.load_model(best_run.artifact_uri + "/machine_sensor_model")

# # Get the MSE value
# mse_value = best_run.metrics.msevalue

#     # Render the template with the model and MSE information
#     return render_template('index.html', model_info=model_info, mse_value=mse_value)

# if __name__ == '__main__':
#     app.run(debug=True)
