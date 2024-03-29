from flask import Flask, render_template, request
import pandas as pd
import joblib
import mlflow

app = Flask(__name__)

mlflow.set_tracking_uri(uri="https://dagshub.com/bionicbeavers/project.mlflow")

mlflow.set_registry_uri(uri="https://dagshub.com/bionicbeavers/project.mlflow")

mlflow.set_experiment("MLflow Machine Sensor")

loaded_model = mlflow.sklearn.load_model("models:/my-best-model/latest")
print(loaded_model)

# Load the saved model
# loaded_model = joblib.load('best_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get input values from form
        machine_id = int(request.form['machine_id'])
        sensor_id = int(request.form['sensor_id'])
        hour = int(request.form['hour'])

        # Create DataFrame with input values
        input_data = pd.DataFrame({
            'Machine_ID': [machine_id],
            'Sensor_ID': [sensor_id],
            'Hour': [hour]
        })

        # Make prediction using the loaded model
        prediction = loaded_model.predict(input_data)

        # Display prediction on the result page
        return render_template('result.html', prediction=prediction[0])

    # If GET request or initial render, display the input form
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)


