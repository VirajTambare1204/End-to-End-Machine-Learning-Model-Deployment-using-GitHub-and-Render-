from flask import Flask, request, jsonify
import pickle
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/', methods=['GET'])
def home():
    return "Heart Disease Prediction API is live! Send a POST request to /predict."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Accept patient details as JSON input
        data = request.get_json()
        
        # Convert JSON to DataFrame (ensures feature names match the trained model)
        input_data = pd.DataFrame([data])
        
        # Predict using the loaded model
        prediction_array = model.predict(input_data)
        prediction_value = int(prediction_array[0])
        
        # Format response
        if prediction_value == 1:
            result = "Heart Disease Detected"
        else:
            result = "No Heart Disease Detected"
            
        # Return the prediction as JSON
        return jsonify({"prediction": result})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)