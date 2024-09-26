from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Initialize the Flask app
app = Flask(__name__)

# Load the preprocessor and trained model
preprocessor = joblib.load('preprocessor.pkl')  # Load the preprocessor
model = joblib.load('trained_model.pkl')        # Load the trained model

# Define the numeric and categorical features
numeric_features = ['age', 'fnlwgt', 'education_num', 'hours_per_week']
categorical_features = ['workclass', 'education', 'occupation', 'race', 'sex', 'native_country']

# Define the route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request
        data = request.json
        
        # Convert the input data to a DataFrame
        input_data = pd.DataFrame([data])  # Wrap in a list for a single input

        # Ensure the input has all required columns
        input_data = input_data[numeric_features + categorical_features]
        
        # Preprocess the input data using the preprocessor
        preprocessed_data = preprocessor.transform(input_data)
        
        # Make prediction using the preprocessed data
        prediction = model.predict(preprocessed_data)
        
        # Convert the prediction result back to a readable format
        result = '>50K' if prediction[0] == 1 else '<=50K'
        
        # Return the prediction result
        return jsonify({'prediction': result})
    
    except Exception as e:
        # Return the error message if something goes wrong
        return jsonify({'error': str(e)})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
