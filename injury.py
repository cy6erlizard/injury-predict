from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load models and encoders
with open('logistic_model.pkl', 'rb') as file:
    logistic_model = pickle.load(file)

with open('random_forest_model.pkl', 'rb') as file:
    random_forest_model = pickle.load(file)

with open('svm_model.pkl', 'rb') as file:
    svm_model = pickle.load(file)

with open('label_encoder_poste.pkl', 'rb') as file:
    label_encoder_poste = pickle.load(file)

with open('label_encoder_blessure.pkl', 'rb') as file:
    label_encoder_blessure = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Define endpoint for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    data = request.get_json()

    # Preprocess the input data
    input_data = pd.DataFrame(data)
    input_data['poste_encoded'] = label_encoder_poste.transform(input_data['poste'].str.strip())
    input_data_scaled = scaler.transform(input_data[['poste_encoded', 'age']])

    # Make predictions using the models
    logistic_prediction = logistic_model.predict(input_data_scaled)
    random_forest_prediction = random_forest_model.predict(input_data_scaled)
    svm_prediction = svm_model.predict(input_data_scaled)

    # Convert encoded predictions back to original labels
    logistic_prediction_label = label_encoder_blessure.inverse_transform(logistic_prediction)
    random_forest_prediction_label = label_encoder_blessure.inverse_transform(random_forest_prediction)
    svm_prediction_label = label_encoder_blessure.inverse_transform(svm_prediction)

    # Create a JSON response
    response = {
        'logistic_prediction': logistic_prediction_label.tolist(),
        'random_forest_prediction': random_forest_prediction_label.tolist(),
        'svm_prediction': svm_prediction_label.tolist()
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
