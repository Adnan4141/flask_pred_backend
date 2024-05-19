from flask import Flask, request, jsonify
import pickle
import pandas as pd
from flask_cors import CORS
 

app = Flask(__name__)
CORS(app)

# Load the model
with open('prediction_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return "Welcome to the Logistic Regression Model API"

@app.route('/test', methods=['GET'])
def test():
     data = {'message': 'Welcome to Flask API'}
     return jsonify(data)

@app.route('/predict', methods=['POST'])
def predict():
  
    data = request.get_json()
    print(data)
    
    input_data = pd.DataFrame([data])
   
    
    prediction = model.predict(input_data)
    print(prediction[0])
    if prediction[0] == 0:
        message =  'The Person does not have a Heart Disease'
    else:
        message =  'The Person has Heart Disease'
   

    return jsonify(message)
if __name__ == '__main__':
    app.run(debug=True)
