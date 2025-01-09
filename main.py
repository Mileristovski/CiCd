import numpy as np
import pickle
import os
from flask import Flask, request, jsonify
import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def load_and_preprocess_data():
    df = pd.read_csv('Food_Delivery_Times.csv')
    
    print(df.head())
    
    print(df.isnull().sum())

    data = df[['Delivery_Time_min', 'Distance_km']].dropna()
    
    X = data[['Distance_km']].values  
    y = data['Delivery_Time_min'].values  
    
    return X, y

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    
    with open('linear_model.pkl', 'wb') as f:
        pickle.dump(model, f)

# Check if the model file exists
if os.path.exists('linear_model.pkl'):
    with open('linear_model.pkl', 'rb') as f:
        model = pickle.load(f)
else:
    X, y = load_and_preprocess_data()
    train_model(X, y)
    with open('linear_model.pkl', 'rb') as f:
        model = pickle.load(f)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if 'distance' not in data:
            return jsonify({'error': "Missing 'distance' in request body"}), 400
        
        distance = np.array([[data['distance']]])
        
        prediction = model.predict(distance)
        
        return jsonify({'predicted_delivery_time': prediction[0]})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
