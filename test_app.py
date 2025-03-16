import os
import json
import pytest
import numpy as np
import pandas as pd
from flask import Flask
from unittest.mock import patch, MagicMock

# Import the functions/classes from your main code.
# If your main code file is named "app.py", do:
from app import app, load_and_preprocess_data, train_model

@pytest.fixture
def client():
    """
    Pytest fixture to create a test client for the Flask app.
    """
    with app.test_client() as client:
        yield client

@pytest.fixture
def sample_data_file(tmp_path):
    """
    Creates a temporary CSV file with sample data for testing 
    load_and_preprocess_data().
    """
    df = pd.DataFrame({
        'Delivery_Time_min': [30, 45, 60],
        'Distance_km': [2.5, 4.0, 5.2]
    })
    file_path = tmp_path / "Food_Delivery_Times.csv"
    df.to_csv(file_path, index=False)
    return file_path

def test_load_and_preprocess_data(sample_data_file):
    """
    Test whether load_and_preprocess_data successfully reads 
    and processes the CSV, returning the correct shapes.
    """
    # Patch pandas.read_csv to return our sample data file
    with patch("pandas.read_csv", return_value=pd.read_csv(sample_data_file)):
        X, y = load_and_preprocess_data()
    
    assert X.shape == (3, 1)  # 3 rows, 1 column
    assert len(y) == 3
    # Check if the values align
    np.testing.assert_array_equal(X[:, 0], [2.5, 4.0, 5.2])
    np.testing.assert_array_equal(y, [30, 45, 60])

def test_train_model(tmp_path):
    """
    Test if train_model creates 'linear_model.pkl' and trains without error.
    """
    # Create sample data for training
    X = np.array([[2], [3], [4], [5]])
    y = np.array([20, 30, 40, 50])
    model_file = tmp_path / "linear_model.pkl"

    # Change working directory to tmp_path so 'linear_model.pkl' writes there
    current_dir = os.getcwd()
    os.chdir(tmp_path)
    try:
        train_model(X, y)
        # Check if file was created
        assert os.path.exists(model_file)
    finally:
        os.chdir(current_dir)

def test_predict_valid_distance(client):
    """
    Test the /predict endpoint with a valid distance value.
    We mock 'model.predict' so we don't rely on an actual model file.
    """
    with patch("app.model.predict", return_value=[42]):  
        payload = {"distance": 3.5}
        response = client.post(
            "/predict",
            data=json.dumps(payload),
            content_type="application/json"
        )
        assert response.status_code == 200
        data = response.get_json()
        assert "predicted_delivery_time" in data
        assert data["predicted_delivery_time"] == 42

def test_predict_missing_distance(client):
    """
    Test the /predict endpoint when 'distance' is missing from the payload.
    """
    payload = {"foo": "bar"}  # distance is missing
    response = client.post(
        "/predict",
        data=json.dumps(payload),
        content_type="application/json"
    )
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data
    assert "Missing 'distance'" in data["error"]
