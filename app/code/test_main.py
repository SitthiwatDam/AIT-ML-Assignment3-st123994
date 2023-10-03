import numpy as np
from main import price_prediction, get_features,Prediction

def test_expected_input():
    MAXPOWER, YEAR, selected_FUEL, selected_brand = 200, 2020, "Petrol", "Toyota"
    result = get_features(MAXPOWER, YEAR, selected_FUEL, selected_brand)
    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 35)
    assert result.dtype == np.float64

def test_expected_output_shape():
    MAXPOWER, YEAR, selected_FUEL, selected_brand = 200, 2020, "Petrol", "Toyota"
    result = price_prediction(MAXPOWER, YEAR, selected_FUEL, selected_brand)
    assert isinstance(result, np.ndarray)
    assert result.shape == (1,)

def test__output_final_result():
    MAXPOWER, YEAR, selected_FUEL, selected_brand, SUBMIT = 200, 2020, "Petrol", "Toyota", 1
    price_mapping= {0: 'Low price',1: 'Good price',2: 'Great price',3: 'Expensive price'}
    result = Prediction(MAXPOWER, YEAR, selected_FUEL, selected_brand, SUBMIT)
    assert isinstance(result, str)
    assert result in price_mapping.values()
