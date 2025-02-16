# serialization_test.py
import os
import pickle
import logging
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
import numpy as np

logging.basicConfig(level=logging.INFO)

def test_serialization():
    # Create a simple model
    X = np.random.rand(100, 4)
    y = np.random.rand(100)
    
    # Test individual model
    rf = RandomForestRegressor(n_estimators=10)
    rf.fit(X, y)
    
    # Test stacking ensemble
    estimators = [
        ('rf', RandomForestRegressor(n_estimators=10)),
        ('lr', LinearRegression())
    ]
    stack = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())
    stack.fit(X, y)
    
    # Test saving individual model
    rf_path = "artifacts/test_rf.pkl"
    stack_path = "artifacts/test_stack.pkl"
    
    os.makedirs("artifacts", exist_ok=True)
    
    # Test pickle
    logging.info("\nTesting with pickle")
    try:
        with open(rf_path, 'wb') as f:
            pickle.dump(rf, f)
        size = os.path.getsize(rf_path)
        logging.info(f"RandomForest saved successfully with pickle. Size: {size} bytes")
    except Exception as e:
        logging.error(f"Pickle error with RandomForest: {str(e)}")
    
    # Test joblib
    logging.info("\nTesting with joblib")
    try:
        import joblib
        joblib.dump(stack, stack_path)
        size = os.path.getsize(stack_path)
        logging.info(f"StackingRegressor saved successfully with joblib. Size: {size} bytes")
    except Exception as e:
        logging.error(f"Joblib error with StackingRegressor: {str(e)}")
    
    # Verify files
    logging.info("\nVerifying saved files:")
    for path in [rf_path, stack_path]:
        if os.path.exists(path):
            size = os.path.getsize(path)
            logging.info(f"{path}: {size} bytes")
        else:
            logging.error(f"{path} does not exist")

if __name__ == "__main__":
    test_serialization()