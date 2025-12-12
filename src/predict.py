import joblib
import pandas as pd

def make_prediction(model_path: str, data: dict):
    """Make a prediction using the trained model."""
    model = joblib.load(model_path)
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return prediction[0]
