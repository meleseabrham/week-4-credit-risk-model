import argparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from .data_processing import load_data, preprocess_data

def train_model(data_path: str, model_path: str):
    """Train the model and save it."""
    df = load_data(data_path)
    df = preprocess_data(df)
    
    X = df.drop('target', axis=1) # Assuming 'target' column
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    # Add argument parsing here
    pass
