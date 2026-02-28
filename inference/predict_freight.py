import joblib
import pandas as pd

MODEL_PATH = '/Volumes/Sachin/ML-Projects/new-project/freight_cost_prediction/models/predict_freight_model.pkl'

def load_model(model_path: str = MODEL_PATH):
    """
    Load trained freight cost prediction model
    """
    with open(model_path, 'rb') as f:
        model = joblib.load(f)
    return model

def predict_freight(input_data):
    """
    Predict freight cost for new vendor invoices

    Parameters:
    -----------
    input_data: dict
        Dictionary containing 'Dollars' key with list of invoice dollar amounts
        Example: {"Dollars": [214.26, 1500.50, 15527.25]}

    Returns:
    --------
    pd.DataFrame
        DataFrame with input 'Dollars' and 'Predicted_Freight' columns
    """
    model = load_model()
    input_df = pd.DataFrame(input_data)
    input_df['Predicted_Freight'] = model.predict(input_df).round()
    return input_df


if __name__ == "__main__":

    # Example usage with sample data
    # Model is trained to predict Freight based on Dollars
    sample_data = {
        "Dollars": [214.26, 1500.50, 15527.25, 137483.78, 5000.00]
    }

    prediction = predict_freight(sample_data)
    print("\nFreight Cost Predictions:")
    print("=" * 50)
    print(prediction[['Dollars', 'Predicted_Freight']])