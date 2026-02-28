import joblib
import pandas as pd

MODEL_PATH = '/Volumes/Sachin/ML-Projects/new-project/invoice_flagging/models/predict_flag_invoice.pkl'
SCALER_PATH = '/Volumes/Sachin/ML-Projects/new-project/invoice_flagging/models/scaler.pkl'

def load_model(model_path: str = MODEL_PATH):
    """
    Load trained invoice flagging prediction model
    """
    with open(model_path, 'rb') as f:
        model = joblib.load(f)
    return model

def load_scaler(scaler_path: str = SCALER_PATH):
    """
    Load trained StandardScaler for feature scaling
    """
    with open(scaler_path, 'rb') as f:
        scaler = joblib.load(f)
    return scaler

def predict_invoice_flag(input_data):
    """
    Predict invoice flagging for vendor invoices

    Parameters:
    -----------
    input_data: dict
        Dictionary containing these required features:
        - 'invoice_quantity': Invoice quantity
        - 'invoice_dollars': Invoice dollar amount
        - 'Freight': Freight cost
        - 'total_item_quantity': Total item quantity from purchase orders
        - 'total_item_dollars': Total item dollars from purchase orders

    Returns:
    --------
    pd.DataFrame
        DataFrame with input features and 'Predicted_Flag' column
        Flag 0 = Normal invoice, Flag 1 = Suspicious invoice
    """
    model = load_model()
    scaler = load_scaler()
    input_df = pd.DataFrame(input_data)

    # Scale features before prediction
    input_scaled = scaler.transform(input_df)
    input_df['Predicted_Flag'] = model.predict(input_scaled)

    return input_df


if __name__ == "__main__":

    # Example usage with sample data
    # Model expects 5 features to predict invoice flagging
    sample_data = {
        "invoice_quantity": [6, 15, 5, 10100, 1935],
        "invoice_dollars": [214.26, 140.55, 106.60, 137483.78, 15527.25],
        "Freight": [3.47, 8.57, 4.61, 2935.20, 429.20],
        "total_item_quantity": [6, 15, 5, 10100, 1935],
        "total_item_dollars": [214.26, 140.55, 106.60, 137483.78, 15527.25]
    }

    prediction = predict_invoice_flag(sample_data)
    print("\nInvoice Flagging Predictions:")
    print("=" * 80)
    print(prediction)
    print("\nFlag 0 = Normal Invoice, Flag 1 = Suspicious Invoice")