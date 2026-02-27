from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_linear_regression(X_train, y_train):
    model1 = LinearRegression()
    model1.fit(X_train, y_train)
    return model1    

def train_decision_tree(X_train, y_train):
    model2 = DecisionTreeRegressor(max_depth=4, random_state=42)
    model2.fit(X_train, y_train)
    return model2

def train_random_forest(X_train, y_train):
    model3 = RandomForestRegressor(max_depth=5, random_state=42)
    model3.fit(X_train, y_train)
    return model3


def evaluate_model(model, X_test, y_test, model_name):
    preds = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds) * 100
    
    print(f"\n{model_name} Performance: ")
    print(f"MAE : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R2 : {r2:.2f}%")
    
    return {
        "model_name":model_name,
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }