from pathlib import Path
import joblib
import pandas as pd
from metrica import regression_metrics

def evaluate_saved_model(model_path, X_test, y_test):
    model = joblib.load(model_path)
    pred = model.predict(X_test)
    return regression_metrics(y_test, pred)

if __name__ == "__main__":
    print("Este modulo contiene la funcion para evaluar modelos guardados.")
