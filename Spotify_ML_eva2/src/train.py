from pathlib import Path
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from preprocess import build_preprocessor

def build_models(preprocessor, random_state=42):
    return {
        "Regresion Lineal": Pipeline([("preprocessor", preprocessor), ("model", LinearRegression())]),
        "Random Forest": Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_leaf=2, random_state=random_state, n_jobs=-1)),
        ]),
    }

def train_and_save(X_train, y_train, numeric_features, categorical_features, output_dir, random_state=42):
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    models = build_models(preprocessor, random_state)
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        filename = "regresion_lineal.joblib" if name == "Regresion Lineal" else "random_forest.joblib"
        joblib.dump(model, output_dir / filename)
        trained[name] = model
    return trained

if __name__ == "__main__":
    print("Use 1_EDA.ipynb y 2_Modelizacion.ipynb para ejecutar el flujo completo.")
