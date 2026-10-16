from pathlib import Path
import joblib
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from preprocess import build_preprocessor

# Configuracion final del Random Forest.
# - max_depth=None: cada arbol puede crecer hasta que las hojas tengan >= 10 canciones.
# - max_features=0.33: en cada division solo mira un tercio de las variables; asi las 113 columnas
#   de genero (one-hot) no acaparan todas las divisiones y el audio tambien participa.
# - min_samples_leaf=10: limita el sobreajuste y mantiene el archivo .joblib en un tamano razonable.
RF_PARAMS = dict(n_estimators=100, max_depth=None, min_samples_leaf=10, max_features=0.33)

# Configuracion inicial (max_depth=15). Se conserva SOLO para el diagnostico del notebook 2:
# con ella los arboles gastan sus 15 niveles separando generos y ~40-80% de las canciones recibe
# una prediccion casi identica (la "linea horizontal" del grafico real vs. predicho).
RF_PARAMS_INICIAL = dict(n_estimators=100, max_depth=15, min_samples_leaf=2)

MODEL_FILES = {"Regresion Lineal": "regresion_lineal", "Random Forest": "random_forest"}


def build_models(preprocessor, random_state=42, rf_params=None):
    """Crea los dos modelos supervisados.

    Cada pipeline recibe su propia copia (`clone`) del preprocesador: asi, entrenar el mismo
    preprocesador con otro conjunto de datos (por ejemplo, sin ceros) no altera modelos ya entrenados.
    """
    rf_params = RF_PARAMS if rf_params is None else rf_params
    return {
        "Regresion Lineal": Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("model", LinearRegression()),
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("model", RandomForestRegressor(**rf_params, random_state=random_state, n_jobs=-1)),
        ]),
    }


def save_models(trained, output_dir, suffix):
    """Guarda modelos ya entrenados como `<modelo>_<suffix>.joblib` (comprimidos). Devuelve las rutas."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, model in trained.items():
        paths[name] = output_dir / f"{MODEL_FILES[name]}_{suffix}.joblib"
        joblib.dump(model, paths[name], compress=3)
    return paths


def train_and_save(X_train, y_train, numeric_features, categorical_features, output_dir,
                   random_state=42, suffix="con_ceros"):
    """Entrena ambos modelos desde cero y los guarda (ver `save_models`)."""
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    models = build_models(preprocessor, random_state)
    for model in models.values():
        model.fit(X_train, y_train)
    save_models(models, output_dir, suffix)
    return models


if __name__ == "__main__":
    print("Use 1_EDA.ipynb y 2_Modelizacion.ipynb para ejecutar el flujo completo.")
