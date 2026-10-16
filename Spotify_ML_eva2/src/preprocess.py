from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

DROP_COLUMNS = ["Unnamed: 0", "track_id", "track_name", "album_name", "artists"]

# Variables que describen el audio de una grabacion. Si dos filas coinciden en todas,
# son la misma grabacion publicada en otro album/version (con otro track_id).
AUDIO_FINGERPRINT = [
    "duration_ms", "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo", "key", "mode", "time_signature",
]
# Identidad "humana" de una cancion.
SONG_KEYS = ["track_name", "artists"]


def deduplicate_songs(df: pd.DataFrame, return_report: bool = False):
    """Deja UNA fila por cancion en tres pasos (siempre conserva la primera ocurrencia):

    1. mismo `track_id` (la misma cancion listada en varios generos),
    2. misma huella de audio (la misma grabacion con otro `track_id`: single, album, recopilatorio),
    3. mismo titulo + artista (versiones de la misma cancion).

    Asi una cancion no puede quedar a la vez en train y en test (fuga de datos).
    Con `return_report=True` devuelve ademas una tabla con filas y ceros tras cada paso.
    """
    required = ["track_id"] + AUDIO_FINGERPRINT + SONG_KEYS + ["popularity"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en el dataset: {missing}")

    def snapshot(step, frame):
        return {
            "paso": step,
            "filas": len(frame),
            "ceros": int((frame["popularity"] == 0).sum()),
            "pct_ceros": round(float((frame["popularity"] == 0).mean() * 100), 2),
        }

    rows = [snapshot("Dataset original", df)]
    out = df.loc[~df["track_id"].duplicated(keep="first")]
    rows.append(snapshot("Sin track_id repetido", out))
    out = out.loc[~out.duplicated(subset=AUDIO_FINGERPRINT, keep="first")]
    rows.append(snapshot("Sin audio idéntico (otro track_id)", out))
    out = out.loc[~out.duplicated(subset=SONG_KEYS, keep="first")]
    rows.append(snapshot("Sin título+artista repetido", out))
    if return_report:
        return out, pd.DataFrame(rows)
    return out


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplica por cancion y elimina las columnas que no se usan como variables del modelo."""
    if "track_id" not in df.columns:
        raise ValueError("El dataset debe contener la columna track_id.")
    return deduplicate_songs(df).drop(columns=DROP_COLUMNS, errors="ignore").copy()


def genres_per_song(df: pd.DataFrame) -> pd.DataFrame:
    """Variante "multi-hot": una columna 0/1 por genero para cada cancion unica.

    Aplica los MISMOS tres pasos de `deduplicate_songs` (mismas filas, mismo orden), pero en vez de
    descartar las copias, junta todos los generos en los que aparecio la cancion. Asi una cancion que
    estaba en 'pop' y en 'dance' queda con un 1 en ambas columnas (en la version original conserva solo
    el primer genero). Devuelve un DataFrame con indice igual al de `deduplicate_songs(df)`.
    """
    def first_index(frame, key_cols):
        keys = [frame[c] for c in key_cols]
        return frame.index.to_series().groupby(keys, sort=False, dropna=False).transform("first")

    rep = first_index(df, ["track_id"])                     # paso 1 (para todas las filas)
    f1 = df.loc[rep.index[rep.index == rep.values]]         # representantes tras el paso 1
    r2 = first_index(f1, AUDIO_FINGERPRINT)                 # paso 2
    f2 = f1.loc[r2.index[r2.index == r2.values]]
    r3 = first_index(f2, SONG_KEYS)                         # paso 3
    final = rep.map(r2).map(r3)                              # representante final de cada fila original
    dummies = pd.get_dummies(df["track_genre"]).astype(int)
    multi = dummies.groupby(final.values).max()
    keep = deduplicate_songs(df).index
    if not multi.index.equals(keep):
        multi = multi.reindex(keep)
    multi.columns = [f"g_{c}" for c in multi.columns]
    return multi


def clean_dataset_multihot(df: pd.DataFrame) -> pd.DataFrame:
    """Igual que `clean_dataset`, pero reemplaza `track_genre` por sus columnas multi-hot `g_<genero>`."""
    base = clean_dataset(df).drop(columns=["track_genre"])
    out = base.join(genres_per_song(df))
    out["explicit"] = out["explicit"].astype(int)
    return out


def split_and_save_data(df_model: pd.DataFrame, output_root: str | Path, test_size=0.20, random_state=42):
    # Importante: aqui NO se agrega ninguna columna derivada de `popularity` (por ejemplo es_cero),
    # porque quedaria como variable de entrada y filtraria la respuesta al modelo.
    output_root = Path(output_root)
    train_dir = output_root / "Train"
    test_dir = output_root / "Test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    train_df, test_df = train_test_split(df_model, test_size=test_size, random_state=random_state)
    train_path = train_dir / "train.csv"
    test_path = test_dir / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    return {"train": train_path, "test": test_path}


def load_processed_data(train_path, test_path):
    return pd.read_csv(train_path), pd.read_csv(test_path)


def build_preprocessor(numeric_features, categorical_features):
    # El escalado no afecta al Random Forest, pero es imprescindible para la Regresion Lineal:
    # sin el, `duration_ms` (~200.000) frente a columnas 0/1 deja el sistema mal condicionado y el
    # ajuste (solver iterativo con datos dispersos) no converge: R2 ~ 0,01 en vez de ~ 0,40.
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])
    return ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])


def build_preprocessor_multihot(numeric_features, genre_features):
    """Preprocesador para la variante multi-hot: numericas imputadas y estandarizadas; las columnas
    de genero (ya 0/1) pasan tal cual."""
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    return ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("genre", "passthrough", genre_features),
    ])
