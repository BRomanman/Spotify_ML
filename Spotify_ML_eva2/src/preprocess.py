from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

DROP_COLUMNS = ["Unnamed: 0", "track_id", "track_name", "album_name", "artists"]

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica las mismas decisiones de preparacion documentadas en el notebook original."""
    if "track_id" not in df.columns:
        raise ValueError("El dataset debe contener la columna track_id.")
    cleaned = df.loc[~df["track_id"].duplicated(keep="first")].drop(columns=DROP_COLUMNS, errors="ignore").copy()
    return cleaned

def split_and_save_data(df_model: pd.DataFrame, output_root: str | Path, test_size=0.20, random_state=42):
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
    numeric_transformer = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])
    return ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])
