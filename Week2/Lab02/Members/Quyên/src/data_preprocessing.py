from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DROP_COLUMNS = ["Id", "Alley", "PoolQC", "Fence", "MiscFeature"]
TARGET = "SalePrice"


def load_house_prices(data_dir):
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    return train, test


def prepare_features(train, test, target=TARGET, drop_columns=None):
    drop_columns = list(DROP_COLUMNS if drop_columns is None else drop_columns)
    train_model = train.drop(columns=drop_columns).copy()
    test_model = test.drop(columns=drop_columns).copy()
    x = train_model.drop(columns=[target])
    y = train_model[target].copy()
    x_test = test_model.copy()
    return x, y, x_test


def build_preprocessor(x_train):
    numeric_features = x_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = x_train.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )
    return preprocessor, numeric_features, categorical_features
