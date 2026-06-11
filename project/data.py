import os
import re
import logging
import pandas as pd
import numpy as np
import openml.datasets
from pymfe.mfe import MFE

from config import RANDOM_STATE, MIN_SAMPLES_FOR_SMOTE, application_path

logger = logging.getLogger(__name__)

_SMOTE_TECHNIQUES = {'SMOTE', 'BorderlineSMOTE', 'KMeansSMOTE', 'SVMSMOTE'}

_TARGET_COLUMN_NAMES = {'class', 'target', 'label', 'y', 'outcome'}


def validate_dataset(X, y, balancing_technique_name):
    """
    Validate dataset for compatibility with resampling techniques.

    Args:
        X: Feature matrix
        y: Target labels
        balancing_technique_name: Name of the balancing technique to validate

    Returns:
        tuple: (is_valid, error_message)
    """
    if len(X) < 2:
        return False, "Dataset has fewer than 2 samples"

    if balancing_technique_name in _SMOTE_TECHNIQUES:
        unique, counts = np.unique(y, return_counts=True)
        min_samples = counts.min()
        if min_samples < MIN_SAMPLES_FOR_SMOTE:
            return False, f"Class with {min_samples} samples is too small for {balancing_technique_name} (requires at least {MIN_SAMPLES_FOR_SMOTE})"

    return True, None


def read_file(path):
    """Read a CSV dataset from disk and drop missing rows."""
    df = pd.read_csv(path, sep=None, engine='python')
    df = df.dropna()
    return df, os.path.basename(path)


def read_file_openml(dataset_id):
    """Load an OpenML dataset and normalize it into a DataFrame."""
    dataset = openml.datasets.get_dataset(dataset_id)

    X, y, categorical_indicator, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute, dataset_format="dataframe")

    df = pd.DataFrame(X, columns=attribute_names)
    df["class"] = y

    dataset_name = dataset.name + " (id:" + str(dataset_id) + ")"

    df = df.dropna()

    return df, dataset_name


def toSeries(y):
    target = y.iloc[:, 0] if isinstance(y, pd.DataFrame) else y
    return pd.Series(target).dropna()


def get_problem_type(y):
    """Return the dataset target type as binary or multiclass."""
    return "multiclass" if toSeries(y).nunique() > 2 else "binary"


def get_target_class_count(y):
    """Return the number of distinct target classes."""
    return int(toSeries(y).nunique())


def resolve_dataset_path(dataset_name, problem_type):
    """Resolve a dataset name to its path under input/<problem_type>/."""
    if not dataset_name:
        raise ValueError("dataset_name is required")

    dataset_file = dataset_name if dataset_name.endswith(".csv") else dataset_name + ".csv"
    dataset_path = os.path.join(application_path, "input", problem_type, dataset_file)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    return dataset_path


def resolve_binary_dataset_path(dataset_name):
    """Resolve a dataset name to read datasets from the input/binary folder."""
    return resolve_dataset_path(dataset_name, "binary")


def resolve_multiclass_dataset_path(dataset_name):
    """Resolve a dataset name to read datasets from the input/multiclass folder."""
    return resolve_dataset_path(dataset_name, "multiclass")


def get_kb_file_path(base_name, problem_type):
    """Build the knowledge-base CSV path for a problem type."""
    if problem_type == "binary":
        file_name = f"{base_name}.csv"
    else:
        file_name = f"{base_name}_{problem_type}.csv"
    return os.path.join(application_path, "output", file_name)


def load_kb_dataframe(base_name, problem_type, columns=None):
    """Load a problem-type-specific KB file or return an empty DataFrame."""
    file_path = get_kb_file_path(base_name, problem_type)

    if os.path.exists(file_path):
        dataframe = pd.read_csv(file_path, sep=",")
        if columns is not None:
            for column_name in columns:
                if column_name not in dataframe.columns:
                    dataframe[column_name] = np.nan
            dataframe = dataframe[columns]
        return dataframe

    if columns is not None:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame()


def sanitize_feature_names(X):
    """Normalize feature names to JSON-safe alphanumeric/underscore labels."""
    rename_map = {}
    used_names = set()

    for original_name in X.columns:
        cleaned_name = re.sub(r"[^0-9a-zA-Z_]", "_", str(original_name))
        cleaned_name = cleaned_name.strip("_")

        if not cleaned_name:
            cleaned_name = "feature"

        if cleaned_name[0].isdigit():
            cleaned_name = f"f_{cleaned_name}"

        unique_name = cleaned_name
        suffix = 1
        while unique_name in used_names:
            suffix += 1
            unique_name = f"{cleaned_name}_{suffix}"

        used_names.add(unique_name)
        rename_map[original_name] = unique_name

    return X.rename(columns=rename_map)


def features_labels(df, dataset_name):
    """Split features, encode the target, and compute dataset characteristics."""

    logger.info("\nDataset                      : %s\n", dataset_name)

    named_target = next((col for col in df.columns if str(col).strip().lower() in _TARGET_COLUMN_NAMES), None)

    if named_target is not None:
        X = df.drop(columns=[named_target])
        y = df[named_target].copy()
    else:
        y_candidate_last = df.iloc[:, -1].copy()
        y_candidate_first = df.iloc[:, 0].copy()

        last_unique = y_candidate_last.nunique()
        first_unique = y_candidate_first.nunique()

        if last_unique >= 2:
            X = df.iloc[:, :-1]
            y = y_candidate_last
        elif first_unique >= 2:
            X = df.iloc[:, 1:]
            y = y_candidate_first
        else:
            raise ValueError(f"Cannot find valid target column. Last column has {last_unique} class(es), first column has {first_unique} class(es)")

    encoded_columns = []
    bool_columns = []
    for column_name in X.columns:
        col_dtype = X[column_name].dtype
        if col_dtype == bool:
            bool_columns.append(column_name)
        elif (col_dtype == object or
              col_dtype.name == 'category' or
              col_dtype.name == 'string' or
              pd.api.types.is_string_dtype(col_dtype)):
            encoded_columns.append(column_name)

    if bool_columns:
        X = X.copy()
        for col in bool_columns:
            X[col] = X[col].astype(int)

    if encoded_columns:
        X = pd.get_dummies(X, columns=encoded_columns, drop_first=True)

    for column_name in X.columns:
        col_dtype = X[column_name].dtype
        if (col_dtype == object or
              col_dtype.name == 'category' or
              col_dtype.name == 'string' or
              pd.api.types.is_string_dtype(col_dtype)):
            X[column_name] = pd.to_numeric(X[column_name], errors='coerce')

    X = sanitize_feature_names(X)
    X_array = np.asarray(X, dtype=np.float64)

    y_encoded, _ = pd.factorize(y)
    y_array = np.asarray(y_encoded, dtype=np.int64)

    mfe = MFE(random_state=RANDOM_STATE,
          groups=["complexity", "concept", "general", "itemset", "landmarking", "model-based", "statistical"],
          summary=["mean", "sd", "kurtosis", "skewness"])
    mfe.fit(X_array, y_array)
    names, values = mfe.extract(suppress_warnings=True)

    df_characteristics = pd.DataFrame([values], columns=names)
    df_characteristics.insert(loc=0, column="dataset", value=[dataset_name])

    return X_array, y_array, df_characteristics
