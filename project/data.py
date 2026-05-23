import os
import re
import logging
import pandas as pd
import numpy as np
import openml.datasets
from pymfe.mfe import MFE

from config import logger, RANDOM_STATE, MIN_SAMPLES_FOR_SMOTE, application_path

# Configure module logger
logger = logging.getLogger(__name__)


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
    # Check minimum samples
    if len(X) < 2:
        return False, "Dataset has fewer than 2 samples"
    
    # For SMOTE-based techniques, check if each class has enough neighbors
    smote_techniques = {'SMOTE', 'BorderlineSMOTE', 'KMeansSMOTE', 'SVMSMOTE'}
    if balancing_technique_name in smote_techniques:
        unique, counts = np.unique(y, return_counts=True)
        min_samples = counts.min()
        if min_samples < MIN_SAMPLES_FOR_SMOTE:
            return False, f"Class with {min_samples} samples is too small for {balancing_technique_name} (requires at least {MIN_SAMPLES_FOR_SMOTE})"
    
    return True, None


def read_file(path):
    """Read a CSV dataset from disk and drop missing rows."""
    # Use pandas auto-detection for delimiter
    df = pd.read_csv(path, sep=None, engine='python')
    df = df.dropna()
    return df, os.path.basename(path)


def read_file_openml(id):
    """Load an OpenML dataset and normalize it into a DataFrame."""
    
    dataset = openml.datasets.get_dataset(id)

    X, y, categorical_indicator, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute, dataset_format="dataframe")

    df = pd.DataFrame(X, columns=attribute_names)
    df["class"] = y
    
    dataset_name = dataset.name + " (id:" + str(id) + ")"
    
    df = df.dropna()
    
    return df, dataset_name


def get_problem_type(y):
    """Return the dataset target type as binary or multiclass."""
    target = y.iloc[:, 0] if isinstance(y, pd.DataFrame) else y
    target = pd.Series(target).dropna()

    if target.nunique() > 2:
        return "multiclass"

    return "binary"


def get_target_class_count(y):
    """Return the number of distinct target classes."""
    target = y.iloc[:, 0] if isinstance(y, pd.DataFrame) else y
    target = pd.Series(target).dropna()

    return int(target.nunique())


def resolve_binary_dataset_path(dataset_name):
    """Resolve a dataset name to read datasets from the input/binary folder."""
    if not dataset_name:
        raise ValueError("dataset_name is required")

    dataset_file = dataset_name if dataset_name.endswith(".csv") else dataset_name + ".csv"
    dataset_path = os.path.join(application_path, "input", "binary", dataset_file)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    return dataset_path


def resolve_multiclass_dataset_path(dataset_name):
    """Resolve a dataset name to read datasets from the input/multiclass folder."""
    if not dataset_name:
        raise ValueError("dataset_name is required")

    dataset_file = dataset_name if dataset_name.endswith(".csv") else dataset_name + ".csv"
    dataset_path = os.path.join(application_path, "input", "multiclass", dataset_file)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    return dataset_path


def get_kb_file_path(base_name, problem_type):
    """Build the knowledge-base CSV path for a problem type."""
    if problem_type == "binary":
        file_name = f"{base_name}.csv"
    else:  # multiclass
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
    
    print("\nDataset                      :", dataset_name, "\n")
    
    # Intelligently detect target column: try last column first, then first column
    # This handles datasets where the target might be in the first column
    y_candidate_last = df.iloc[: , -1].copy()
    y_candidate_first = df.iloc[: , 0].copy()
    
    # Determine which column to use as target
    last_unique = y_candidate_last.nunique()
    first_unique = y_candidate_first.nunique()
    
    if last_unique >= 2:
        # Last column is valid (has 2+ classes)
        X = df.iloc[: , :-1]
        y = y_candidate_last
    elif first_unique >= 2:
        # First column is valid and last is not
        X = df.iloc[: , 1:]
        y = y_candidate_first
    else:
        raise ValueError(f"Cannot find valid target column. Last column has {last_unique} class(es), first column has {first_unique} class(es)")

    mfe = MFE(random_state=RANDOM_STATE, 
          groups=["complexity", "concept", "general", "itemset", "landmarking", "model-based", "statistical"], 
          summary=["mean", "sd", "kurtosis","skewness"])

    y_array = pd.factorize(y)[0]
    mfe.fit(X.values, y_array)
    ft = mfe.extract(suppress_warnings=True)
    
    df_characteristics = pd.DataFrame.from_records(ft)
    
    new_header = df_characteristics.iloc[0]
    df_characteristics = df_characteristics[1:]
    df_characteristics.columns = new_header
    
    df_characteristics.insert(loc=0, column="dataset", value=[dataset_name])
    
    
    encoded_columns = []
    for column_name in X.columns:
        col_dtype = X[column_name].dtype
        # Check for object, category, string, or boolean types
        if (col_dtype == object or 
            col_dtype.name == 'category' or 
            col_dtype.name == 'string' or
            col_dtype == bool or 
            pd.api.types.is_string_dtype(col_dtype)):
            encoded_columns.append(column_name)
    
    if encoded_columns:
        X = pd.get_dummies(X, columns=encoded_columns, drop_first=True)

    X = sanitize_feature_names(X)
    
    # Ensure all remaining object/string columns are converted to numeric
    for column_name in X.columns:
        col_dtype = X[column_name].dtype
        if (col_dtype == object or 
            col_dtype.name == 'category' or 
            col_dtype.name == 'string' or
            pd.api.types.is_string_dtype(col_dtype)):
            X[column_name] = pd.to_numeric(X[column_name], errors='coerce')

    y_encoded, y_categories = pd.factorize(y)
    y = np.asarray(y_encoded, dtype=np.int64)
    
    X = np.asarray(X, dtype=np.float64)

    return X, y, df_characteristics
