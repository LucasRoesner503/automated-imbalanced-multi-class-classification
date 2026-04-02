import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import os

def count_missing_values(dataset):
    """
    Calculates the frequency of missing values within the dataset.
    Returns column-specific counts and the aggregate dataset total.
    """
    # Calculate missing values per column
    column_counts = dataset.isna().sum()
    
    # Calculate total missing values in the entire dataset
    total_missing = column_counts.sum()
    
    # Isolate columns containing missing records
    missing_only = column_counts[column_counts > 0]
    
    return missing_only, total_missing

def populate_and_export_missing_values(dataset, file_path):
    """
    Executes imputation and exports the dataset.
    """
    # Remove features containing exclusively null values
    dataset = dataset.dropna(axis=1, how='all')
    
    numerical_features = dataset.select_dtypes(include=[np.number]).columns
    categorical_features = dataset.select_dtypes(exclude=[np.number]).columns

    numerical_imputer = SimpleImputer(strategy='median')
    categorical_imputer = SimpleImputer(strategy='most_frequent')

    if len(numerical_features) > 0:
        dataset[numerical_features] = numerical_imputer.fit_transform(dataset[numerical_features])
        
    if len(categorical_features) > 0:
        dataset[categorical_features] = categorical_imputer.fit_transform(dataset[categorical_features])

    file_name, file_extension = os.path.splitext(file_path)
    export_path = f"{file_name}_clean{file_extension}"
    
    dataset.to_csv(export_path, index=False)

    return dataset, export_path


if __name__ == "__main__":
# Example execution:
 target_file = '/Users/joaolucasroesner/downloaded_datasets/Diabetes130US_seed_2_nrows_2000_nclasses_10_ncols_100_stratify_True.csv'
 dataset = pd.read_csv(target_file)
 column_missing, total_missing = count_missing_values(dataset)
 print(f"Total Missing: {total_missing}")
 print(column_missing)
 populated_dataset, final_path = populate_and_export_missing_values(dataset, target_file)
 print(f"Missing values after imputation: {populated_dataset.isna().sum().sum()}")
 print(f"Cleaned dataset exported to: {final_path}")