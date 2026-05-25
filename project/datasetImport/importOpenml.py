import openml
import pandas as pd
import os
from imblearn.datasets import make_imbalance
import numpy as np

def retrieve_custom_openml_datasets(
    min_classes=3,
    max_classes=50,
    min_instances=500,
    max_instances=10000,
    min_features=10,
    max_features=40,
    min_imbalance=3.0,
    max_imbalance=100.0,
    allow_missing_values=False,
    max_results=100
):
    print("Fetching dataset metadata from OpenML...")
    
    datasets_df = openml.datasets.list_datasets(output_format='dataframe')
    
    datasets_df = datasets_df.dropna(
        subset=[
            'NumberOfClasses', 
            'NumberOfInstances', 
            'NumberOfFeatures', 
            'MajorityClassSize', 
            'MinorityClassSize', 
            'NumberOfMissingValues'
        ]
    )
    
    valid_minority = datasets_df['MinorityClassSize'] > 0
    datasets_df = datasets_df[valid_minority].copy()
    
    datasets_df['ImbalanceRatio'] = datasets_df['MajorityClassSize'] / datasets_df['MinorityClassSize']
    
    condition_classes = (datasets_df['NumberOfClasses'] >= min_classes) & (datasets_df['NumberOfClasses'] <= max_classes)
    condition_instances = (datasets_df['NumberOfInstances'] >= min_instances) & (datasets_df['NumberOfInstances'] <= max_instances)
    condition_features = (datasets_df['NumberOfFeatures'] >= min_features) & (datasets_df['NumberOfFeatures'] <= max_features)
    condition_imbalance = (datasets_df['ImbalanceRatio'] >= min_imbalance) & (datasets_df['ImbalanceRatio'] <= max_imbalance)
    
    if not allow_missing_values:
        condition_missing = datasets_df['NumberOfMissingValues'] == 0
    else:
        condition_missing = pd.Series(True, index=datasets_df.index)
    
    filtered_datasets = datasets_df[
        condition_classes & 
        condition_instances & 
        condition_features & 
        condition_imbalance &
        condition_missing
    ]
    
    filtered_datasets = filtered_datasets.sort_values(by='ImbalanceRatio', ascending=False)
    
    columns_to_show = [
        'did', 'name', 'NumberOfClasses', 'NumberOfInstances', 
        'NumberOfFeatures', 'MajorityClassSize', 'MinorityClassSize', 
        'ImbalanceRatio', 'NumberOfMissingValues'
    ]
    
    results = filtered_datasets[columns_to_show].head(max_results)
    
    return results

def execute_excel_configurations(filename="openml_search_logs.xlsx"):
    if not os.path.exists(filename):
        print(f"File {filename} does not exist.")
        return

    config_df = pd.read_excel(filename)
    
    if 'Datasets_Found' not in config_df.columns:
        config_df['Datasets_Found'] = None

    for index, row in config_df.iterrows():
        if pd.isna(row['Datasets_Found']):
            config_dict = {
                'min_classes': row['min_classes'],
                'max_classes': row['max_classes'],
                'min_instances': row['min_instances'],
                'max_instances': row['max_instances'],
                'min_features': row['min_features'],
                'max_features': row['max_features'],
                'min_imbalance': row['min_imbalance'],
                'max_imbalance': row['max_imbalance'],
                'allow_missing_values': bool(row['allow_missing_values']),
                'max_results': int(row['max_results'])
            }
            
            matching_datasets = retrieve_custom_openml_datasets(**config_dict)
            config_df.at[index, 'Datasets_Found'] = len(matching_datasets)

    config_df.to_excel(filename, index=False)
    print(f"File {filename} overwritten.")

def log_search_to_excel(config_dict, num_results, filename="openml_search_logs.xlsx"):
    log_data = config_dict.copy()
    log_data['Datasets_Found'] = num_results
    
    new_row_df = pd.DataFrame([log_data])
    
    if os.path.exists(filename):
        existing_df = pd.read_excel(filename)
        updated_df = pd.concat([existing_df, new_row_df], ignore_index=True)
        updated_df.to_excel(filename, index=False)
    else:
        new_row_df.to_excel(filename, index=False)
        
    print(f"Search configuration logged to {filename}")

def download_datasets_from_search(search_config, output_dir="project/input/multiclass/datasetsFromOpenML"):

    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Searching for datasets with config: {search_config}")
    matching_datasets = retrieve_custom_openml_datasets(**search_config)
    
    if matching_datasets.empty:
        print("No datasets found matching the criteria.")
        return 0
    
    print(f"Found {len(matching_datasets)} datasets")
    
    downloaded_count = 0
    skipped_count = 0
    
    for idx, row in matching_datasets.iterrows():
        dataset_id = row['did']
        dataset_name = row['name']
        
        safe_filename = f"{dataset_name}.csv"
        file_path = os.path.join(output_dir, safe_filename)
        
        if os.path.exists(file_path):
            print(f"[SKIPPED] {dataset_name} (already exists)")
            skipped_count += 1
            continue
        
        try:
            print(f"[DOWNLOADING] {dataset_name} (ID: {dataset_id})...")
            
            dataset = openml.datasets.get_dataset(dataset_id)
            X, y, categorical_indicator, attribute_names = dataset.get_data(
                target=dataset.default_target_attribute
            )
            
            data = X.copy()
            data[dataset.default_target_attribute] = y
            
            data.to_csv(file_path, index=False)
            print(f"[SUCCESS] Saved to {file_path}")
            downloaded_count += 1
            
        except Exception as e:
            print(f"[ERROR] Failed to download {dataset_name}: {str(e)}")
            continue
    
    print(f"Downloaded: {downloaded_count}, Skipped: {skipped_count}")
    return downloaded_count

def create_imbalanced_datasets(
    source_dir="project/input/multiclass/datasetsFromOpenML",
    output_dir="project/input/multiclass/imbalancedDatasetsFromOpenML",
    imbalance_ratio=4.0,
    random_state=42
):
    """
    Create imbalanced versions of binary or multiclass datasets using imblearn.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(source_dir):
        print(f"Source directory {source_dir} does not exist.")
        return 0
    
    csv_files = [f for f in os.listdir(source_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {source_dir}")
        return 0
    
    print(f"Found {len(csv_files)} datasets.")
    print(f"Target imbalance ratio: {imbalance_ratio:.2f}\n")
    
    processed_count = 0
    
    for csv_file in csv_files:
        try:
            file_path = os.path.join(source_dir, csv_file)
            output_path = os.path.join(output_dir, csv_file)
            
            print(f"[PROCESSING] {csv_file}...")
            
            df = pd.read_csv(file_path)
            
            target_col = df.columns[-1]
            
            X = df.drop(columns=[target_col])
            y = df[target_col]
            
            # Get class distribution
            class_counts = y.value_counts().sort_values(ascending=False)
            n_classes = len(class_counts)
            
            print(f"  - Target column: {target_col}")
            print(f"  - Number of classes: {n_classes} ({'binary' if n_classes == 2 else 'multiclass'})")
            print(f"  - Original class distribution: {dict(class_counts)}")
            
            # For each minority class, set the ratio relative to majority
            sampling_strategy = {}
            majority_count = class_counts.iloc[0]
            
            for i, (cls, count) in enumerate(class_counts.items()):
                if i == 0: # Majority class
                    continue
                target_count = int(majority_count / imbalance_ratio)
                sampling_strategy[cls] = target_count
            
            np.random.seed(random_state)
            X_imbalanced, y_imbalanced = make_imbalance(
                X, y,
                sampling_strategy=sampling_strategy,
                random_state=random_state
            )
            
            imbalanced_df = X_imbalanced.copy()
            imbalanced_df[target_col] = y_imbalanced
            
            imbalanced_df.to_csv(output_path, index=False)
            
            # Calculate new imbalance ratio
            new_class_counts = y_imbalanced.value_counts().sort_values(ascending=False)
            new_majority_class = new_class_counts.index[0]
            new_minority_class = new_class_counts.index[-1]
            new_ratio = new_class_counts[new_majority_class] / new_class_counts[new_minority_class]
            
            print(f"  - New imbalance ratio: {new_ratio:.2f}")
            print(f"  - New class distribution: {dict(new_class_counts)}")
            print(f"  - Saved to {output_path}\n")
            processed_count += 1
            
        except Exception as e:
            print(f"[ERROR] Failed to process {csv_file}: {str(e)}\n")
            continue
    
    print(f"Imbalancing complete!")
    print(f"Processed: {processed_count}")
    return processed_count

if __name__ == "__main__":
    
    create_imbalanced_datasets(imbalance_ratio=4.0)
    
    """#execute_excel_configurations()
    search_config = {
        'min_classes': 3,
        'max_classes': 50,
        'min_instances': 500,
        'max_instances': 10000,
        'min_features': 5,
        'max_features': 100,
        'min_imbalance': 0.0,
        'max_imbalance': 3.0,
        'allow_missing_values': False,
        'max_results': 100
    }
    
    download_datasets_from_search(search_config)"""
    
    """matching_datasets = retrieve_custom_openml_datasets(**search_config)
    datasets_found = len(matching_datasets)
    
    log_search_to_excel(search_config, datasets_found)
    if matching_datasets.empty:
        print("No datasets found matching the criteria.")
    else:
        print("\nShowing results:\n")
        print(matching_datasets.to_string(index=False))
        print(f"\nFound {datasets_found} matching datasets.\n")"""