import openml
import pandas as pd
import os

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

if __name__ == "__main__":
    execute_excel_configurations()
    """search_config = {
        'min_classes': 3,
        'max_classes': 50,
        'min_instances': 500,
        'max_instances': 10000,
        'min_features': 10,
        'max_features': 40,
        'min_imbalance': 3.0,
        'max_imbalance': 100.0,
        'allow_missing_values': False,
        'max_results': 100
    }
    
    matching_datasets = retrieve_custom_openml_datasets(**search_config)
    datasets_found = len(matching_datasets)
    
    log_search_to_excel(search_config, datasets_found)
    if matching_datasets.empty:
        print("No datasets found matching the criteria.")
    else:
        print("\nShowing results:\n")
        print(matching_datasets.to_string(index=False))
        print(f"\nFound {datasets_found} matching datasets.\n")"""