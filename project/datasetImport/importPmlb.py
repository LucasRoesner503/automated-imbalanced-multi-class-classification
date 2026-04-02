import pandas as pd
from pmlb import fetch_data
import os

def retrieve_and_download_pmlb(
    min_classes=3,
    max_classes=100,
    min_instances=500,
    max_instances=20000,
    min_features=5,
    max_features=80,
    min_imbalance=0.1,
    max_imbalance=1.0,
    max_results=100,
    output_dir="pmlb_datasets"
):
    """
    Filters the PMLB metadata and downloads matching datasets locally.
    """
    # Create the storage directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    source_url = "https://raw.githubusercontent.com/EpistasisLab/pmlb/master/pmlb/all_summary_stats.tsv"
    
    # Load summary statistics from the repository
    try:
        datasets_df = pd.read_csv(source_url, sep='\t')
    except Exception as e:
        print(f"Error accessing PMLB metadata: {e}")
        return None

    # Apply filtering criteria
    condition_task = datasets_df['task'] == 'classification'
    condition_classes = (datasets_df['n_classes'] >= min_classes) & (datasets_df['n_classes'] <= max_classes)
    condition_instances = (datasets_df['n_instances'] >= min_instances) & (datasets_df['n_instances'] <= max_instances)
    condition_features = (datasets_df['n_features'] >= min_features) & (datasets_df['n_features'] <= max_features)
    condition_imbalance = (datasets_df['imbalance'] >= min_imbalance) & (datasets_df['imbalance'] <= max_imbalance)
    
    filtered_df = datasets_df[
        condition_task & condition_classes & condition_instances & 
        condition_features & condition_imbalance
    ]
    
    # Sort and limit results
    filtered_df = filtered_df.sort_values(by='imbalance', ascending=False).head(max_results)
    
    if filtered_df.empty:
        print("No datasets match the specified parameters.")
        return filtered_df

    # Iterative download process
    for dataset_name in filtered_df['dataset']:
        try:
            print(f"Fetching dataset: {dataset_name}...")
            
            # Fetch data from PMLB (returns a pandas DataFrame)
            data = fetch_data(dataset_name, local_cache_dir=output_dir)
            
            # Save specifically as a CSV in your output directory
            file_path = os.path.join(output_dir, f"{dataset_name}.csv")
            data.to_csv(file_path, index=False)
            
            print(f"Successfully saved {dataset_name} to {file_path}")
        except Exception as e:
            print(f"Failed to download {dataset_name}: {e}")

    return filtered_df[['dataset', 'n_classes', 'n_instances', 'n_features', 'imbalance']]

if __name__ == "__main__":
    # Define search and download configuration
    config = {
        'min_classes': 3,
        'max_classes': 300,
        'min_instances': 400,
        'max_instances': 20000,
        'min_features': 5,
        'max_features': 150,
        'min_imbalance': 0.2,
        'max_imbalance': 1.0,
        'max_results': 30,  # Reduced for initial testing
        'output_dir': "pmlb_downloads"
    }
    
    results = retrieve_and_download_pmlb(**config)
    
    if results is not None and not results.empty:
        print("\nDownload process complete. Summary of datasets acquired:")
        print(results.to_string(index=False))