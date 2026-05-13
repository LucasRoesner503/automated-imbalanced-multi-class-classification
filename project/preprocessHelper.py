import os
import sys
import time
import datetime
from decimal import Decimal
import pandas as pd
import numpy as np
import openml.datasets
from pymfe.mfe import MFE
from imblearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score, cross_validate
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids, CondensedNearestNeighbour, EditedNearestNeighbours, RepeatedEditedNearestNeighbours, AllKNN, InstanceHardnessThreshold, NearMiss, NeighbourhoodCleaningRule, OneSidedSelection, TomekLinks
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN, BorderlineSMOTE, KMeansSMOTE, SVMSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from imblearn.ensemble import EasyEnsembleClassifier, RUSBoostClassifier, BalancedBaggingClassifier, BalancedRandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, make_scorer, cohen_kappa_score, precision_score, recall_score, matthews_corrcoef
from imblearn.metrics import geometric_mean_score
import traceback
import warnings
warnings.filterwarnings("ignore")


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


# determine if application is a script file or frozen exe
application_path = ""
if getattr(sys, 'frozen', False):
    application_path = os.path.dirname(sys.executable)
elif __file__:
    application_path = os.path.dirname(__file__)



def resolve_multiclass_dataset_path(dataset_name):
    """Resolve a dataset name to a CSV file inside input/multiclass."""
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



def reduce_unprocessed_datasets_to_1500_samples():
    """
    Find multiclass datasets not yet in KB and reduce them to 1500 samples.
    
    Checks kb_results_multiclass.csv to identify which datasets have been processed.
    For all other datasets in input/multiclass, creates a 1500-sample subset with
    preserved class distribution.
    
    Returns:
        dict with keys: processed (count), success (list), skipped (list), failed (list)
    """
    try:
        datasets_dir = os.path.join(application_path, "input", "multiclass")
        
        if not os.path.isdir(datasets_dir):
            raise FileNotFoundError(f"Directory not found: {datasets_dir}")
        
        # Get all CSV files (excluding _subset.csv files)
        dataset_files = sorted([
            file_name for file_name in os.listdir(datasets_dir)
            if file_name.lower().endswith(".csv") and 
               not file_name.lower().endswith("_subset.csv") and
               os.path.isfile(os.path.join(datasets_dir, file_name))
        ])
        
        if not dataset_files:
            print("No CSV datasets found in input/multiclass.")
            return {"processed": 0, "success": [], "skipped": [], "failed": []}
        
        # Load already-processed datasets from KB
        kb_results_path = get_kb_file_path("kb_results", "multiclass")
        processed_datasets = set()
        if os.path.exists(kb_results_path):
            df_kb_results = pd.read_csv(kb_results_path, sep=",")
            if "dataset" in df_kb_results.columns:
                processed_datasets = set(df_kb_results["dataset"].dropna().astype(str).tolist())
        
        success = []
        failed = []
        skipped = []
        
        print(f"\nFound {len(dataset_files)} multiclass datasets.")
        print(f"Already processed: {len(processed_datasets)}")
        
        unprocessed_count = 0
        for index, dataset_file in enumerate(dataset_files, start=1):
            if dataset_file in processed_datasets:
                skipped.append(dataset_file)
                continue
            
            unprocessed_count += 1
            print(f"\n[{unprocessed_count}] Reducing to 1500 samples: {dataset_file}")
            
            try:
                df_reduced, orig_size, red_size, dist_orig, dist_red = reduce_dataset_to_1000_samples(
                    dataset_file, 
                    target_size=1500
                )
                
                if df_reduced is not None:
                    success.append(dataset_file)
                else:
                    failed.append(dataset_file)
                    
            except Exception as e:
                print(f"  Error: {type(e).__name__}: {e}")
                failed.append(dataset_file)
        
        print("\n" + "="*60)
        print("Batch reduction finished.")
        print(f"  Reduced to 1500 samples: {len(success)}")
        print(f"  Already processed:       {len(skipped)}")
        print(f"  Failed:                  {len(failed)}")
        print("="*60)
        
        if failed:
            print("\nFailed datasets:")
            for dataset_file in failed:
                print(f"  - {dataset_file}")
        
        return {
            "processed": len(success) + len(failed),
            "success": success,
            "skipped": skipped,
            "failed": failed,
        }
        
    except Exception as e:
        print(f"Error in reduce_unprocessed_datasets_to_1500_samples: {type(e).__name__}: {e}")
        traceback.print_exc()
        return {"processed": 0, "success": [], "skipped": [], "failed": []}


def reduce_dataset_to_1000_samples(dataset_name, target_size=1000):
    """
    Reduce a multiclass dataset to 1000 records while maintaining imbalance ratio.
    
    Uses stratified sampling to preserve the class distribution proportions.
    
    Args:
        dataset_name: Name of the CSV file in input/multiclass (with or without .csv extension)
        target_size: Target number of records (default: 1000)
    
    Returns:
        Tuple of (reduced_df, original_size, reduced_size, class_distribution_original, class_distribution_reduced)
    """
    try:
        # Load the dataset
        dataset_path = resolve_multiclass_dataset_path(dataset_name)
        df = pd.read_csv(dataset_path)
        df = df.dropna()
        
        original_size = len(df)
        
        # Get the target column (last column)
        target_col = df.columns[-1]
        
        # Calculate original class distribution
        class_dist_original = df[target_col].value_counts(normalize=True).to_dict()
        
        # If dataset is already smaller than target, return as-is
        if original_size <= target_size:
            print(f"Dataset {dataset_name} has {original_size} records (< {target_size}). No reduction needed.")
            class_dist_reduced = df[target_col].value_counts(normalize=True).to_dict()
            return df, original_size, original_size, class_dist_original, class_dist_reduced
        
        # Use stratified sampling to maintain class proportions
        # Group by target, sample from each group, then concat back
        sampled_groups = []
        for class_val in df[target_col].unique():
            class_df = df[df[target_col] == class_val]
            n_samples = max(1, int(len(class_df) * target_size / original_size))
            sampled_groups.append(class_df.sample(n=n_samples, random_state=42))
        
        df_reduced = pd.concat(sampled_groups, ignore_index=True)
        
        # Fine-tune to exact target size if needed
        current_size = len(df_reduced)
        if current_size > target_size:
            df_reduced = df_reduced.sample(n=target_size, random_state=42).reset_index(drop=True)
        elif current_size < target_size:
            # If we're under, add a few more samples maintaining proportions
            needed = target_size - current_size
            for _ in range(needed):
                # Add one sample from the class with highest count in reduced set
                class_counts = df_reduced[target_col].value_counts()
                most_common_class = class_counts.idxmax()
                
                # Sample from original dataset for this class
                class_samples = df[df[target_col] == most_common_class]
                new_sample = class_samples.sample(n=1, random_state=42)
                df_reduced = pd.concat([df_reduced, new_sample], ignore_index=True)
        
        reduced_size = len(df_reduced)
        class_dist_reduced = df_reduced[target_col].value_counts(normalize=True).to_dict()
        
        print(f"\nDataset reduction summary for {dataset_name}:")
        print(f"  Original size: {original_size} records")
        print(f"  Reduced size:  {reduced_size} records")
        print(f"  Reduction: {100 * (original_size - reduced_size) / original_size:.1f}%")
        print(f"\n  Original class distribution:")
        for cls, prop in sorted(class_dist_original.items()):
            print(f"    {cls}: {prop:.3f}")
        print(f"\n  Reduced class distribution:")
        for cls, prop in sorted(class_dist_reduced.items()):
            print(f"    {cls}: {prop:.3f}")
        
        # Save the subset dataset with "_subset" suffix
        dataset_base = os.path.splitext(dataset_name)[0]  # Remove .csv extension
        subset_filename = f"{dataset_base}_subset.csv"
        subset_path = os.path.join(os.path.dirname(dataset_path), subset_filename)
        df_reduced.to_csv(subset_path, index=False, sep=",")
        print(f"\n  Subset saved to: {subset_filename}")
        
        return df_reduced, original_size, reduced_size, class_dist_original, class_dist_reduced
        
    except Exception as e:
        print(f"Error reducing dataset {dataset_name}: {type(e).__name__}: {e}")
        traceback.print_exc()
        return None, 0, 0, {}, {}
