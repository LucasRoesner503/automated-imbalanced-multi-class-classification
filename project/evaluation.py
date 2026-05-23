import os
import logging
import pandas as pd
import numpy as np
import shutil

from config import logger, application_path, get_full_results_columns
from data import (
    get_kb_file_path, load_kb_dataframe, get_problem_type,
    resolve_binary_dataset_path, resolve_multiclass_dataset_path
)
from ml import execute_byCharacteristics, get_best_results_by_characteristics

# Configure module logger
logger = logging.getLogger(__name__)


def _create_loocv_kb_files(problem_type, exclude_dataset):
    """
    Create temporary KB files excluding a specific dataset for LOOCV.
    
    Args:
        problem_type: "binary" or "multiclass"
        exclude_dataset: Name of dataset to exclude from KB
    
    Returns:
        Tuple of (temp_char_path, temp_results_path, original_char_path, original_results_path)
    """
    original_char_path = get_kb_file_path("kb_characteristics", problem_type)
    original_results_path = get_kb_file_path("kb_full_results", problem_type)
    
    # Create backup paths
    temp_char_path = original_char_path + ".loocv_temp"
    temp_results_path = original_results_path + ".loocv_temp"
    
    # Load current KB data
    df_char = load_kb_dataframe("kb_characteristics", problem_type)
    df_results = load_kb_dataframe("kb_full_results", problem_type, columns=get_full_results_columns())
    
    # Remove the dataset to evaluate
    df_char_filtered = df_char.loc[df_char['dataset'] != exclude_dataset].copy()
    df_results_filtered = df_results.loc[df_results['dataset'] != exclude_dataset].copy()
    
    # Save temporary KB files
    df_char_filtered.to_csv(temp_char_path, sep=",", index=False)
    df_results_filtered.to_csv(temp_results_path, sep=",", index=False)
    
    # Backup original files
    shutil.copy(original_char_path, original_char_path + ".backup")
    shutil.copy(original_results_path, original_results_path + ".backup")
    
    # Copy temp files to original locations
    shutil.copy(temp_char_path, original_char_path)
    shutil.copy(temp_results_path, original_results_path)
    
    return temp_char_path, temp_results_path, original_char_path, original_results_path


def _restore_kb_files(problem_type, original_char_path, original_results_path, temp_char_path, temp_results_path):
    """Restore original KB files after LOOCV evaluation."""
    try:
        # Restore from backups
        shutil.copy(original_char_path + ".backup", original_char_path)
        shutil.copy(original_results_path + ".backup", original_results_path)
        
        # Clean up temporary files
        for path in [temp_char_path, temp_results_path, 
                     original_char_path + ".backup", original_results_path + ".backup"]:
            if os.path.exists(path):
                os.remove(path)
        
        logger.debug(f"KB files restored for {problem_type}")
    except Exception as e:
        logger.error(f"Error restoring KB files: {str(e)}", exc_info=True)


def loocv_evaluate_recommendations(problem_type="multiclass"):
    """
    Perform leave-one-out cross-validation on recommendation system.
    
    For each dataset in the KB, temporarily removes it from the KB files,
    calls execute_byCharacteristics to get recommendations from other datasets,
    then compares against the actual best combination to evaluate recommendation quality.
    """
    
    logger.info(f"Starting LOOCV evaluation for {problem_type} datasets...")
    
    # Remove old evaluation results
    output_file = os.path.join(application_path, "output", f"loocv_evaluation_{problem_type}.csv")
    if os.path.exists(output_file):
        try:
            os.remove(output_file)
            logger.debug(f"Removed old evaluation file: {output_file}")
        except Exception as e:
            logger.warning(f"Could not remove old evaluation file: {str(e)}")
    
    # Load original KB data for reference
    df_characteristics_original = load_kb_dataframe("kb_characteristics", problem_type)
    df_full_results_original = load_kb_dataframe("kb_full_results", problem_type, columns=get_full_results_columns())
    
    if df_characteristics_original.empty or df_full_results_original.empty:
        logger.error(f"KB data not available for problem_type: {problem_type}")
        return pd.DataFrame()
    
    results = []
    dataset_names = df_characteristics_original["dataset"].unique()
    
    logger.info(f"Found {len(dataset_names)} datasets to evaluate")
    
    for idx, dataset_name in enumerate(dataset_names, 1):
        temp_char_path = None
        temp_results_path = None
        original_char_path = None
        original_results_path = None
        
        try:
            logger.info(f"[{idx}/{len(dataset_names)}] LOOCV evaluating: {dataset_name}")
            
            # Create temporary KB files without this dataset
            temp_char_path, temp_results_path, original_char_path, original_results_path = \
                _create_loocv_kb_files(problem_type, dataset_name)
            
            # Get the actual best combination for this dataset (before removing from KB)
            actual_results = df_full_results_original.loc[df_full_results_original['dataset'] == dataset_name]
            
            if actual_results.empty:
                logger.warning(f"No results found for {dataset_name}")
                _restore_kb_files(problem_type, original_char_path, original_results_path, 
                                temp_char_path, temp_results_path)
                continue
            
            # Find the actual best (highest final score)
            best_idx = actual_results["final score"].idxmax()
            best_result = actual_results.loc[best_idx]
            best_preproc = best_result["pre processing"]
            best_algorithm = best_result["algorithm"]
            best_score = best_result["final score"]
            
            # Get stored recommendation (what was stored as best for this dataset)
            dataset_char = df_characteristics_original.loc[df_characteristics_original['dataset'] == dataset_name]
            
            if dataset_char.empty:
                logger.warning(f"Dataset {dataset_name} not found in characteristics")
                _restore_kb_files(problem_type, original_char_path, original_results_path, 
                                temp_char_path, temp_results_path)
                continue
            
            
            # Get the top recommendation from the system based on similar datasets
            # (with current dataset temporarily excluded from KB)
            try:
                # Pass the dataset's characteristics so function can use them for similarity computation
                # even though the dataset is excluded from the comparison KB
                recommendations_df = get_best_results_by_characteristics(dataset_name, problem_type, current_features_df=dataset_char)
                
                if recommendations_df is not None and not recommendations_df.empty:
                    # Get the top recommendation (first row)
                    recommended_preproc = recommendations_df.iloc[0]["pre processing"]
                    recommended_algorithm = recommendations_df.iloc[0]["algorithm"]
                    logger.debug(f"Top recommendation for {dataset_name}: {recommended_preproc} + {recommended_algorithm}")
                else:
                    # No recommendations available
                    recommended_preproc = "NO_RECOMMENDATION"
                    recommended_algorithm = "NO_RECOMMENDATION"
                    logger.debug(f"No recommendations found for {dataset_name}")
                    
            except Exception as e:
                logger.warning(f"Could not get recommendations via get_best_results_by_characteristics: {str(e)}", exc_info=True)
                # Mark as no recommendation on error
                recommended_preproc = "NO_RECOMMENDATION"
                recommended_algorithm = "NO_RECOMMENDATION"
            
            # Get the score of the recommended combination
            recommended_results = actual_results.loc[
                (actual_results["pre processing"] == recommended_preproc) & 
                (actual_results["algorithm"] == recommended_algorithm)
            ]
            
            if recommended_results.empty:
                logger.warning(f"Recommended combination not found in results for {dataset_name}")
                recommended_score = np.nan
                rank = np.nan
                is_top_1 = False
                is_top_3 = False
            else:
                recommended_score = recommended_results.iloc[0]["final score"]
                
                # Rank the recommended combination within all combinations
                sorted_scores = actual_results["final score"].sort_values(ascending=False).reset_index(drop=True)
                rank = (sorted_scores == recommended_score).idxmax() + 1  # 1-indexed rank
                
                is_top_1 = (rank == 1)
                is_top_3 = (rank <= 3)
            
            # Calculate delta
            delta_score = recommended_score - best_score if not np.isnan(recommended_score) else np.nan
            
            # Get additional metrics from both recommended and best combinations
            rec_balanced_acc = recommended_results.iloc[0]["balanced accuracy"] if not recommended_results.empty else np.nan
            rec_f1 = recommended_results.iloc[0]["f1 score"] if not recommended_results.empty else np.nan
            rec_roc_auc = recommended_results.iloc[0]["roc auc"] if not recommended_results.empty else np.nan
            
            best_balanced_acc = best_result.get("balanced accuracy", np.nan)
            best_f1 = best_result.get("f1 score", np.nan)
            best_roc_auc = best_result.get("roc auc", np.nan)
            
            # For multiclass, use multiclass metrics
            if problem_type == "multiclass":
                rec_precision = recommended_results.iloc[0]["multiclass precision macro"] if not recommended_results.empty else np.nan
                rec_recall = recommended_results.iloc[0]["multiclass recall macro"] if not recommended_results.empty else np.nan
                rec_f1_weighted = recommended_results.iloc[0]["multiclass f1 weighted"] if not recommended_results.empty else np.nan
                
                best_precision = best_result.get("multiclass precision macro", np.nan)
                best_recall = best_result.get("multiclass recall macro", np.nan)
                best_f1_weighted = best_result.get("multiclass f1 weighted", np.nan)
                
                results.append({
                    "dataset": dataset_name,
                    "recommended_preproc": recommended_preproc,
                    "recommended_algorithm": recommended_algorithm,
                    "recommended_score": round(recommended_score, 4) if not np.isnan(recommended_score) else np.nan,
                    "best_preproc": best_preproc,
                    "best_algorithm": best_algorithm,
                    "best_score": round(best_score, 4),
                    "rank": rank if not np.isnan(rank) else np.nan,
                    "is_top_1": is_top_1,
                    "is_top_3": is_top_3,
                    "delta_score": round(delta_score, 4) if not np.isnan(delta_score) else np.nan,
                    "rec_precision_macro": round(rec_precision, 4) if not np.isnan(rec_precision) else np.nan,
                    "rec_recall_macro": round(rec_recall, 4) if not np.isnan(rec_recall) else np.nan,
                    "rec_f1_weighted": round(rec_f1_weighted, 4) if not np.isnan(rec_f1_weighted) else np.nan,
                    "best_precision_macro": round(best_precision, 4) if not np.isnan(best_precision) else np.nan,
                    "best_recall_macro": round(best_recall, 4) if not np.isnan(best_recall) else np.nan,
                    "best_f1_weighted": round(best_f1_weighted, 4) if not np.isnan(best_f1_weighted) else np.nan,
                })
            else:
                results.append({
                    "dataset": dataset_name,
                    "recommended_preproc": recommended_preproc,
                    "recommended_algorithm": recommended_algorithm,
                    "recommended_score": round(recommended_score, 4) if not np.isnan(recommended_score) else np.nan,
                    "best_preproc": best_preproc,
                    "best_algorithm": best_algorithm,
                    "best_score": round(best_score, 4),
                    "rank": rank if not np.isnan(rank) else np.nan,
                    "is_top_1": is_top_1,
                    "is_top_3": is_top_3,
                    "delta_score": round(delta_score, 4) if not np.isnan(delta_score) else np.nan,
                    "rec_balanced_accuracy": round(rec_balanced_acc, 4) if not np.isnan(rec_balanced_acc) else np.nan,
                    "rec_f1_score": round(rec_f1, 4) if not np.isnan(rec_f1) else np.nan,
                    "rec_roc_auc": round(rec_roc_auc, 4) if not np.isnan(rec_roc_auc) else np.nan,
                    "best_balanced_accuracy": round(best_balanced_acc, 4) if not np.isnan(best_balanced_acc) else np.nan,
                    "best_f1_score": round(best_f1, 4) if not np.isnan(best_f1) else np.nan,
                    "best_roc_auc": round(best_roc_auc, 4) if not np.isnan(best_roc_auc) else np.nan,
                })
        
        except Exception as e:
            logger.error(f"Error evaluating {dataset_name}: {str(e)}", exc_info=True)
        
        finally:
            # Always restore KB files
            if original_char_path and original_results_path and temp_char_path and temp_results_path:
                _restore_kb_files(problem_type, original_char_path, original_results_path, 
                                temp_char_path, temp_results_path)
    
    df_results = pd.DataFrame(results)
    
    if df_results.empty:
        logger.warning("No results generated from LOOCV evaluation")
        return df_results
    
    # Calculate summary statistics
    total_datasets = len(df_results)
    top_1_count = df_results["is_top_1"].sum()
    top_3_count = df_results["is_top_3"].sum()
    avg_rank = df_results["rank"].mean()
    avg_delta = df_results["delta_score"].mean()
    
    logger.info(f"\nLOOCV Evaluation Summary ({problem_type}):")
    logger.info(f"  Total datasets: {total_datasets}")
    logger.info(f"  Top 1 (optimal): {top_1_count} ({100*top_1_count/total_datasets:.1f}%)")
    logger.info(f"  Top 3: {top_3_count} ({100*top_3_count/total_datasets:.1f}%)")
    logger.info(f"  Average rank: {avg_rank:.2f}")
    logger.info(f"  Average delta score: {avg_delta:.4f}")
    
    # Save results to CSV
    output_path = os.path.join(application_path, "output", f"loocv_evaluation_{problem_type}.csv")
    df_results.to_csv(output_path, sep=",", index=False)
    logger.info(f"Results saved to: {output_path}")
    
    return df_results


def loocv_evaluate_all(problem_type=None):
    """
    Run LOOCV evaluation for specified problem types.
    
    Args:
        problem_type: str or None
            - "binary": Run only binary classification evaluation
            - "multiclass": Run only multiclass classification evaluation
            - None or "both": Run both binary and multiclass (default)
    
    Returns:
        pd.DataFrame: Evaluation results
    """
    if problem_type is None:
        problem_type = "both"
    
    valid_types = ["binary", "multiclass", "both"]
    if problem_type not in valid_types:
        raise ValueError(f"problem_type must be one of {valid_types}, got '{problem_type}'")
    
    logger.info(f"Starting LOOCV evaluation for: {problem_type}")
    
    # Remove old evaluation result files
    if problem_type in ["binary", "both"]:
        binary_file = os.path.join(application_path, "output", "loocv_evaluation_binary.csv")
        if os.path.exists(binary_file):
            try:
                os.remove(binary_file)
                logger.debug(f"Removed old binary evaluation file: {binary_file}")
            except Exception as e:
                logger.warning(f"Could not remove old binary evaluation file: {str(e)}")
    
    if problem_type in ["multiclass", "both"]:
        multiclass_file = os.path.join(application_path, "output", "loocv_evaluation_multiclass.csv")
        if os.path.exists(multiclass_file):
            try:
                os.remove(multiclass_file)
                logger.debug(f"Removed old multiclass evaluation file: {multiclass_file}")
            except Exception as e:
                logger.warning(f"Could not remove old multiclass evaluation file: {str(e)}")
    
    if problem_type == "both":
        combined_file = os.path.join(application_path, "output", "loocv_evaluation_combined.csv")
        if os.path.exists(combined_file):
            try:
                os.remove(combined_file)
                logger.debug(f"Removed old combined evaluation file: {combined_file}")
            except Exception as e:
                logger.warning(f"Could not remove old combined evaluation file: {str(e)}")
    
    results = []
    
    if problem_type in ["binary", "both"]:
        df_binary = loocv_evaluate_recommendations(problem_type="binary")
        if not df_binary.empty:
            results.append(df_binary)
    
    if problem_type in ["multiclass", "both"]:
        df_multiclass = loocv_evaluate_recommendations(problem_type="multiclass")
        if not df_multiclass.empty:
            results.append(df_multiclass)
    
    if not results:
        logger.warning("No evaluation results generated")
        return pd.DataFrame()
    
    # Handle combination if both were run
    if len(results) == 2:
        df_binary, df_multiclass = results
        df_binary["problem_type"] = "binary"
        df_multiclass["problem_type"] = "multiclass"
        df_combined = pd.concat([df_binary, df_multiclass], ignore_index=True)
        
        # Save combined results
        output_path = os.path.join(application_path, "output", "loocv_evaluation_combined.csv")
        df_combined.to_csv(output_path, sep=",", index=False)
        logger.info(f"Combined results saved to: {output_path}")
        
        return df_combined
    else:
        return results[0]


if __name__ == "__main__":
    # Run evaluation when executed directly
    # Options: "binary", "multiclass", "both" (default)
    loocv_evaluate_all(problem_type="multiclass")
