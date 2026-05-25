import os
import logging
import pandas as pd
import numpy as np
from scipy import stats

from config import logger, application_path, get_full_results_columns
from data import (load_kb_dataframe)
from ml import get_best_results_by_characteristics

logger = logging.getLogger(__name__)


def get_filtered_kb_data(problem_type, exclude_dataset, df_char_cache=None, df_results_cache=None):
    """
    Get KB data with a dataset excluded.
    """
    # Load or use cache
    if df_char_cache is None:
        df_char = load_kb_dataframe("kb_characteristics", problem_type)
    else:
        df_char = df_char_cache.copy()
    
    if df_results_cache is None:
        df_results = load_kb_dataframe("kb_full_results", problem_type, columns=get_full_results_columns())
    else:
        df_results = df_results_cache.copy()
    
    df_char_filtered = df_char.loc[df_char['dataset'] != exclude_dataset].copy()
    df_results_filtered = df_results.loc[df_results['dataset'] != exclude_dataset].copy()
    
    dataset_info = {
        'char': df_char.loc[df_char['dataset'] == exclude_dataset].copy(),
        'results': df_results.loc[df_results['dataset'] == exclude_dataset].copy()
    }
    
    return df_char_filtered, df_results_filtered, dataset_info


def calculate_rank_percentile(value, all_values):
    """
    Calculate percentile rank (0-100) for a value within a distribution.
    """
    valid_values = all_values[~np.isnan(all_values)]
    if len(valid_values) == 0 or np.isnan(value):
        return np.nan
    
    percentile = (valid_values < value).sum() / len(valid_values) * 100
    return round(percentile, 2)


def compute_metric_deltas(recommended_row, best_row, problem_type):
    """
    Compute deltas for all relevant metrics.
    """
    deltas = {}
    
    if problem_type == "multiclass":
        metric_cols = [
            "final score",
            "multiclass precision macro", 
            "multiclass recall macro",
            "multiclass f1 weighted"
        ]
    else:
        metric_cols = [
            "final score",
            "balanced accuracy",
            "f1 score",
            "roc auc"
        ]
    
    for col in metric_cols:
        if col in recommended_row.index and col in best_row.index:
            rec_val = recommended_row[col]
            best_val = best_row[col]
            if pd.notna(rec_val) and pd.notna(best_val):
                deltas[f"delta_{col.replace(' ', '_')}"] = round(float(rec_val - best_val), 4)
            else:
                deltas[f"delta_{col.replace(' ', '_')}"] = np.nan
    
    return deltas





def loocv_evaluate_recommendations(problem_type="multiclass"):
    """
    Perform leave-one-out cross-validation on recommendation system.
    """
    
    logger.info(f"Starting improved LOOCV evaluation for {problem_type} datasets...")
    
    # Remove old evaluation results
    output_file = os.path.join(application_path, "output", f"loocv_evaluation_{problem_type}.csv")
    if os.path.exists(output_file):
        try:
            os.remove(output_file)
            logger.debug(f"Removed old evaluation file: {output_file}")
        except Exception as e:
            logger.warning(f"Could not remove old evaluation file: {str(e)}")
    
    df_characteristics_original = load_kb_dataframe("kb_characteristics", problem_type)
    df_full_results_original = load_kb_dataframe("kb_full_results", problem_type, columns=get_full_results_columns())
    
    if df_characteristics_original.empty or df_full_results_original.empty:
        logger.error(f"KB data not available for problem_type: {problem_type}")
        return pd.DataFrame()
    
    results = []
    dataset_names = df_characteristics_original["dataset"].unique()
    total_datasets = len(dataset_names)
    
    logger.info(f"Found {total_datasets} datasets to evaluate")
    
    for idx, dataset_name in enumerate(dataset_names, 1):
        try:
            logger.info(f"[{idx}/{total_datasets}] LOOCV evaluating: {dataset_name}")
            
            # Get filtered KB
            df_char_filtered, df_results_filtered, dataset_info = get_filtered_kb_data(
                problem_type, dataset_name,
                df_char_cache=df_characteristics_original,
                df_results_cache=df_full_results_original
            )
            
            if dataset_info['results'].empty:
                logger.warning(f"No results found for {dataset_name}")
                continue
            
            actual_results = dataset_info['results']
            
            # Get the actual best combination (highest final score)
            final_scores = actual_results["final score"].values
            valid_scores = final_scores[~np.isnan(final_scores)]
            
            if len(valid_scores) == 0:
                logger.warning(f"No valid final scores for {dataset_name}")
                continue
            
            best_idx = actual_results["final score"].idxmax()
            best_result = actual_results.loc[best_idx]
            best_preproc = best_result["pre processing"]
            best_algorithm = best_result["algorithm"]
            best_score = best_result["final score"]
            
            # Get dataset characteristics
            dataset_char = dataset_info['char']
            if dataset_char.empty:
                logger.warning(f"Dataset {dataset_name} not found in characteristics")
                continue
            
            # Get recommendations from filtered KB
            try:
                recommendations_df = get_best_results_by_characteristics(
                    dataset_name, problem_type, 
                    current_features_df=dataset_char
                )
                
                if recommendations_df is not None and not recommendations_df.empty:
                    recommended_preproc = recommendations_df.iloc[0]["pre processing"]
                    recommended_algorithm = recommendations_df.iloc[0]["algorithm"]
                    logger.debug(f"Top recommendation for {dataset_name}: {recommended_preproc} + {recommended_algorithm}")
                else:
                    recommended_preproc = "NO_RECOMMENDATION"
                    recommended_algorithm = "NO_RECOMMENDATION"
                    logger.debug(f"No recommendations found for {dataset_name}")
                    
            except Exception as e:
                logger.warning(f"Error getting recommendations: {str(e)}", exc_info=True)
                recommended_preproc = "NO_RECOMMENDATION"
                recommended_algorithm = "NO_RECOMMENDATION"
            
            # Find recommended combination in actual results
            recommended_results = actual_results.loc[
                (actual_results["pre processing"] == recommended_preproc) & 
                (actual_results["algorithm"] == recommended_algorithm)
            ]
            
            if recommended_results.empty or recommended_preproc == "NO_RECOMMENDATION":
                recommended_score = np.nan
                rank = np.nan
                percentile_rank = np.nan
                is_top_1 = False
                is_top_3 = False
                is_top_5 = False
                metric_deltas = {}
            else:
                recommended_score = recommended_results.iloc[0]["final score"]
                recommended_row = recommended_results.iloc[0]
                
                sorted_scores = actual_results["final score"].sort_values(ascending=False).values
                rank_count = (sorted_scores >= recommended_score).sum()
                
                percentile_rank = calculate_rank_percentile(recommended_score, sorted_scores)
                
                rank = rank_count
                
                is_top_1 = (rank == 1)
                is_top_3 = (rank <= 3)
                is_top_5 = (rank <= 5)
                
                metric_deltas = compute_metric_deltas(recommended_row, best_result, problem_type)
            
            delta_score = recommended_score - best_score if not np.isnan(recommended_score) else np.nan
            
            # Build result row
            result_row = {
                "dataset": dataset_name,
                "num_combinations": len(actual_results), 
                "recommended_preproc": recommended_preproc,
                "recommended_algorithm": recommended_algorithm,
                "recommended_score": round(recommended_score, 4) if not np.isnan(recommended_score) else np.nan,
                "best_preproc": best_preproc,
                "best_algorithm": best_algorithm,
                "best_score": round(best_score, 4),
                "delta_score": round(delta_score, 4) if not np.isnan(delta_score) else np.nan,
                "rank": rank if not np.isnan(rank) else np.nan,
                "percentile_rank": percentile_rank,
                "is_top_1": is_top_1,
                "is_top_3": is_top_3,
                "is_top_5": is_top_5,
            }
            
            result_row.update(metric_deltas)
            
            if problem_type == "multiclass":
                for metric in ["multiclass precision macro", "multiclass recall macro", "multiclass f1 weighted"]:
                    if metric in best_result.index:
                        result_row[f"best_{metric.replace(' ', '_')}"] = round(best_result[metric], 4) if pd.notna(best_result[metric]) else np.nan
                    if metric in recommended_results.columns and not recommended_results.empty:
                        result_row[f"rec_{metric.replace(' ', '_')}"] = round(recommended_results.iloc[0][metric], 4) if pd.notna(recommended_results.iloc[0][metric]) else np.nan
            else:
                for metric in ["balanced accuracy", "f1 score", "roc auc"]:
                    if metric in best_result.index:
                        result_row[f"best_{metric.replace(' ', '_')}"] = round(best_result[metric], 4) if pd.notna(best_result[metric]) else np.nan
                    if metric in recommended_results.columns and not recommended_results.empty:
                        result_row[f"rec_{metric.replace(' ', '_')}"] = round(recommended_results.iloc[0][metric], 4) if pd.notna(recommended_results.iloc[0][metric]) else np.nan
            
            results.append(result_row)
            
        except Exception as e:
            logger.error(f"Error evaluating {dataset_name}: {str(e)}", exc_info=True)
    
    df_results = pd.DataFrame(results)
    
    if df_results.empty:
        logger.warning("No results generated from LOOCV evaluation")
        return df_results
    
    # Get total counts and averages for summary statistics
    total_datasets = len(df_results)
    top_1_count = df_results["is_top_1"].sum()
    top_3_count = df_results["is_top_3"].sum()
    top_5_count = df_results["is_top_5"].sum()
    avg_rank = df_results["rank"].mean()
    avg_percentile = df_results["percentile_rank"].mean()
    avg_delta = df_results["delta_score"].mean()
    std_delta = df_results["delta_score"].std()
    median_delta = df_results["delta_score"].median()
    
    # Compute 95% confidence interval for average delta
    delta_values = df_results["delta_score"].dropna()
    if len(delta_values) > 1:
        se = stats.sem(delta_values) 
        if pd.notna(se) and se > 0:
            try:
                ci_95 = stats.t.interval(0.95, len(delta_values)-1, loc=avg_delta, scale=se)
                logger.info(f"Average delta 95% CI: [{ci_95[0]:.4f}, {ci_95[1]:.4f}]")
            except Exception as e:
                logger.debug(f"Could not compute confidence interval: {str(e)}")
        else:
            logger.debug(f"Standard error is zero or invalid (se={se}), skipping CI computation")
    
    logger.info(f"\nLOOCV Evaluation Summary ({problem_type}):")
    logger.info(f"  Total datasets: {total_datasets}")
    logger.info(f"  Top 1 (optimal): {top_1_count} ({100*top_1_count/total_datasets:.1f}%)")
    logger.info(f"  Top 3: {top_3_count} ({100*top_3_count/total_datasets:.1f}%)")
    logger.info(f"  Top 5: {top_5_count} ({100*top_5_count/total_datasets:.1f}%)")
    logger.info(f"  Average rank: {avg_rank:.2f}")
    logger.info(f"  Average percentile rank: {avg_percentile:.2f}%")
    logger.info(f"  Average delta score: {avg_delta:.4f} ± {std_delta:.4f}")
    logger.info(f"  Median delta score: {median_delta:.4f}")
    
    output_path = os.path.join(application_path, "output", f"loocv_evaluation_{problem_type}.csv")
    df_results.to_csv(output_path, sep=",", index=False)
    logger.info(f"Results saved to: {output_path}")
    
    return df_results



def loocv_evaluate_all(problem_type=None):
    """
    Run LOOCV evaluation for specified problem types.
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
