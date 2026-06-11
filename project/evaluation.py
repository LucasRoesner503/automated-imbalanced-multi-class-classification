import os
import time
import logging
import pandas as pd
import numpy as np
from scipy import stats

from config import application_path, get_full_results_columns
from data import (load_kb_dataframe)
from ml import get_best_results_by_characteristics

logger = logging.getLogger(__name__)


def get_metric_cols(problem_type):
    if problem_type == "multiclass":
        return ["multiclass precision macro", "multiclass recall macro", "multiclass f1 weighted"]
    else:
        return ["balanced accuracy", "f1 score", "roc auc"]


def get_filtered_kb_data(problem_type, exclude_dataset, df_char_cache=None, df_results_cache=None):
    """
    Get KB data with a dataset excluded.
    """
    if df_char_cache is None:
        df_char = load_kb_dataframe("kb_characteristics", problem_type)
    else:
        df_char = df_char_cache

    if df_results_cache is None:
        df_results = load_kb_dataframe("kb_full_results", problem_type, columns=get_full_results_columns())
    else:
        df_results = df_results_cache

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

    for col in get_metric_cols(problem_type):
        if col in recommended_row.index and col in best_row.index:
            rec_val = recommended_row[col]
            best_val = best_row[col]
            if pd.notna(rec_val) and pd.notna(best_val):
                deltas[f"delta_{col.replace(' ', '_')}"] = round(float(rec_val - best_val), 4)
            else:
                deltas[f"delta_{col.replace(' ', '_')}"] = np.nan

    return deltas


def loocv_evaluate_recommendations(problem_type="multiclass", output_suffix="", n_neighbors=None, feature_reduction="pca"):
    """
    Perform leave-one-out cross-validation on recommendation system.

    output_suffix is appended to the output file name, e.g. "_knn" writes
    loocv_evaluation_multiclass_knn.csv instead of loocv_evaluation_multiclass.csv.
    n_neighbors is forwarded to the recommender (1 = single nearest neighbor).
    feature_reduction is forwarded to the recommender ("pca" or "lasso").
    """

    logger.info(f"Starting improved LOOCV evaluation for {problem_type} datasets...")

    output_path = os.path.join(application_path, "output", f"loocv_evaluation_{problem_type}{output_suffix}.csv")
    if os.path.exists(output_path):
        try:
            os.remove(output_path)
            logger.debug(f"Removed old evaluation file: {output_path}")
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

            df_char_filtered, df_results_filtered, dataset_info = get_filtered_kb_data(
                problem_type, dataset_name,
                df_char_cache=df_characteristics_original,
                df_results_cache=df_full_results_original
            )

            if dataset_info['results'].empty:
                logger.warning(f"No results found for {dataset_name}")
                continue

            actual_results = dataset_info['results']

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

            dataset_char = dataset_info['char']
            if dataset_char.empty:
                logger.warning(f"Dataset {dataset_name} not found in characteristics")
                continue

            try:
                t0 = time.perf_counter()
                recommendations_df = get_best_results_by_characteristics(
                    dataset_name, problem_type,
                    current_features_df=dataset_char,
                    n_neighbors=n_neighbors,
                    feature_reduction=feature_reduction
                )
                recommendation_time = round(time.perf_counter() - t0, 4)

                if recommendations_df is not None and not recommendations_df.empty:
                    recommended_preproc = recommendations_df.iloc[0]["pre processing"]
                    recommended_algorithm = recommendations_df.iloc[0]["algorithm"]
                    logger.debug(f"Top recommendation for {dataset_name}: {recommended_preproc} + {recommended_algorithm}")
                else:
                    recommended_preproc = "NO_RECOMMENDATION"
                    recommended_algorithm = "NO_RECOMMENDATION"
                    logger.debug(f"No recommendations found for {dataset_name}")

            except Exception as e:
                recommendation_time = round(time.perf_counter() - t0, 4)
                logger.warning(f"Error getting recommendations: {str(e)}", exc_info=True)
                recommended_preproc = "NO_RECOMMENDATION"
                recommended_algorithm = "NO_RECOMMENDATION"

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
                rank = (sorted_scores >= recommended_score).sum()
                percentile_rank = calculate_rank_percentile(recommended_score, sorted_scores)

                is_top_1 = (rank == 1)
                is_top_3 = (rank <= 3)
                is_top_5 = (rank <= 5)

                metric_deltas = compute_metric_deltas(recommended_row, best_result, problem_type)

            delta_score = recommended_score - best_score if not np.isnan(recommended_score) else np.nan

            # Best of the top-3 recommendations: evaluate every returned
            # recommendation and keep the one scoring highest on the
            # held-out dataset's own grid
            best3_preproc = "NO_RECOMMENDATION"
            best3_algorithm = "NO_RECOMMENDATION"
            best3_score = np.nan
            if recommendations_df is not None and not recommendations_df.empty:
                for _, rec in recommendations_df.head(3).iterrows():
                    match = actual_results.loc[
                        (actual_results["pre processing"] == rec["pre processing"]) &
                        (actual_results["algorithm"] == rec["algorithm"])
                    ]
                    if match.empty:
                        continue
                    match_score = match.iloc[0]["final score"]
                    if pd.isna(match_score):
                        continue
                    if np.isnan(best3_score) or match_score > best3_score:
                        best3_score = match_score
                        best3_preproc = rec["pre processing"]
                        best3_algorithm = rec["algorithm"]

            if np.isnan(best3_score):
                best3_delta = np.nan
                best3_rank = np.nan
                best3_percentile = np.nan
                best3_is_top_1 = False
                best3_is_top_3 = False
                best3_is_top_5 = False
            else:
                best3_delta = best3_score - best_score
                sorted_scores_b3 = actual_results["final score"].sort_values(ascending=False).values
                best3_rank = (sorted_scores_b3 >= best3_score).sum()
                best3_percentile = calculate_rank_percentile(best3_score, sorted_scores_b3)
                best3_is_top_1 = (best3_rank == 1)
                best3_is_top_3 = (best3_rank <= 3)
                best3_is_top_5 = (best3_rank <= 5)

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
                "rank": rank,
                "percentile_rank": percentile_rank,
                "is_top_1": is_top_1,
                "is_top_3": is_top_3,
                "is_top_5": is_top_5,
                "best_top3_preproc": best3_preproc,
                "best_top3_algorithm": best3_algorithm,
                "best_top3_score": round(best3_score, 4) if not np.isnan(best3_score) else np.nan,
                "best_top3_delta": round(best3_delta, 4) if not np.isnan(best3_delta) else np.nan,
                "best_top3_rank": best3_rank,
                "best_top3_percentile": best3_percentile,
                "best_top3_is_top_1": best3_is_top_1,
                "best_top3_is_top_3": best3_is_top_3,
                "best_top3_is_top_5": best3_is_top_5,
                "recommendation_time_s": recommendation_time,
            }

            result_row.update(metric_deltas)

            for metric in get_metric_cols(problem_type):
                if metric in best_result.index:
                    result_row[f"best_{metric.replace(' ', '_')}"] = round(best_result[metric], 4) if pd.notna(best_result[metric]) else np.nan
                if not recommended_results.empty and metric in recommended_results.columns:
                    result_row[f"rec_{metric.replace(' ', '_')}"] = round(recommended_results.iloc[0][metric], 4) if pd.notna(recommended_results.iloc[0][metric]) else np.nan

            results.append(result_row)

        except Exception as e:
            logger.error(f"Error evaluating {dataset_name}: {str(e)}", exc_info=True)

    df_results = pd.DataFrame(results)

    if df_results.empty:
        logger.warning("No results generated from LOOCV evaluation")
        return df_results

    total_datasets = len(df_results)
    top_1_count = df_results["is_top_1"].sum()
    top_3_count = df_results["is_top_3"].sum()
    top_5_count = df_results["is_top_5"].sum()
    avg_rank = df_results["rank"].mean()
    avg_percentile = df_results["percentile_rank"].mean()
    avg_delta = df_results["delta_score"].mean()
    std_delta = df_results["delta_score"].std()
    median_delta = df_results["delta_score"].median()

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
    logger.info(f"  Best of top-3 recommendations:")
    logger.info(f"    Top 1 (optimal): {df_results['best_top3_is_top_1'].sum()} ({100*df_results['best_top3_is_top_1'].mean():.1f}%)")
    logger.info(f"    Top 3: {df_results['best_top3_is_top_3'].sum()} ({100*df_results['best_top3_is_top_3'].mean():.1f}%)")
    logger.info(f"    Top 5: {df_results['best_top3_is_top_5'].sum()} ({100*df_results['best_top3_is_top_5'].mean():.1f}%)")
    logger.info(f"    Average delta score: {df_results['best_top3_delta'].mean():.4f}")
    logger.info(f"    Median delta score: {df_results['best_top3_delta'].median():.4f}")

    df_results.to_csv(output_path, sep=",", index=False)
    logger.info(f"Results saved to: {output_path}")

    return df_results


def loocv_evaluate_all(problem_type=None, output_suffix=""):
    """
    Run LOOCV evaluation for specified problem types.
    """
    if problem_type is None:
        problem_type = "both"

    valid_types = ["binary", "multiclass", "both"]
    if problem_type not in valid_types:
        raise ValueError(f"problem_type must be one of {valid_types}, got '{problem_type}'")

    logger.info(f"Starting LOOCV evaluation for: {problem_type}")

    results = []

    if problem_type in ["binary", "both"]:
        df_binary = loocv_evaluate_recommendations(problem_type="binary", output_suffix=output_suffix)
        if not df_binary.empty:
            results.append(df_binary)

    if problem_type in ["multiclass", "both"]:
        df_multiclass = loocv_evaluate_recommendations(problem_type="multiclass", output_suffix=output_suffix)
        if not df_multiclass.empty:
            results.append(df_multiclass)

    if not results:
        logger.warning("No evaluation results generated")
        return pd.DataFrame()

    if len(results) == 2:
        df_binary["problem_type"] = "binary"
        df_multiclass["problem_type"] = "multiclass"
        df_combined = pd.concat([df_binary, df_multiclass], ignore_index=True)

        output_path = os.path.join(application_path, "output", "loocv_evaluation_combined.csv")
        df_combined.to_csv(output_path, sep=",", index=False)
        logger.info(f"Combined results saved to: {output_path}")

        return df_combined
    else:
        return results[0]


if __name__ == "__main__":
    # Options: "binary", "multiclass", "both" (default)
    loocv_evaluate_all(problem_type="multiclass")
