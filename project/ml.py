import os
import time
import datetime
import logging
import pandas as pd
import numpy as np

from config import logger, TOP_RECOMMENDATIONS, get_results_columns, get_full_results_columns
from data import (
    read_file, read_file_openml, features_labels, validate_dataset,
    get_problem_type, resolve_binary_dataset_path, resolve_multiclass_dataset_path, 
    get_kb_file_path, load_kb_dataframe
)
from models import classify_evaluate, pre_processing

# Configure module logger
logger = logging.getLogger(__name__)


def find_best_result(resultsList):
    """
    Select the result with the highest composite score.
    
    Uses entropy_weighted_score to compute a single comparable metric
    from all evaluation metrics.
    """
    scores = []
    for result in resultsList:
        scores.append(calculate_result_score(result))

    best_score = max(scores)
    index = scores.index(best_score)
    best_result = resultsList[index]
    
    string_balancing = best_result.balancing
    
    logger.info(f"\nBest classifier: {best_result.algorithm} with {string_balancing}\n")
    
    return best_result


def calculate_result_score(result):
    """Compute the composite score used to compare results."""
    if result.problem_type == "multiclass":
        metrics = [
            result.multiclass_precision_macro,
            result.multiclass_recall_macro,
            result.multiclass_f1_weighted,
            result.multiclass_matthews_corrcoef,
        ]
    else:
        metrics = [
            result.balanced_accuracy,
            result.f1_score,
            result.roc_auc_score,
            result.g_mean_score,
            result.cohen_kappa_score,
        ]

    return entropy_weighted_score(metrics)


def entropy_weighted_score(metrics):
    """
    Compute an entropy-weighted composite score from a metric vector.
    """
    cleaned_metrics = [float(metric) for metric in metrics if pd.notna(metric)]
    if not cleaned_metrics:
        return np.nan

    metric_values = np.asarray(cleaned_metrics, dtype=float)
    if np.any(metric_values < 0):
        metric_values = metric_values - np.min(metric_values)

    total = float(np.sum(metric_values))
    if total <= 0:
        return round(float(np.mean(cleaned_metrics)), 3)

    shares = metric_values / total
    valid_shares = shares[shares > 0]
    if valid_shares.size == 0:
        return round(float(np.mean(cleaned_metrics)), 3)

    entropy_base = np.log(len(metric_values))
    if entropy_base <= 0:
        return round(float(np.mean(cleaned_metrics)), 3)

    entropy_contrib = np.zeros_like(shares)
    entropy_contrib[shares > 0] = -(shares[shares > 0] * np.log(shares[shares > 0])) / entropy_base

    weights = shares * entropy_contrib
    weight_total = float(np.sum(weights))
    if weight_total <= 0:
        weights = np.full(len(metric_values), 1.0 / len(metric_values))
    else:
        weights = weights / weight_total

    return round(float(np.dot(metric_values, weights)), 3)


def calculate_kb_row_score(row, problem_type):
    """
    Compute a comparable score from a stored KB row.
    """
    if problem_type == "multiclass":
        metrics = [
            row['multiclass precision macro'],
            row['multiclass recall macro'],
            row['multiclass f1 weighted'],
            row['multiclass matthews corrcoef'],
        ]
    else:
        metrics = [
            row['balanced accuracy'],
            row['f1 score'],
            row['roc auc'],
            row['geometric mean'],
            row['cohen kappa'],
        ]

    return entropy_weighted_score(metrics)


def get_kb_metric_payload(result):
    """Return metric values aligned with KB columns for binary or multiclass."""
    if result.problem_type == "multiclass":
        return {
            'balanced accuracy': np.nan,
            'balanced accuracy std': np.nan,
            'f1 score': np.nan,
            'f1 score std': np.nan,
            'roc auc': np.nan,
            'roc auc std': np.nan,
            'geometric mean': np.nan,
            'geometric mean std': np.nan,
            'cohen kappa': np.nan,
            'cohen kappa std': np.nan,
            'multiclass precision macro': result.multiclass_precision_macro,
            'multiclass precision macro std': result.multiclass_precision_macro_std,
            'multiclass recall macro': result.multiclass_recall_macro,
            'multiclass recall macro std': result.multiclass_recall_macro_std,
            'multiclass f1 weighted': result.multiclass_f1_weighted,
            'multiclass f1 weighted std': result.multiclass_f1_weighted_std,
            'multiclass matthews corrcoef': result.multiclass_matthews_corrcoef,
            'multiclass matthews corrcoef std': result.multiclass_matthews_corrcoef_std,
        }

    return {
        'balanced accuracy': result.balanced_accuracy,
        'balanced accuracy std': result.balanced_accuracy_std,
        'f1 score': result.f1_score,
        'f1 score std': result.f1_score_std,
        'roc auc': result.roc_auc_score,
        'roc auc std': result.roc_auc_score_std,
        'geometric mean': result.g_mean_score,
        'geometric mean std': result.g_mean_score_std,
        'cohen kappa': result.cohen_kappa_score,
        'cohen kappa std': result.cohen_kappa_score_std,
        'multiclass precision macro': np.nan,
        'multiclass precision macro std': np.nan,
        'multiclass recall macro': np.nan,
        'multiclass recall macro std': np.nan,
        'multiclass f1 weighted': np.nan,
        'multiclass f1 weighted std': np.nan,
        'multiclass matthews corrcoef': np.nan,
        'multiclass matthews corrcoef std': np.nan,
    }


def build_kb_row_values(result, metric_payload, tail_value):
    """Build a row list aligned with KB results/full-results columns."""
    return [
        result.dataset_name,
        result.balancing,
        result.algorithm,
        result.time,
        metric_payload['balanced accuracy'],
        metric_payload['balanced accuracy std'],
        metric_payload['f1 score'],
        metric_payload['f1 score std'],
        metric_payload['roc auc'],
        metric_payload['roc auc std'],
        metric_payload['geometric mean'],
        metric_payload['geometric mean std'],
        metric_payload['cohen kappa'],
        metric_payload['cohen kappa std'],
        metric_payload['multiclass precision macro'],
        metric_payload['multiclass precision macro std'],
        metric_payload['multiclass recall macro'],
        metric_payload['multiclass recall macro std'],
        metric_payload['multiclass f1 weighted'],
        metric_payload['multiclass f1 weighted std'],
        metric_payload['multiclass matthews corrcoef'],
        metric_payload['multiclass matthews corrcoef std'],
        tail_value,
    ]


def write_characteristics(df_characteristics, best_result, result_updated, problem_type):
    """
    Persist dataset characteristics and the selected pipeline metadata.
    
    Updates or inserts rows in the KB based on whether
    the new result improves upon previously stored results.
    """
    if df_characteristics.empty:
        logger.error("df_characteristics is empty in write_characteristics")
        return False
    
    try:
    
        df_kb_c = load_kb_dataframe(
            "kb_characteristics",
            problem_type,
            columns=list(df_characteristics.columns) + ["pre processing", "algorithm"],
        )
        
        dataset_name = df_characteristics["dataset"].iloc[0]
        df_kb_c_without = df_kb_c.loc[df_kb_c["dataset"] != dataset_name].copy()
        df_kb_c_selected = df_kb_c.loc[df_kb_c["dataset"] == dataset_name].copy()
        
        df_characteristics = pd.concat([df_characteristics, df_kb_c_without], ignore_index=True)
        df_characteristics = df_characteristics.reset_index(drop=True)
        
        #execute_ml
        if best_result and best_result.balancing and best_result.algorithm:
            #row updated or new line
            if result_updated or df_kb_c_selected.empty:
                df_characteristics.at[0, 'pre processing'] = best_result.balancing
                df_characteristics.at[0, 'algorithm'] = best_result.algorithm
                
                logger.info("Characteristics written, row added or updated")
            
            #it was worse
            else:
                df_characteristics.at[0, 'pre processing'] = df_kb_c_selected["pre processing"].values[0]
                df_characteristics.at[0, 'algorithm'] = df_kb_c_selected["algorithm"].values[0]
                
                logger.info("Characteristics not written (previous result was better)")
        
        #execute_byCharacteristics
        else:
            #new row
            if df_kb_c_selected.empty:
                df_characteristics.at[0, 'pre processing'] = "?"
                df_characteristics.at[0, 'algorithm'] = "?"
            #remains value
            else:
                df_characteristics = df_kb_c
        
        df_characteristics.to_csv(get_kb_file_path("kb_characteristics", problem_type), sep=",", index=False)
        
    except Exception as e:
        logger.error(f"write_characteristics failed: {str(e)}", exc_info=True)
        return False

    return True   


def write_results(best_result, elapsed_time):
    """
    Persist the best overall result if it improves the stored record.
    
    Compares the new result's composite score with any existing result for the
    dataset and only updates if there's an improvement.
    """
    if not best_result:
        logger.error(f"write_results: invalid inputs - best_result={best_result}, elapsed_time={elapsed_time}")
        return False
    
    result_updated = False
    
    try:
        
        current_value = calculate_result_score(best_result)
        
        elapsed_time_str = str(datetime.timedelta(seconds=round(elapsed_time,0)))
        
        logger.info(f"Best Final Score: {current_value}, Elapsed Time: {elapsed_time_str}")
        
        df_kb_r = load_kb_dataframe("kb_results", best_result.problem_type, columns=get_results_columns())

        metric_payload = get_kb_metric_payload(best_result)
        
        df_kb_r2 = df_kb_r.loc[df_kb_r['dataset'] == best_result.dataset_name]
        
        if not df_kb_r2.empty :
            row = df_kb_r2.iloc[0]
            previous_value = calculate_kb_row_score(row, best_result.problem_type)
            
            if current_value > previous_value:
                
                index = df_kb_r2.index.values[0]
                df_kb_r.at[index, 'pre processing'] = best_result.balancing
                df_kb_r.at[index, 'algorithm'] = best_result.algorithm
                df_kb_r.at[index, 'time'] = best_result.time
                for column_name, column_value in metric_payload.items():
                    df_kb_r.at[index, column_name] = column_value
                df_kb_r.at[index, 'total elapsed time'] = elapsed_time_str
                
                df_kb_r.to_csv(get_kb_file_path("kb_results", best_result.problem_type), sep=",", index=False)
                
                result_updated = True
                
                logger.info("Results written, row updated")

            else:
                logger.info("Results not written (previous result was better)")
                
        else:
            df_kb_r.loc[len(df_kb_r.index)] = build_kb_row_values(best_result, metric_payload, elapsed_time_str)

            df_kb_r.to_csv(get_kb_file_path("kb_results", best_result.problem_type), sep=",", index=False)
            
            logger.info("Results written, row added")
        
    except Exception as e:
        logger.error(f"write_results failed: {str(e)}", exc_info=True)
        return False
    
    return result_updated


def write_full_results(resultsList, dataset_name):
    """
    Persist all evaluated combinations for a dataset the first time only.
    
    Prevents duplicate processing by checking if the dataset already exists
    in the knowledge base before writing.
    """
    if not resultsList or not dataset_name:
        logger.error(f"write_full_results: invalid inputs - resultsList={bool(resultsList)}, dataset_name={dataset_name}")
        return False
    
    try:
    
        problem_type = resultsList[0].problem_type
        df_kb_r = load_kb_dataframe("kb_full_results", problem_type, columns=get_full_results_columns())
        
        df_kb_r2 = df_kb_r.loc[df_kb_r['dataset'] == dataset_name]
        
        if df_kb_r2.empty :
        
            for result in resultsList:
                metric_payload = get_kb_metric_payload(result)
                df_kb_r.loc[len(df_kb_r.index)] = build_kb_row_values(
                    result,
                    metric_payload,
                    calculate_result_score(result)
                )

            df_kb_r.sort_values(by=['final score'], ascending=False, inplace=True)

            df_kb_r.to_csv(get_kb_file_path("kb_full_results", problem_type), sep=",", index=False)
            
            logger.info(f"Full Results written: {len(resultsList)} rows added")
        
        else:
            logger.info("Full Results not written (dataset already in KB)")
        
    except Exception as e:
        logger.error(f"write_full_results failed: {str(e)}", exc_info=True)
        return False
    
    return True


def get_best_results_by_characteristics(dataset_name, problem_type):
    """
    Find the most similar past datasets.
    """
    if not dataset_name:
        logger.error(f"get_best_results_by_characteristics: dataset_name is invalid: {dataset_name}")
        return False
    
    df_c = load_kb_dataframe("kb_characteristics", problem_type)
    if df_c.empty:
        logger.error(f"kb_characteristics is empty for problem_type: {problem_type}")
        return False

    df_c = df_c.dropna(axis=1)
    df_c = df_c.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
    
    # Get current dataset characteristics
    current_dataset_chars = df_c.loc[df_c['dataset'] == dataset_name]
    current_dataset_chars = current_dataset_chars.drop(['dataset', 'pre processing','algorithm'], axis=1)
    current_features = current_dataset_chars.values.tolist()[0]
    min_current, max_current = min(current_features), max(current_features)
    if min_current == max_current:
        current_features_norm = [0.0] * len(current_features)
    else:
        current_features_norm = [(float(i) - min_current) / (max_current - min_current) for i in current_features]

    df_c = df_c.loc[df_c['dataset'] != dataset_name]
    distances = []
    for index, row in df_c.iterrows():
        comparison_chars = row.to_frame()
        comparison_chars = comparison_chars.drop(['dataset', 'pre processing','algorithm'])
        comparison_features = comparison_chars.values.tolist()
        comparison_features = [x for xs in comparison_features for x in xs]
        min_comparison, max_comparison = min(comparison_features), max(comparison_features)
        if min_comparison == max_comparison:
            comparison_features_norm = [0.0] * len(comparison_features)
        else:
            comparison_features_norm = [(float(i) - min_comparison) / (max_comparison - min_comparison) for i in comparison_features]
        distances.append((row['dataset'], row['pre processing'], row['algorithm'], np.linalg.norm(np.array(current_features_norm) - np.array(comparison_features_norm))))
        
    df_dist = pd.DataFrame(distances, columns=["dataset", "pre processing", "algorithm","distance"])
    df_dist = df_dist.sort_values(by=['distance'])
    df_dist = df_dist.drop_duplicates(subset=['pre processing', 'algorithm'], keep='first')
    df_dist = df_dist.reset_index(drop=True)
    df_dist = df_dist.head(TOP_RECOMMENDATIONS)
    
    logger.info(f"Top {TOP_RECOMMENDATIONS} recommendations:\n{df_dist}")
    
    df_dist = df_dist[['pre processing', 'algorithm']]
    
    return df_dist


def display_final_results(df_dist):
    """Format the top recommendations as a display string."""
    df_dist.loc[-1] = ['Pre Processing', 'Algorithm']
    df_dist.index = df_dist.index + 1
    df_dist = df_dist.sort_index()
    df_dist.insert(loc=0, column='rank', value=['Rank',1,2,3])
    
    str_output = "Top performing combinations of Pre Processing Technique with a Classifier Algorithm\n\n"
    str_output += "\n".join("{:7} {:25} {:25}".format(x, y, z) for x, y, z in zip(df_dist['rank'], df_dist['pre processing'], df_dist['algorithm']))
    str_output += "\n"
    return str_output


def execute_ml(dataset_location, id_openml):
    """Run the full workflow and persist the best result."""
    
    try:
        if dataset_location:
            df, dataset_name = read_file(dataset_location)
        elif id_openml:
            df, dataset_name = read_file_openml(id_openml)
        else:
            return False
        
        start_time = time.time()
        
        X, y, df_characteristics = features_labels(df, dataset_name)
        problem_type = get_problem_type(y)
        
        array_balancing = [
            "RandomOverSampler", "SMOTE",
            "SMOTETomek", "ADASYN", "EditedNearestNeighbours",
            "RandomUnderSampler", "TomekLinks"
        ]
        
        resultsList = []
        i = 1
        for balancing in array_balancing:
            try:
                logger.info(f"Loading: {i} of {len(array_balancing)} - {balancing}")
                i += 1
                
                # Validate dataset compatibility with this technique
                is_valid, error_msg = validate_dataset(X, y, balancing)
                if not is_valid:
                    logger.warning(f"Skipping '{balancing}': {error_msg}")
                    continue
                
                balancing_technique = pre_processing(balancing)
                if balancing_technique is None:
                    logger.warning(f"pre_processing returned None for '{balancing}'")
                    continue
                resultsList += classify_evaluate(X, y, balancing, balancing_technique, dataset_name, problem_type)
            except (ValueError, KeyError) as e:
                logger.error(f"Error with balancing technique '{balancing}': {e}")
            except Exception as e:
                logger.error(f"Unexpected error with balancing technique '{balancing}': {e}", exc_info=True)
        
        finish_time = (round(time.time() - start_time,3))

        if not resultsList:
            print("No valid model result was produced for this dataset.")
            return False
        
        best_result = find_best_result(resultsList)
        
        result_updated = write_results(best_result, finish_time)
        
        write_full_results(resultsList, dataset_name)
        
        write_characteristics(df_characteristics, best_result, result_updated, problem_type)
        
        return dataset_name
    
    except Exception as e:
        logger.error(f"execute_ml failed: {str(e)}", exc_info=True)
        return False


def execute_byCharacteristics(dataset_location, id_openml):
    """Return the top pre-processing and classifier options by similarity."""
    try:
        if dataset_location:
            df, dataset_name = read_file(dataset_location)
        elif id_openml:
            df, dataset_name = read_file_openml(id_openml)
        else:
            return False
        
        X, y, df_characteristics = features_labels(df, dataset_name)
        problem_type = get_problem_type(y)
        
        write_characteristics(df_characteristics, None, False, problem_type)
        df_dist = get_best_results_by_characteristics(dataset_name, problem_type)
        str_output = display_final_results(df_dist)
        
        return str_output
        
    except Exception as e:
        logger.error(f"execute_byCharacteristics failed: {str(e)}", exc_info=True)
        return False


def run_execute_ml_for_all_datasets(is_binary=True):
    """Run execute_ml for every CSV dataset found in input/binary or input/multiclass."""
    from config import application_path
    
    problem_type = "binary" if is_binary else "multiclass"
    datasets_dir = os.path.join(application_path, "input", problem_type)

    if not os.path.isdir(datasets_dir):
        raise FileNotFoundError(f"Directory not found: {datasets_dir}")

    dataset_files = sorted([
        file_name for file_name in os.listdir(datasets_dir)
        if file_name.lower().endswith(".csv") and os.path.isfile(os.path.join(datasets_dir, file_name))
    ])

    if not dataset_files:
        logger.info(f"No CSV datasets found in input/{problem_type}.")
        return {
            "processed": 0,
            "success": [],
            "failed": [],
            "skipped": [],
        }

    kb_results_path = get_kb_file_path("kb_results", problem_type)
    processed_datasets = set()
    if os.path.exists(kb_results_path):
        df_kb_results = pd.read_csv(kb_results_path, sep=",")
        if "dataset" in df_kb_results.columns:
            processed_datasets = set(df_kb_results["dataset"].dropna().astype(str).tolist())

    success = []
    failed = []
    skipped = []

    logger.info(f"Found {len(dataset_files)} {problem_type} datasets.")

    for index, dataset_file in enumerate(dataset_files, start=1):
        if dataset_file in processed_datasets:
            logger.info(f"[{index}/{len(dataset_files)}] Skipping (already processed): {dataset_file}")
            skipped.append(dataset_file)
            continue

        dataset_path = os.path.join(datasets_dir, dataset_file)
        logger.info(f"[{index}/{len(dataset_files)}] Processing: {dataset_file}")

        result = execute_ml(dataset_path, None)
        if result:
            success.append(dataset_file)
        else:
            failed.append(dataset_file)

    logger.info("\nBatch run finished.")
    logger.info(f"Successful: {len(success)}, Failed: {len(failed)}, Skipped: {len(skipped)}")

    if failed:
        logger.warning(f"Failed datasets ({len(failed)}):")
        for dataset_file in failed:
            logger.warning(f"  - {dataset_file}")

    return {
        "processed": len(success) + len(failed),
        "success": success,
        "failed": failed,
        "skipped": skipped,
    }


def btTest(dataset_name):
    dataset_path = resolve_binary_dataset_path(dataset_name)
    return execute_ml(dataset_path, None)


def mcTest(dataset_name):
    dataset_path = resolve_multiclass_dataset_path(dataset_name)
    return execute_ml(dataset_path, None)
