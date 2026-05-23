import os
import sys
import time
import datetime
import re
import logging
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
from sklearn.preprocessing import LabelEncoder
from imblearn.metrics import geometric_mean_score
import traceback
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration constants
CV_N_SPLITS = 10
CV_N_REPEATS = 3
RANDOM_STATE = 42
MIN_SAMPLES_FOR_SMOTE = 6  # Minimum samples needed per class for SMOTE to work
TOP_RECOMMENDATIONS = 3


def validate_dataset(X, y, balancing_technique_name):
    """
    Validate dataset for compatibility with resampling techniques.
    """
    # Check minimum samples
    if len(X) < 2:
        return False,
    
    # For SMOTE-based techniques, check if each class has enough neighbors
    smote_techniques = {'SMOTE', 'BorderlineSMOTE', 'KMeansSMOTE', 'SVMSMOTE'}
    if balancing_technique_name in smote_techniques:
        unique, counts = np.unique(y, return_counts=True)
        min_samples = counts.min()
        if min_samples < MIN_SAMPLES_FOR_SMOTE:
            return False, f"Class with {min_samples} samples is too small for {balancing_technique_name} (requires at least {MIN_SAMPLES_FOR_SMOTE})"
    
    return True, None


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
        
        # array_balancing = [
        #     "(no pre processing)", 
        #     "ClusterCentroids", "CondensedNearestNeighbour", "EditedNearestNeighbours", "RepeatedEditedNearestNeighbours", "AllKNN", "InstanceHardnessThreshold", "NearMiss", "NeighbourhoodCleaningRule", "OneSidedSelection", "RandomUnderSampler", "TomekLinks",
        #     "RandomOverSampler", "SMOTE", "ADASYN", "BorderlineSMOTE", "KMeansSMOTE", "SVMSMOTE",
        #     "SMOTEENN", "SMOTETomek"
        # ]
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


#  TEST VERSION
def execute_ml_test(dataset_location, id_openml):
    """Run the workflow without writing knowledge-base outputs."""
    
    try:
        start_time = time.time()
        
        if dataset_location:
            df, dataset_name = read_file(dataset_location)
        elif id_openml:
            df, dataset_name = read_file_openml(id_openml)
        else:
            return False
        
        X, y, df_characteristics = features_labels(df, dataset_name)
        problem_type = get_problem_type(y)
        
        logger.info("features_labels completed!")
        
        #  TEST VERSION
        
        array_balancing = ["(no pre processing)"]
        resultsList = []
        for balancing in array_balancing:
            try:
                balancing_technique = pre_processing(balancing)
                if balancing_technique is None:
                    logger.warning(f"pre_processing returned None for '{balancing}'")
                    continue
                resultsList += classify_evaluate(X, y, balancing, balancing_technique, dataset_name, problem_type)
            except (ValueError, KeyError) as e:
                logger.error(f"Error with balancing technique '{balancing}': {e}")
            except Exception as e:
                logger.error(f"Unexpected error with balancing technique '{balancing}': {e}", exc_info=True)
        
        #  TEST VERSION
        
        finish_time = (round(time.time() - start_time,3))

        if not resultsList:
            logger.warning("No valid model result was produced for this dataset.")
            return False
        
        best_result = find_best_result(resultsList)

        current_value = calculate_result_score(best_result)
        elapsed_time = str(datetime.timedelta(seconds=round(finish_time,0)))
        
        logger.info(f"Best Final Score: {current_value}, Elapsed Time: {elapsed_time}")
        
        #  TEST VERSION
        
        return dataset_name
    
    except Exception as e:
        logger.error(f"execute_ml_test failed: {str(e)}", exc_info=True)
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



def build_classifiers(problem_type, n_classes):
    """Build the classifier list for the detected problem type."""
    classifiers = [
        #LogisticRegression(random_state=RANDOM_STATE, max_iter=10000, class_weight='balanced'),
        #GaussianNB(),
        #SVC(random_state=RANDOM_STATE, class_weight='balanced', probability=True),
        #KNeighborsClassifier(),
        RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1),
        #ExtraTreesClassifier(random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1),
        AdaBoostClassifier(random_state=RANDOM_STATE),
        BaggingClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        EasyEnsembleClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        RUSBoostClassifier(random_state=RANDOM_STATE),
        BalancedBaggingClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        BalancedRandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
    ]

    #if problem_type == "multiclass":
    #    classifiers.extend([
    #        LGBMClassifier(
    #            random_state=42,
    #            objective='multiclass',
    #            num_class=n_classes,
    #            class_weight='balanced',
    #            force_col_wise=True,
    #            n_jobs=-1,
    #        ),
    #        XGBClassifier(
    #            random_state=42,
    #            use_label_encoder=False,
    #            objective='multi:softprob',
    #            num_class=n_classes,
    #            eval_metric='mlogloss',
    #            n_jobs=-1,
    #        ),
    #    ])
    #else:
    #    classifiers.extend([
    #        LGBMClassifier(
    #            random_state=42,
    #            objective='binary',
    #            class_weight='balanced',
    #            force_col_wise=True,
    #            n_jobs=-1,
    #        ),
    #        XGBClassifier(
    #            random_state=42,
    #            use_label_encoder=False,
    #            objective='binary:logistic',
    #            eval_metric='logloss',
    #            n_jobs=-1,
    #        ),
    #    ])

    return classifiers

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



# Balancing techniques mapping for efficient lookup
_BALANCING_TECHNIQUES = {
    "ClusterCentroids": lambda: ClusterCentroids(random_state=RANDOM_STATE),
    "CondensedNearestNeighbour": lambda: CondensedNearestNeighbour(random_state=RANDOM_STATE, n_jobs=-1),
    "EditedNearestNeighbours": lambda: EditedNearestNeighbours(n_jobs=-1),
    "RepeatedEditedNearestNeighbours": lambda: RepeatedEditedNearestNeighbours(n_jobs=-1),
    "AllKNN": lambda: AllKNN(n_jobs=-1),
    "InstanceHardnessThreshold": lambda: InstanceHardnessThreshold(random_state=RANDOM_STATE, n_jobs=-1),
    "NearMiss": lambda: NearMiss(n_jobs=-1),
    "NeighbourhoodCleaningRule": lambda: NeighbourhoodCleaningRule(n_jobs=-1),
    "OneSidedSelection": lambda: OneSidedSelection(random_state=RANDOM_STATE, n_jobs=-1),
    "RandomUnderSampler": lambda: RandomUnderSampler(random_state=RANDOM_STATE),
    "TomekLinks": lambda: TomekLinks(n_jobs=-1),
    "RandomOverSampler": lambda: RandomOverSampler(random_state=RANDOM_STATE),
    "SMOTE": lambda: SMOTE(random_state=RANDOM_STATE),
    "ADASYN": lambda: ADASYN(random_state=RANDOM_STATE),
    "BorderlineSMOTE": lambda: BorderlineSMOTE(random_state=RANDOM_STATE, n_jobs=-1),
    "KMeansSMOTE": lambda: KMeansSMOTE(random_state=RANDOM_STATE, n_jobs=-1),
    "SVMSMOTE": lambda: SVMSMOTE(random_state=RANDOM_STATE),
    "SMOTEENN": lambda: SMOTEENN(random_state=RANDOM_STATE, n_jobs=-1),
    "SMOTETomek": lambda: SMOTETomek(random_state=RANDOM_STATE, n_jobs=-1),
}


def pre_processing(balancing):
    """Create the configured resampling strategy for a given name."""
    if balancing not in _BALANCING_TECHNIQUES:
        raise ValueError(f"Unknown balancing technique: '{balancing}'. Available options: {', '.join(sorted(_BALANCING_TECHNIQUES.keys()))}")
    return _BALANCING_TECHNIQUES[balancing]()



# initial:  1 + 19  balancing techniques and    11  classification algorithms   = 220   combinations
# second:   1 + 14  balancing techniques and    8   classification algorithms   = 120   combinations
# third:    12      balancing techniques and    6   classification algorithms   = 72    combinations
# fourth:   7       balancing techniques and    4   classification algorithms   = 28    combinations
# fifth:    5       balancing techniques and    3   classification algorithms   = 15    combinations
# final:    4       balancing techniques and    3   classification algorithms   = 12    combinations
def classify_evaluate(X, y, balancing, balancing_technique, dataset_name, problem_type):
    """
    Evaluate each classifier with the selected resampling strategy.
    """

    n_classes = get_target_class_count(y)
    array_classifiers = build_classifiers(problem_type, n_classes)
    
    resultsList = []
    
    for classifier in array_classifiers:
        start_time = time.time()
        logger.info(f"Evaluating: balancing={balancing}, classifier={classifier.__class__.__name__}")
        
        model = make_pipeline(
            balancing_technique,
            classifier
        )
        
        cv = RepeatedStratifiedKFold(n_splits=CV_N_SPLITS, n_repeats=CV_N_REPEATS, random_state=RANDOM_STATE)
        
        scoring = get_scoring(problem_type)
        
        # Convert categorical labels to integer scalars
        #le = LabelEncoder()
        #y_encoded = le.fit_transform(y.values.ravel())
        
        #scores = cross_validate(model, X, y_encoded, scoring=scoring, cv=cv, n_jobs=-1) #, return_train_score=True
        
        scores = cross_validate(model, X, y.ravel() if hasattr(y, 'ravel') else y, scoring=scoring, cv=cv, n_jobs=-1)
        
        finish_time = round(time.time() - start_time,3)
        
        balanced_accuracy = round(np.mean(scores['test_balanced_accuracy']),3)
        f1_score = round(np.mean(scores['test_f1']),3)
        roc_auc_score = round(np.mean(scores['test_roc_auc']),3)
        g_mean_score = round(np.mean(scores['test_g_mean']),3)
        cohen_kappa = round(np.mean(scores['test_cohen_kappa']),3)
        
        balanced_accuracy_std = round(np.std(scores['test_balanced_accuracy']),3)
        f1_score_std = round(np.std(scores['test_f1']),3)
        roc_auc_score_std = round(np.std(scores['test_roc_auc']),3)
        g_mean_score_std = round(np.std(scores['test_g_mean']),3)
        cohen_kappa_std = round(np.std(scores['test_cohen_kappa']),3)

        multiclass_precision_macro = None
        multiclass_precision_macro_std = None
        multiclass_recall_macro = None
        multiclass_recall_macro_std = None
        multiclass_f1_weighted = None
        multiclass_f1_weighted_std = None
        multiclass_matthews_corrcoef = None
        multiclass_matthews_corrcoef_std = None

        if problem_type == "multiclass":
            (
                multiclass_precision_macro,
                multiclass_precision_macro_std,
                multiclass_recall_macro,
                multiclass_recall_macro_std,
                multiclass_f1_weighted,
                multiclass_f1_weighted_std,
                multiclass_matthews_corrcoef,
                multiclass_matthews_corrcoef_std,
            ) = get_multiclass_metrics_from_scores(scores)

        r1 = Results(dataset_name, balancing, classifier.__class__.__name__, finish_time, balanced_accuracy, balanced_accuracy_std, f1_score, f1_score_std, roc_auc_score, roc_auc_score_std, g_mean_score, g_mean_score_std, cohen_kappa, cohen_kappa_std, problem_type, multiclass_precision_macro, multiclass_precision_macro_std, multiclass_recall_macro, multiclass_recall_macro_std, multiclass_f1_weighted, multiclass_f1_weighted_std, multiclass_matthews_corrcoef, multiclass_matthews_corrcoef_std)
        resultsList.append(r1)
        
    return resultsList

def get_scoring(problem_type):
    """Build the scoring dictionary for binary or multiclass evaluation."""
    if problem_type == "multiclass":
        scoring = {
            'balanced_accuracy': 'balanced_accuracy',
            'f1': 'f1_macro',
            'roc_auc': 'roc_auc_ovr',
            'g_mean': make_scorer(geometric_mean_score, average='multiclass', greater_is_better=True),
            'cohen_kappa': make_scorer(cohen_kappa_score, greater_is_better=True),
        }
        scoring.update(get_multiclass_scoring())
        return scoring

    return {
        'balanced_accuracy': 'balanced_accuracy',
        'f1': 'f1',
        'roc_auc': 'roc_auc',
        'g_mean': make_scorer(geometric_mean_score, average='binary', greater_is_better=True),
        'cohen_kappa': make_scorer(cohen_kappa_score, greater_is_better=True),
    }


def get_multiclass_scoring():
    """Append additional scorers that only make sense for multiclass data."""
    return {
        'precision_macro': make_scorer(precision_score, average='macro', zero_division=0),
        'recall_macro': make_scorer(recall_score, average='macro', zero_division=0),
        'f1_weighted': 'f1_weighted',
        'matthews_corrcoef': make_scorer(matthews_corrcoef),
    }

def get_multiclass_metrics_from_scores(scores):
    """Summarize multiclass-only metrics from cross-validation scores."""
    precision_macro = round(np.mean(scores['test_precision_macro']), 3)
    precision_macro_std = round(np.std(scores['test_precision_macro']), 3)

    recall_macro = round(np.mean(scores['test_recall_macro']), 3)
    recall_macro_std = round(np.std(scores['test_recall_macro']), 3)

    f1_weighted = round(np.mean(scores['test_f1_weighted']), 3)
    f1_weighted_std = round(np.std(scores['test_f1_weighted']), 3)

    matthews_corrcoef = round(np.mean(scores['test_matthews_corrcoef']), 3)
    matthews_corrcoef_std = round(np.std(scores['test_matthews_corrcoef']), 3)

    return (
        precision_macro,
        precision_macro_std,
        recall_macro,
        recall_macro_std,
        f1_weighted,
        f1_weighted_std,
        matthews_corrcoef,
        matthews_corrcoef_std,
    )

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


#writes if best
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



#only writes at first time 
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


#by Euclidean Distance
def get_best_results_by_characteristics(dataset_name, problem_type):
    """
    Find the most similar past datasets and reuse their best pipelines.
    
    Uses Euclidean distance on normalized meta-features to rank similar datasets
    and returns the top {TOP_RECOMMENDATIONS} recommendations.
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
        current_features_norm = [0.0] * len(current_features)  # All values are identical, normalize to 0
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
            comparison_features_norm = [0.0] * len(comparison_features)  # All values are identical, normalize to 0
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
    
    Combines multiple metrics into a single comparable value using entropy-based
    weighting, giving more weight to more discriminative metrics. Handles edge
    cases like NaN values, negative numbers, and zero-sum scenarios.
    
    Args:
        metrics: List of metric values to combine
    
    Returns:
        Weighted composite score (float)
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
    
    Uses the same entropy_weighted_score logic to ensure consistency
    between newly computed scores and historical KB entries.
    """
    """Compute a comparable score from a stored KB row."""
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



def get_results_columns():
    """Return the stored-results column layout."""
    return [
        "dataset",
        "pre processing",
        "algorithm",
        "time",
        "balanced accuracy",
        "balanced accuracy std",
        "f1 score",
        "f1 score std",
        "roc auc",
        "roc auc std",
        "geometric mean",
        "geometric mean std",
        "cohen kappa",
        "cohen kappa std",
        "multiclass precision macro",
        "multiclass precision macro std",
        "multiclass recall macro",
        "multiclass recall macro std",
        "multiclass f1 weighted",
        "multiclass f1 weighted std",
        "multiclass matthews corrcoef",
        "multiclass matthews corrcoef std",
        "total elapsed time",
    ]


def get_full_results_columns():
    """Return the full-results column layout."""
    return [
        "dataset",
        "pre processing",
        "algorithm",
        "time",
        "balanced accuracy",
        "balanced accuracy std",
        "f1 score",
        "f1 score std",
        "roc auc",
        "roc auc std",
        "geometric mean",
        "geometric mean std",
        "cohen kappa",
        "cohen kappa std",
        "multiclass precision macro",
        "multiclass precision macro std",
        "multiclass recall macro",
        "multiclass recall macro std",
        "multiclass f1 weighted",
        "multiclass f1 weighted std",
        "multiclass matthews corrcoef",
        "multiclass matthews corrcoef std",
        "final score",
    ]

class Results(object):
    """Container for one classifier evaluation result."""
    def __init__(self, dataset_name, balancing, algorithm, time, balanced_accuracy, balanced_accuracy_std, f1_score, f1_score_std, roc_auc_score, roc_auc_score_std, g_mean_score, g_mean_score_std, cohen_kappa_score, cohen_kappa_score_std, problem_type, multiclass_precision_macro=None, multiclass_precision_macro_std=None, multiclass_recall_macro=None, multiclass_recall_macro_std=None, multiclass_f1_weighted=None, multiclass_f1_weighted_std=None, multiclass_matthews_corrcoef=None, multiclass_matthews_corrcoef_std=None):
        """Store all metrics and metadata for a single run."""
        self.dataset_name = dataset_name
        self.balancing = balancing
        self.algorithm = algorithm
        self.time = time
        self.balanced_accuracy = balanced_accuracy
        self.balanced_accuracy_std = balanced_accuracy_std
        self.f1_score = f1_score
        self.f1_score_std = f1_score_std
        self.roc_auc_score = roc_auc_score
        self.roc_auc_score_std = roc_auc_score_std
        self.g_mean_score = g_mean_score
        self.g_mean_score_std = g_mean_score_std
        self.cohen_kappa_score = cohen_kappa_score
        self.cohen_kappa_score_std = cohen_kappa_score_std
        self.problem_type = problem_type
        self.multiclass_precision_macro = multiclass_precision_macro
        self.multiclass_precision_macro_std = multiclass_precision_macro_std
        self.multiclass_recall_macro = multiclass_recall_macro
        self.multiclass_recall_macro_std = multiclass_recall_macro_std
        self.multiclass_f1_weighted = multiclass_f1_weighted
        self.multiclass_f1_weighted_std = multiclass_f1_weighted_std
        self.multiclass_matthews_corrcoef = multiclass_matthews_corrcoef
        self.multiclass_matthews_corrcoef_std = multiclass_matthews_corrcoef_std

def mcTest(dataset_name):
    """Run execute_ml_test for a dataset in input/multiclass."""
    dataset_path = resolve_multiclass_dataset_path(dataset_name)
    #return execute_byCharacteristics(dataset_path, None)
    #return execute_ml_test(dataset_path, None)
    return execute_ml(dataset_path, None)


def run_execute_ml_for_all_multiclass_datasets():
    """Run execute_ml for every CSV dataset found in input/multiclass."""
    datasets_dir = os.path.join(application_path, "input", "multiclass")

    if not os.path.isdir(datasets_dir):
        raise FileNotFoundError(f"Directory not found: {datasets_dir}")

    dataset_files = sorted([
        file_name for file_name in os.listdir(datasets_dir)
        if file_name.lower().endswith(".csv") and os.path.isfile(os.path.join(datasets_dir, file_name))
    ])

    if not dataset_files:
        print("No CSV datasets found in input/multiclass.")
        return {
            "processed": 0,
            "success": [],
            "failed": [],
            "skipped": [],
        }

    kb_results_path = get_kb_file_path("kb_results", "multiclass")
    processed_datasets = set()
    if os.path.exists(kb_results_path):
        df_kb_results = pd.read_csv(kb_results_path, sep=",")
        if "dataset" in df_kb_results.columns:
            processed_datasets = set(df_kb_results["dataset"].dropna().astype(str).tolist())

    success = []
    failed = []
    skipped = []

    print(f"Found {len(dataset_files)} multiclass datasets.")

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


if __name__ == "__main__":
    #run_execute_ml_for_all_multiclass_datasets()
    result = mcTest("car_subset.csv")