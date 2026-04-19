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


def build_classifiers(problem_type, n_classes):
    """Build the classifier list for the detected problem type."""
    classifiers = []

    if problem_type == "multiclass":
        classifiers.append(
            LGBMClassifier(
                random_state=42,
                objective='multiclass',
                num_class=n_classes,
                class_weight='balanced',
                n_jobs=-1,
            )
        )
        classifiers.append(
            XGBClassifier(
                random_state=42,
                use_label_encoder=False,
                objective='multi:softprob',
                num_class=n_classes,
                eval_metric='mlogloss',
                n_jobs=-1,
            )
        )
    else:
        classifiers.append(
            LGBMClassifier(
                random_state=42,
                objective='binary',
                class_weight='balanced',
                n_jobs=-1,
            )
        )
        classifiers.append(
            XGBClassifier(
                random_state=42,
                use_label_encoder=False,
                objective='binary:logistic',
                eval_metric='logloss',
                n_jobs=-1,
            )
        )

    classifiers.append(GradientBoostingClassifier(random_state=42))

    return classifiers


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
    """Build additional scorers that only make sense for multiclass data."""
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


def calculate_result_score(result):
    """Compute the composite score used to compare results."""
    return round(np.mean([
        result.balanced_accuracy,
        result.f1_score,
        result.roc_auc_score,
        result.g_mean_score,
        result.cohen_kappa_score,
    ]), 3)


def get_kb_file_path(base_name, problem_type):
    """Build the knowledge-base CSV path for a problem type."""
    return os.path.join(application_path, "output", f"{base_name}_{problem_type}.csv")


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
        
        # array_balancing = ["(no pre processing)"]
        # array_balancing = [
        #     "(no pre processing)", 
        #     "ClusterCentroids", "CondensedNearestNeighbour", "EditedNearestNeighbours", "RepeatedEditedNearestNeighbours", "AllKNN", "InstanceHardnessThreshold", "NearMiss", "NeighbourhoodCleaningRule", "OneSidedSelection", "RandomUnderSampler", "TomekLinks",
        #     "RandomOverSampler", "SMOTE", "ADASYN", "BorderlineSMOTE", "KMeansSMOTE", "SVMSMOTE",
        #     "SMOTEENN", "SMOTETomek"
        # ]
        array_balancing = [
            "RandomOverSampler", "SMOTE", "SVMSMOTE",
            "SMOTETomek"
        ]
        
        resultsList = []
        i = 1
        for balancing in array_balancing:
            try:
                print("loading: ", i, " of ", len(array_balancing))
                i += 1
                balancing_technique = pre_processing(balancing) 
                resultsList += classify_evaluate(X, y, balancing, balancing_technique, dataset_name, problem_type)
            except Exception:
                traceback.print_exc()
        
        finish_time = (round(time.time() - start_time,3))
        
        best_result = find_best_result(resultsList)
        
        result_updated = write_results(best_result, finish_time)
        
        write_full_results(resultsList, dataset_name)
        
        write_characteristics(df_characteristics, best_result, result_updated, problem_type)
        
        return dataset_name
    
    except Exception:
        traceback.print_exc()
        return False



#  TEST VERSION - execute algorithms without pre processing nor writting to any KB file
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
        
        print("features_labels done!")
        
        #  TEST VERSION
        
        array_balancing = ["(no pre processing)"]
        resultsList = []
        for balancing in array_balancing:
            try:
                balancing_technique = pre_processing(balancing) 
                resultsList += classify_evaluate(X, y, balancing, balancing_technique, dataset_name, problem_type)
            except Exception:
                traceback.print_exc()
        
        #  TEST VERSION
        
        finish_time = (round(time.time() - start_time,3))
        
        best_result = find_best_result(resultsList)

        current_value = calculate_result_score(best_result)
        elapsed_time = str(datetime.timedelta(seconds=round(finish_time,0)))
        
        print("Best Final Score Obtained    :", current_value)
        print("Elapsed Time                 :", elapsed_time, "\n")
        
        #  TEST VERSION
        
        return dataset_name
    
    except Exception:
        traceback.print_exc()
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
        
    except Exception:
        traceback.print_exc()
        return False



def read_file(path):
    """Read a CSV dataset from disk and drop missing rows."""
    df = pd.read_csv(path)
    df = df.dropna()
    return df, path.split('/')[-1]



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
    
    X = df.iloc[: , :-1]
    y = df.iloc[: , -1].copy()

    mfe = MFE(random_state=42, 
          groups=["complexity", "concept", "general", "itemset", "landmarking", "model-based", "statistical"], 
          summary=["mean", "sd", "kurtosis","skewness"])

    mfe.fit(X.values, y.values)
    ft = mfe.extract(suppress_warnings=True)
    
    df_characteristics = pd.DataFrame.from_records(ft)
    
    new_header = df_characteristics.iloc[0]
    df_characteristics = df_characteristics[1:]
    df_characteristics.columns = new_header
    
    df_characteristics.insert(loc=0, column="dataset", value=[dataset_name])
    
    
    encoded_columns = []
    for column_name in X.columns:
        if X[column_name].dtype == object or X[column_name].dtype.name == 'category' or X[column_name].dtype == bool or X[column_name].dtype == str:
            encoded_columns.extend([column_name])
    
    if encoded_columns:
        X = pd.get_dummies(X, columns=X[encoded_columns].columns, drop_first=True)

    if y.dtype == object or y.dtype.name == 'category' or y.dtype == bool or y.dtype == str:
        y = pd.Series(pd.factorize(y)[0], name=y.name)
    else:
        y = pd.Series(y, name=y.name)

    return X, y, df_characteristics



def pre_processing(balancing):
    """Create the configured resampling strategy for a given name."""
    
    balancing_technique = None
    
    # -- Under-sampling methods --
    if balancing == "ClusterCentroids":
        balancing_technique = ClusterCentroids(random_state=42)

    if balancing == "CondensedNearestNeighbour":
        balancing_technique = CondensedNearestNeighbour(random_state=42, n_jobs=-1)

    if balancing == "EditedNearestNeighbours":
        balancing_technique = EditedNearestNeighbours(n_jobs=-1)

    if balancing == "RepeatedEditedNearestNeighbours":
        balancing_technique = RepeatedEditedNearestNeighbours(n_jobs=-1)

    if balancing == "AllKNN":
        balancing_technique = AllKNN(n_jobs=-1)

    if balancing == "InstanceHardnessThreshold":
        balancing_technique = InstanceHardnessThreshold(random_state=42, n_jobs=-1)

    if balancing == "NearMiss":
        balancing_technique = NearMiss(n_jobs=-1)

    if balancing == "NeighbourhoodCleaningRule":
        balancing_technique = NeighbourhoodCleaningRule(n_jobs=-1)

    if balancing == "OneSidedSelection":
        balancing_technique = OneSidedSelection(random_state=42, n_jobs=-1)

    if balancing == "RandomUnderSampler":
        balancing_technique = RandomUnderSampler(random_state=42) #sampling_strategy=0.5
    
    if balancing == "TomekLinks":
        balancing_technique = TomekLinks(n_jobs=-1)
    
    
    # -- Over-sampling methods --
    if balancing == "RandomOverSampler":
        balancing_technique = RandomOverSampler(random_state=42) #sampling_strategy=0.5
    
    if balancing == "SMOTE":
        balancing_technique = SMOTE(random_state=42, n_jobs=-1) #sampling_strategy=0.5
    
    if balancing == "ADASYN":
        balancing_technique = ADASYN(random_state=42, n_jobs=-1)
    
    if balancing == "BorderlineSMOTE":
        balancing_technique = BorderlineSMOTE(random_state=42, n_jobs=-1)
    
    if balancing == "KMeansSMOTE":
        #UserWarning: MiniBatchKMeans
        # kmeans = MiniBatchKMeans(batch_size=2048)
        # , kmeans_estimator=kmeans
        
        # imbalance_ratio = 0
        # if y.values.tolist().count([0]) > 0 and y.values.tolist().count([1]) > 0:
        #     if y.values.tolist().count([0]) >= y.values.tolist().count([1]):
        #         imbalance_ratio = round(y.values.tolist().count([0])/y.values.tolist().count([1]),3)
        #     else:
        #         imbalance_ratio = round(y.values.tolist().count([1])/y.values.tolist().count([0]),3)
        
        # n_clusters = 1/imbalance_ratio
        
        balancing_technique = KMeansSMOTE(random_state=42, n_jobs=-1) #cluster_balance_threshold=n_clusters
    
    if balancing == "SVMSMOTE":
        balancing_technique = SVMSMOTE(random_state=42, n_jobs=-1)
    
    
    # -- Combination of over- and under-sampling methods --
    if balancing == "SMOTEENN":
        balancing_technique = SMOTEENN(random_state=42, n_jobs=-1)
        
    if balancing == "SMOTETomek":
        balancing_technique = SMOTETomek(random_state=42, n_jobs=-1)
    
    return balancing_technique



# initial:  1 + 19  balancing techniques and    11  classification algorithms   = 220   combinations
# second:   1 + 14  balancing techniques and    8   classification algorithms   = 120   combinations
# third:    12      balancing techniques and    6   classification algorithms   = 72    combinations
# fourth:   7       balancing techniques and    4   classification algorithms   = 28    combinations
# fifth:    5       balancing techniques and    3   classification algorithms   = 15    combinations
# final:    4       balancing techniques and    3   classification algorithms   = 12    combinations
def classify_evaluate(X, y, balancing, balancing_technique, dataset_name, problem_type):
    """Evaluate each classifier with the selected resampling strategy."""

    n_classes = get_target_class_count(y)
    array_classifiers = build_classifiers(problem_type, n_classes)
    
    resultsList = []
    
    for classifier in array_classifiers:
        start_time = time.time()
        
        model = make_pipeline(
            balancing_technique,
            classifier
        )
        
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
        
        scoring = get_scoring(problem_type)
        
        scores = cross_validate(model, X, y.values.ravel(), scoring=scoring,cv=cv, n_jobs=-1) #, return_train_score=True
        
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



def find_best_result(resultsList):
    """Select the result with the highest composite score."""
    scores = []
    for result in resultsList:
        scores.append(calculate_result_score(result))

    best_score = max(scores)
    index = scores.index(best_score)
    best_result = resultsList[index]
    
    string_balancing = best_result.balancing
    
    print("\nBest classifier is", best_result.algorithm, "with", string_balancing, "\n")
    
    return best_result



# determine if application is a script file or frozen exe
application_path = ""
if getattr(sys, 'frozen', False):
    application_path = os.path.dirname(sys.executable)
elif __file__:
    application_path = os.path.dirname(__file__)



def write_characteristics(df_characteristics, best_result, result_updated, problem_type):
    """Persist dataset characteristics and the selected pipeline metadata."""
    if df_characteristics.empty:
        print("--df_characteristics not valid on write_characteristics--")
        print("df_characteristics:", df_characteristics)
        return False
    
    try:
    
        df_kb_c = load_kb_dataframe(
            "kb_characteristics",
            problem_type,
            columns=list(df_characteristics.columns) + ["pre processing", "algorithm"],
        )
        #print(df_kb_c, '\n')
        
        dataset_name = df_characteristics["dataset"].iloc[0]
        df_kb_c_without = df_kb_c.loc[df_kb_c["dataset"] != dataset_name]
        df_kb_c_selected = df_kb_c.loc[df_kb_c["dataset"] == dataset_name]
        
        df_characteristics = pd.concat([df_characteristics, df_kb_c_without])
        df_characteristics = df_characteristics.reset_index(drop=True)
        
        #execute_ml
        if best_result and best_result.balancing and best_result.algorithm:
            #row updated or new line
            if result_updated or df_kb_c_selected.empty:
                df_characteristics.at[0, 'pre processing'] = best_result.balancing
                df_characteristics.at[0, 'algorithm'] = best_result.algorithm
                
                print("Characteristics written, row added or updated!","\n")
            
            #it was worse
            else:
                df_characteristics.at[0, 'pre processing'] = df_kb_c_selected["pre processing"].values[0]
                df_characteristics.at[0, 'algorithm'] = df_kb_c_selected["algorithm"].values[0]
                
                print("Characteristics not written!","\n")
        
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
        
    except Exception:
        traceback.print_exc()
        return False

    return True   



#writes if best
def write_results(best_result, elapsed_time):
    """Persist the best overall result if it improves the stored record."""
    if not best_result:
        print("--best_result or elapsed_time not valid on write_results--")
        print("best_result:", best_result)
        print("elapsed_time:", elapsed_time)
        return False
    
    result_updated = False
    
    try:
        
        current_value = calculate_result_score(best_result)
        
        elapsed_time = str(datetime.timedelta(seconds=round(elapsed_time,0)))
        
        print("Best Final Score Obtained    :", current_value)
        print("Elapsed Time                 :", elapsed_time, "\n")
        
        df_kb_r = load_kb_dataframe("kb_results", best_result.problem_type, columns=get_results_columns())

        if best_result.problem_type == "multiclass":
            multiclass_precision_macro = best_result.multiclass_precision_macro
            multiclass_precision_macro_std = best_result.multiclass_precision_macro_std
            multiclass_recall_macro = best_result.multiclass_recall_macro
            multiclass_recall_macro_std = best_result.multiclass_recall_macro_std
            multiclass_f1_weighted = best_result.multiclass_f1_weighted
            multiclass_f1_weighted_std = best_result.multiclass_f1_weighted_std
            multiclass_matthews_corrcoef = best_result.multiclass_matthews_corrcoef
            multiclass_matthews_corrcoef_std = best_result.multiclass_matthews_corrcoef_std
        else:
            multiclass_precision_macro = np.nan
            multiclass_precision_macro_std = np.nan
            multiclass_recall_macro = np.nan
            multiclass_recall_macro_std = np.nan
            multiclass_f1_weighted = np.nan
            multiclass_f1_weighted_std = np.nan
            multiclass_matthews_corrcoef = np.nan
            multiclass_matthews_corrcoef_std = np.nan
        
        df_kb_r2 = df_kb_r.loc[df_kb_r['dataset'] == best_result.dataset_name]
        
        if not df_kb_r2.empty :
            
            previous_value = round(np.mean([df_kb_r2['balanced accuracy'], df_kb_r2['f1 score'], df_kb_r2['roc auc'], df_kb_r2['geometric mean'], df_kb_r2['cohen kappa']]), 3)
            
            if current_value > previous_value:
                
                index = df_kb_r2.index.values[0]
                df_kb_r.at[index, 'pre processing'] = best_result.balancing
                df_kb_r.at[index, 'algorithm'] = best_result.algorithm
                df_kb_r.at[index, 'time'] = best_result.time
                df_kb_r.at[index, 'balanced accuracy'] = best_result.balanced_accuracy
                df_kb_r.at[index, 'balanced accuracy std'] = best_result.balanced_accuracy_std
                df_kb_r.at[index, 'f1 score'] = best_result.f1_score
                df_kb_r.at[index, 'f1 score std'] = best_result.f1_score_std
                df_kb_r.at[index, 'roc auc'] = best_result.roc_auc_score
                df_kb_r.at[index, 'roc auc std'] = best_result.roc_auc_score_std
                df_kb_r.at[index, 'geometric mean'] = best_result.g_mean_score
                df_kb_r.at[index, 'geometric mean std'] = best_result.g_mean_score_std
                df_kb_r.at[index, 'cohen kappa'] = best_result.cohen_kappa_score
                df_kb_r.at[index, 'cohen kappa std'] = best_result.cohen_kappa_score_std
                df_kb_r.at[index, 'multiclass precision macro'] = multiclass_precision_macro
                df_kb_r.at[index, 'multiclass precision macro std'] = multiclass_precision_macro_std
                df_kb_r.at[index, 'multiclass recall macro'] = multiclass_recall_macro
                df_kb_r.at[index, 'multiclass recall macro std'] = multiclass_recall_macro_std
                df_kb_r.at[index, 'multiclass f1 weighted'] = multiclass_f1_weighted
                df_kb_r.at[index, 'multiclass f1 weighted std'] = multiclass_f1_weighted_std
                df_kb_r.at[index, 'multiclass matthews corrcoef'] = multiclass_matthews_corrcoef
                df_kb_r.at[index, 'multiclass matthews corrcoef std'] = multiclass_matthews_corrcoef_std
                df_kb_r.at[index, 'total elapsed time'] = elapsed_time
                
                df_kb_r.to_csv(get_kb_file_path("kb_results", best_result.problem_type), sep=",", index=False)
                
                result_updated = True
                
                print("Results written, row updated!","\n")

            else:
                print("Results not written!","\n")
                
        else:
            
            df_kb_r.loc[len(df_kb_r.index)] = [
                best_result.dataset_name,
                best_result.balancing,
                best_result.algorithm,
                best_result.time,
                best_result.balanced_accuracy,
                best_result.balanced_accuracy_std,
                best_result.f1_score,
                best_result.f1_score_std,
                best_result.roc_auc_score,
                best_result.roc_auc_score_std,
                best_result.g_mean_score,
                best_result.g_mean_score_std,
                best_result.cohen_kappa_score,
                best_result.cohen_kappa_score_std,
                multiclass_precision_macro,
                multiclass_precision_macro_std,
                multiclass_recall_macro,
                multiclass_recall_macro_std,
                multiclass_f1_weighted,
                multiclass_f1_weighted_std,
                multiclass_matthews_corrcoef,
                multiclass_matthews_corrcoef_std,
                elapsed_time
            ]

            df_kb_r.to_csv(get_kb_file_path("kb_results", best_result.problem_type), sep=",", index=False)
            
            print("Results written, row added!","\n")  
        
    except Exception:
        traceback.print_exc()
        return False
    
    return result_updated



#only writes at first time 
def write_full_results(resultsList, dataset_name):
    """Persist all evaluated combinations for a dataset the first time only."""
    if not resultsList or not dataset_name:
        print("--resultsList not valid on write_full_results--")
        print("resultsList:", resultsList)
        print("dataset_name:", dataset_name)
        return False
    
    try:
    
        problem_type = resultsList[0].problem_type
        df_kb_r = load_kb_dataframe("kb_full_results", problem_type, columns=get_full_results_columns())
        
        df_kb_r2 = df_kb_r.loc[df_kb_r['dataset'] == dataset_name]
        
        if df_kb_r2.empty :
        
            for result in resultsList:

                if result.problem_type == "multiclass":
                    multiclass_precision_macro = result.multiclass_precision_macro
                    multiclass_precision_macro_std = result.multiclass_precision_macro_std
                    multiclass_recall_macro = result.multiclass_recall_macro
                    multiclass_recall_macro_std = result.multiclass_recall_macro_std
                    multiclass_f1_weighted = result.multiclass_f1_weighted
                    multiclass_f1_weighted_std = result.multiclass_f1_weighted_std
                    multiclass_matthews_corrcoef = result.multiclass_matthews_corrcoef
                    multiclass_matthews_corrcoef_std = result.multiclass_matthews_corrcoef_std
                else:
                    multiclass_precision_macro = np.nan
                    multiclass_precision_macro_std = np.nan
                    multiclass_recall_macro = np.nan
                    multiclass_recall_macro_std = np.nan
                    multiclass_f1_weighted = np.nan
                    multiclass_f1_weighted_std = np.nan
                    multiclass_matthews_corrcoef = np.nan
                    multiclass_matthews_corrcoef_std = np.nan
                
                df_kb_r.loc[len(df_kb_r.index)] = [
                        result.dataset_name,
                        result.balancing,
                        result.algorithm,
                        result.time,
                        result.balanced_accuracy,
                        result.balanced_accuracy_std,
                        result.f1_score,
                        result.f1_score_std,
                        result.roc_auc_score,
                        result.roc_auc_score_std,
                        result.g_mean_score,
                        result.g_mean_score_std,
                        result.cohen_kappa_score,
                        result.cohen_kappa_score_std,
                        multiclass_precision_macro,
                        multiclass_precision_macro_std,
                        multiclass_recall_macro,
                        multiclass_recall_macro_std,
                        multiclass_f1_weighted,
                        multiclass_f1_weighted_std,
                        multiclass_matthews_corrcoef,
                        multiclass_matthews_corrcoef_std,
                        calculate_result_score(result)
                    ]

            df_kb_r.sort_values(by=['final score'], ascending=False, inplace=True)

            df_kb_r.to_csv(get_kb_file_path("kb_full_results", problem_type), sep=",", index=False)
            
            print("Full Results written, rows added!","\n")
        
        else:
            print("Full Results not written!","\n")
        
    except Exception:
        traceback.print_exc()
        return False
    
    return True


def resolve_multiclass_dataset_path(dataset_name):
    """Resolve a dataset name to a CSV file inside input/multiclass."""
    if not dataset_name:
        raise ValueError("dataset_name is required")

    dataset_file = dataset_name if dataset_name.endswith(".csv") else dataset_name + ".csv"
    dataset_path = os.path.join(application_path, "input", "multiclass", dataset_file)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    return dataset_path


def main(dataset_name):
    """Run execute_ml_test for a dataset in input/multiclass."""
    dataset_path = resolve_multiclass_dataset_path(dataset_name)
    #return execute_byCharacteristics(dataset_path, None)
    #return execute_ml_test(dataset_path, None)
    return execute_ml(dataset_path, None)



#by Euclidean Distance
def get_best_results_by_characteristics(dataset_name, problem_type):
    """Find the most similar past datasets and reuse their best pipelines."""
    if not dataset_name:
        print("--dataset_name not valid on get_best_results_by_characteristics--")
        print("best_result:", dataset_name)
        return False
    
    df_c = load_kb_dataframe("kb_characteristics", problem_type)
    if df_c.empty:
        print("--kb_characteristics is empty for problem type--")
        print("problem_type:", problem_type)
        return False

    df_c = df_c.dropna(axis=1)
    df_c = df_c.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
    
    df_c_a = df_c.loc[df_c['dataset'] == dataset_name]
    df_c_a = df_c_a.drop(['dataset', 'pre processing','algorithm'], axis=1)
    list_a = df_c_a.values.tolist()[0]
    list_a = [(float(i)-min(list_a))/(max(list_a)-min(list_a)) for i in list_a]

    df_c = df_c.loc[df_c['dataset'] != dataset_name]
    list_dist = []
    for index, row in df_c.iterrows():
        df_c_b = row.to_frame()
        df_c_b = df_c_b.drop(['dataset', 'pre processing','algorithm'])
        list_b = df_c_b.values.tolist()
        list_b = [x for xs in list_b for x in xs]
        list_b = [(float(i)-min(list_b))/(max(list_b)-min(list_b)) for i in list_b]
        list_dist.append((row['dataset'], row['pre processing'], row['algorithm'], np.linalg.norm(np.array(list_a) - np.array(list_b))))
        
    df_dist = pd.DataFrame(list_dist, columns=["dataset", "pre processing", "algorithm","result"])
    df_dist = df_dist.sort_values(by=['result'])
    df_dist = df_dist.drop_duplicates(subset=['pre processing', 'algorithm'], keep='first')
    df_dist = df_dist.reset_index(drop=True)
    df_dist = df_dist.head(3)
    
    print("Results:\n", df_dist)
    
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


if __name__ == "__main__":
    dataset_name = "car_evaluation.csv"
    main(dataset_name)
