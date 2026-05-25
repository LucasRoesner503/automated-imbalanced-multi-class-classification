import time
import logging
import numpy as np
import pandas as pd
from imblearn.pipeline import make_pipeline
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
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
from sklearn.metrics import make_scorer, cohen_kappa_score, precision_score, recall_score, matthews_corrcoef
from imblearn.metrics import geometric_mean_score

from config import logger, CV_N_SPLITS, CV_N_REPEATS, RANDOM_STATE, Results
from data import get_target_class_count

# Configure module logger
logger = logging.getLogger(__name__)


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
    #            random_state=RANDOM_STATE,
    #            objective='multiclass',
    #            num_class=n_classes,
    #            class_weight='balanced',
    #            force_col_wise=True,
    #            n_jobs=-1,
    #        ),
    #        XGBClassifier(
    #            random_state=RANDOM_STATE,
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
    #            random_state=RANDOM_STATE,
    #            objective='binary',
    #            class_weight='balanced',
    #            force_col_wise=True,
    #            n_jobs=-1,
    #        ),
    #        XGBClassifier(
    #            random_state=RANDOM_STATE,
    #            use_label_encoder=False,
    #            objective='binary:logistic',
    #            eval_metric='logloss',
    #            n_jobs=-1,
    #        ),
    #    ])

    return classifiers


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

    matthews_corrcoef_val = round(np.mean(scores['test_matthews_corrcoef']), 3)
    matthews_corrcoef_std = round(np.std(scores['test_matthews_corrcoef']), 3)

    return (
        precision_macro,
        precision_macro_std,
        recall_macro,
        recall_macro_std,
        f1_weighted,
        f1_weighted_std,
        matthews_corrcoef_val,
        matthews_corrcoef_std,
    )


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
        
        try:
            scores = cross_validate(model, X, y.ravel() if hasattr(y, 'ravel') else y, scoring=scoring, cv=cv, n_jobs=-1)
        except ValueError as e:
            if 'roc_auc' in str(e):
                logger.warning(f"ROC-AUC failed for {balancing} + {classifier.__class__.__name__}, retrying without ROC-AUC")
                scoring_without_roc = {k: v for k, v in scoring.items() if k != 'roc_auc'}
                scores = cross_validate(model, X, y.ravel() if hasattr(y, 'ravel') else y, scoring=scoring_without_roc, cv=cv, n_jobs=-1)
                scores['test_roc_auc'] = np.full(len(scores['test_balanced_accuracy']), np.nan)
            else:
                raise
        
        finish_time = round(time.time() - start_time, 3)
        
        balanced_accuracy = round(np.mean(scores['test_balanced_accuracy']), 3)
        f1_score = round(np.mean(scores['test_f1']), 3)
        roc_auc_score = round(np.mean(scores['test_roc_auc']), 3)
        g_mean_score = round(np.mean(scores['test_g_mean']), 3)
        cohen_kappa = round(np.mean(scores['test_cohen_kappa']), 3)
        
        balanced_accuracy_std = round(np.std(scores['test_balanced_accuracy']), 3)
        f1_score_std = round(np.std(scores['test_f1']), 3)
        roc_auc_score_std = round(np.std(scores['test_roc_auc']), 3)
        g_mean_score_std = round(np.std(scores['test_g_mean']), 3)
        cohen_kappa_std = round(np.std(scores['test_cohen_kappa']), 3)

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
