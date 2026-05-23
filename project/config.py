import os
import sys
import logging
import numpy as np

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

# Determine if application is a script file or frozen exe
application_path = ""
if getattr(sys, 'frozen', False):
    application_path = os.path.dirname(sys.executable)
elif __file__:
    application_path = os.path.dirname(__file__)


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
