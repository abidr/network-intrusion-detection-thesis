"""Hyperparameters exactly as specified in the paper.

Every value below is quoted from CyberAdaptAI section 4.1 (Experimental Setup) unless it is
marked NOT-IN-PAPER, in which case the paper left it unspecified and the value is a documented
choice by this reimplementation (see README, "Where the paper is ambiguous").
"""

from dataclasses import dataclass, field
from enum import Enum

SEED = 42  # NOT-IN-PAPER: section 4.1 seeds nothing. Fixed here for reproducibility.


class WeightRule(str, Enum):
    """Which of the paper's two conflicting weight updates to apply. See README ambiguity 1."""

    EQ3 = "eq3"        # w = alpha*A + (1-alpha)*D          (Eq 3, no smoothing)
    EQ8 = "eq8"        # w = gamma*w + (1-gamma)*(1-E)      (Eq 8, no diversity; Algorithm 1)
    HYBRID = "hybrid"  # Eq-3 target, smoothed by Eq-8 gamma (reconciles both)


class DiversityMeasure(str, Enum):
    """See README ambiguity 2 -- raw Q has the wrong sign for Eq 3."""

    DISAGREEMENT = "disagreement"  # (N10+N01)/N, matches the paper's prose
    Q_STATISTIC = "q_statistic"    # Eq 4 verbatim
    NEG_Q = "neg_q"                # (1 - Q)/2, sign-corrected Eq 4


class RetrainScope(str, Enum):
    """What data a drift-triggered retrain uses. Algorithm 1 line 11 says only "retrain
    underperforming classifiers"; section 4.1 says "retrained with new data from the current
    window". Neither says whether the original training data is retained.

    WINDOW      refit on the current window alone (~5 batches). Literal reading of section 4.1,
                but destructive: a learner trained on 124k rows is rebuilt from ~10k.
    CUMULATIVE  refit on the initial training set plus everything streamed so far. Keeps the
                "new data" of section 4.1 while not discarding what the model already had.
    """

    WINDOW = "window"
    CUMULATIVE = "cumulative"


class ScalerFit(str, Enum):
    """Scope of min/max in Eq 1. See README ambiguity 3."""

    FIRST_BATCH = "first_batch"  # leakage-safe
    GLOBAL = "global"            # likely the paper's intent; leaks future batches
    RUNNING = "running"          # running min/max, updated per batch


@dataclass
class Config:
    # --- Stream simulation (section 3.2, 4.1) ---
    # "chronologically ordered mini-batches of approximately 1,500 to 2,000 samples per batch"
    batch_size: int = 2000
    train_test_split: float = 0.8  # "Holding the ratio fixed at 80:20 in train to test"

    # --- Base learner pool (section 4.1) ---
    # "Random Forest was configured with 100 estimators and a maximum depth of 15"
    rf_n_estimators: int = 100
    rf_max_depth: int = 15
    # "XGBoost was set with a learning rate of 0.1, 150 estimators, and a maximum depth of 5"
    xgb_learning_rate: float = 0.1
    xgb_n_estimators: int = 150
    xgb_max_depth: int = 5
    # "Extra Trees was initialized with 100 trees, entropy as the splitting criterion,
    #  and maximum features set to 'sqrt'"
    et_n_estimators: int = 100
    et_criterion: str = "entropy"
    et_max_features: str = "sqrt"
    # "The Hoeffding Tree ... was sourced from the River library and used default confidence
    #  parameters for incremental learning."  (optional dependency; see base_learners.py)
    use_hoeffding_tree: bool = True

    # --- Adaptive weighting (Eq 3, Eq 8; section 4.1) ---
    # "Adaptive weights were initialized equally"
    # "the smoothing parameter gamma for the exponential moving average error was set to 0.8"
    gamma: float = 0.8
    # "The weighting trade-off parameter alpha between accuracy and diversity was fixed at 0.6"
    alpha: float = 0.6
    weight_rule: WeightRule = WeightRule.HYBRID
    diversity_measure: DiversityMeasure = DiversityMeasure.DISAGREEMENT

    # --- Drift detection (Eq 6-7; section 3.4, 4.1) ---
    # "ADWIN was used to address concept drift (delta=0.002)"
    adwin_delta: float = 0.002
    # "A sliding window of size five was used to observe a drift in the accuracy of the model"
    accuracy_window: int = 5
    # "If accuracy decreases consistently by more than 8% over three consecutive batches,
    #  weaker classifiers are retrained with new data from the current window."
    accuracy_drop_threshold: float = 0.08
    accuracy_drop_batches: int = 3
    adwin_max_window: int = 2000  # NOT-IN-PAPER: bound on the error window, for cost control
    retrain_scope: RetrainScope = RetrainScope.WINDOW

    # --- Preprocessing (Eq 1; section 3.2) ---
    scaler_fit: ScalerFit = ScalerFit.FIRST_BATCH
    drop_duplicates: bool = True  # "eliminating duplicate records"
    impute_strategy: str = "mean"  # "mean imputation of missing values"

    # --- Explainability (section 3.5, 4.1) ---
    # "SHAP values were calculated using TreeExplainer for the XGBoost model"
    enable_shap: bool = True
    shap_sample_size: int = 200  # NOT-IN-PAPER: TreeExplainer cost control

    # --- Misc ---
    seed: int = SEED
    verbose: bool = True

    def describe(self) -> str:
        lines = [
            "CyberAdaptAI config",
            f"  batch_size        = {self.batch_size}",
            f"  weight_rule       = {self.weight_rule.value}",
            f"  diversity_measure = {self.diversity_measure.value}",
            f"  alpha (Eq 3)      = {self.alpha}",
            f"  gamma (Eq 8)      = {self.gamma}",
            f"  ADWIN delta       = {self.adwin_delta}",
            f"  scaler_fit        = {self.scaler_fit.value}",
        ]
        return "\n".join(lines)
