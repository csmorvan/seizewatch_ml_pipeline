# ---------------------
# Stage 4: Supervised Calibrator
# ---------------------
#
# Uses seizure-log labels to learn how to combine the earlier pipeline stages.
#
# Inputs come from:
#   Stage 2 -> deterministic seizure-rule scores
#   Stage 3 -> latent-space kNN anomaly score
#
# Output:
#   A compact logistic-regression layer that maps those signals to a seizure
#   probability, then derives WARN / ALARM thresholds for the main pipeline.
#
# Unlike Stages 1-3, this stage does use ground-truth labels from the
# reference event log.
#
# Install: pip install scikit-learn numpy


import numpy as np
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------
# 1.  SETTINGS
# ---------------------------------------------

FEATURE_NAMES = [
    'smooth_score',
    'ml_score',
    'ensemble_score',
    'burst_std_surprise',
    'burst_jerk_mean_surprise',
    'ml_surprise',
    'accel_score',
    'jerk_score',
    'hr_score',
    'jerk_mean',
    'jerk_max',
    'burst_std_1s_max',
    'burst_jerk_mean_1s_max',
    'burst_jerk_max_1s_max',
    'hr',
    'focal_flag',
    'legacy_warn_flag',
    'legacy_alarm_flag',
]

WARN_TARGET_RECALL  = 0.80
ALARM_TARGET_RECALL = 0.70
WARN_MAX_FALSE_POSITIVE_RATE  = 0.20
ALARM_MAX_FALSE_POSITIVE_RATE = 0.10
MODEL_C = 0.35
MIN_POSITIVE_CLASS_WEIGHT = 3.0
MAX_POSITIVE_CLASS_WEIGHT = 12.0
TONIC_CLONIC_POSITIVE_MULTIPLIER = 2.5
STARTLE_POSITIVE_MULTIPLIER = 2.0


# ---------------------------------------------
# 2.  CALIBRATOR CONTAINER
# ---------------------------------------------

@dataclass
class SupervisedCalibrator:
    scaler: StandardScaler
    model: LogisticRegression
    feature_names: list
    warn_threshold: float
    alarm_threshold: float

    def score_rows(self, rows: list) -> np.ndarray:
        """Return seizure probabilities for the supplied rows."""
        feats = build_supervised_features(rows)
        norm_feats = self.scaler.transform(feats)
        return self.model.predict_proba(norm_feats)[:, 1].astype(np.float32)


# ---------------------------------------------
# 3.  FEATURE + LABEL BUILDERS
# ---------------------------------------------

def build_supervised_features(rows: list) -> np.ndarray:
    """Feature matrix for Stage 4, built from the earlier pipeline stages."""
    return np.array([
        [
            float(r.get('smooth_score', 0.0)),
            float(r.get('ml_score', 0.0)),
            float(r.get('ensemble_score', 0.0)),
            float(r.get('burst_std_surprise', 1.0)),
            float(r.get('burst_jerk_mean_surprise', 1.0)),
            float(r.get('ml_surprise', 0.0)),
            float(r.get('accel_score', 0.0)),
            float(r.get('jerk_score', 0.0)),
            float(r.get('hr_score', 0.0)),
            float(r.get('jerk_mean', 0.0)),
            float(r.get('jerk_max', 0.0)),
            float(r.get('burst_std_1s_max', 0.0)),
            float(r.get('burst_jerk_mean_1s_max', 0.0)),
            float(r.get('burst_jerk_max_1s_max', 0.0)),
            float(r.get('hr', 0.0)),
            float(bool(r.get('focal_flag', False))),
            float(int(r.get('legacy_state', 0)) >= 1),
            float(int(r.get('legacy_state', 0)) >= 2),
        ]
        for r in rows
    ], dtype=np.float32)


def build_supervised_labels(rows: list) -> np.ndarray:
    """Binary labels from the seizure-log annotations added in main_pipeline."""
    return np.array([1 if r.get('log_is_seizure') else 0 for r in rows], dtype=np.int32)


# ---------------------------------------------
# 4.  CLASS WEIGHTING
# ---------------------------------------------

def build_class_weight(labels: np.ndarray) -> dict:
    """
    Use a moderated positive-class weight.

    The previous fully-balanced weighting made positives ~50x heavier than
    normals in this dataset, which drove Stage 4 toward near-constant alarms.
    """
    positive_total = int(np.sum(labels == 1))
    negative_total = int(np.sum(labels == 0))
    if positive_total == 0 or negative_total == 0:
        return {0: 1.0, 1: 1.0}

    imbalance = negative_total / positive_total
    positive_weight = float(np.clip(
        np.sqrt(imbalance),
        MIN_POSITIVE_CLASS_WEIGHT,
        MAX_POSITIVE_CLASS_WEIGHT,
    ))
    return {0: 1.0, 1: positive_weight}


def build_sample_weight(rows: list,
                        labels: np.ndarray,
                        class_weight: dict) -> np.ndarray:
    """
    Per-row weights used during Stage 4 fitting.

    Startle seizures are upweighted because short startle-like events are
    clinically important and otherwise easier for the detector to miss.
    """
    sample_weight = np.ones(len(rows), dtype=np.float32)
    for idx, row in enumerate(rows):
        label = int(labels[idx])
        sample_weight[idx] = float(class_weight.get(label, 1.0))
        if label != 1:
            continue
        if str(row.get('log_event_type', '')).strip().lower() == 'tonic-clonic':
            sample_weight[idx] *= TONIC_CLONIC_POSITIVE_MULTIPLIER
        elif bool(row.get('log_is_startle', False)):
            sample_weight[idx] *= STARTLE_POSITIVE_MULTIPLIER
    return sample_weight


# ---------------------------------------------
# 5.  THRESHOLD SELECTION
# ---------------------------------------------

def choose_threshold(scores: np.ndarray,
                     labels: np.ndarray,
                     target_recall: float,
                     max_false_positive_rate: float = None) -> float:
    """
    Pick the highest threshold that still achieves the target recall while
    minimising false positives on normal-labelled windows.
    """
    positive_total = int(labels.sum())
    negative_total = int((labels == 0).sum())
    if positive_total == 0:
        return 0.50

    candidates = np.unique(np.concatenate([
        np.linspace(0.0, 1.0, 201, dtype=np.float32),
        np.asarray(scores, dtype=np.float32),
    ]))

    feasible = []
    fallback = []
    for threshold in np.sort(candidates):
        preds = scores >= threshold
        tp = int(np.sum(preds & (labels == 1)))
        fp = int(np.sum(preds & (labels == 0)))
        recall = tp / positive_total if positive_total else 0.0
        fpr = fp / negative_total if negative_total else 0.0
        precision = tp / max(int(np.sum(preds)), 1)
        record = (fpr, -precision, -threshold, threshold, recall)
        if recall >= target_recall and (
            max_false_positive_rate is None or fpr <= max_false_positive_rate
        ):
            feasible.append(record)
        else:
            deficit = max(0.0, target_recall - recall)
            extra_fpr = max(
                0.0,
                0.0 if max_false_positive_rate is None else fpr - max_false_positive_rate,
            )
            fallback.append((deficit, extra_fpr, fpr, -precision, -threshold, threshold))

    if feasible:
        feasible.sort()
        return float(feasible[0][3])

    fallback.sort()
    return float(fallback[0][5]) if fallback else 0.50


def compute_threshold_metrics(scores: np.ndarray,
                              labels: np.ndarray,
                              threshold: float) -> dict:
    """Window-level recall / precision / false-positive-rate summary."""
    preds = scores >= threshold
    tp = int(np.sum(preds & (labels == 1)))
    fp = int(np.sum(preds & (labels == 0)))
    fn = int(np.sum((~preds) & (labels == 1)))
    tn = int(np.sum((~preds) & (labels == 0)))

    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    false_positive_rate = fp / max(fp + tn, 1)

    return {
        'threshold': float(threshold),
        'recall': float(recall),
        'precision': float(precision),
        'false_positive_rate': float(false_positive_rate),
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
    }


# ---------------------------------------------
# 6.  TRAINING
# ---------------------------------------------

def fit_supervised_calibrator(rows: list,
                              warn_target_recall: float = WARN_TARGET_RECALL,
                              alarm_target_recall: float = ALARM_TARGET_RECALL,
                              warn_max_false_positive_rate: float = WARN_MAX_FALSE_POSITIVE_RATE,
                              alarm_max_false_positive_rate: float = ALARM_MAX_FALSE_POSITIVE_RATE):
    """
    Train a compact supervised layer on top of Stages 1-3.

    Returns (calibrator, training_summary).
    """
    feats = build_supervised_features(rows)
    labels = build_supervised_labels(rows)

    scaler = StandardScaler()
    norm_feats = scaler.fit_transform(feats)

    class_weight = build_class_weight(labels)
    sample_weight = build_sample_weight(rows, labels, class_weight)
    model = LogisticRegression(
        max_iter=1000,
        C=MODEL_C,
        random_state=42,
        solver='lbfgs',
    )
    model.fit(norm_feats, labels, sample_weight=sample_weight)

    train_scores = model.predict_proba(norm_feats)[:, 1]
    warn_threshold = choose_threshold(
        train_scores,
        labels,
        target_recall=warn_target_recall,
        max_false_positive_rate=warn_max_false_positive_rate,
    )
    alarm_threshold = choose_threshold(
        train_scores,
        labels,
        target_recall=alarm_target_recall,
        max_false_positive_rate=alarm_max_false_positive_rate,
    )

    if warn_threshold >= alarm_threshold:
        warn_threshold = max(0.0, min(alarm_threshold - 0.05, alarm_threshold * 0.8))

    calibrator = SupervisedCalibrator(
        scaler=scaler,
        model=model,
        feature_names=list(FEATURE_NAMES),
        warn_threshold=float(warn_threshold),
        alarm_threshold=float(alarm_threshold),
    )

    coef_pairs = sorted(
        zip(FEATURE_NAMES, model.coef_[0]),
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    top_features = [(name, float(weight)) for name, weight in coef_pairs[:5]]

    summary = {
        'n_rows': int(len(rows)),
        'n_normal': int(np.sum(labels == 0)),
        'n_seizure': int(np.sum(labels == 1)),
        'n_startle': int(sum(1 for row in rows if row.get('log_is_startle'))),
        'n_tonic_clonic': int(sum(
            1 for row in rows if str(row.get('log_event_type', '')).strip().lower() == 'tonic-clonic'
        )),
        'class_weight': {int(k): float(v) for k, v in class_weight.items()},
        'warn_metrics': compute_threshold_metrics(train_scores, labels, calibrator.warn_threshold),
        'alarm_metrics': compute_threshold_metrics(train_scores, labels, calibrator.alarm_threshold),
        'top_features': top_features,
    }
    return calibrator, summary
