# ---------------------
# Stage 3: kNN Anomaly Detector
# ---------------------
#
# No ground-truth labels used — scores windows purely from sensor signals.
#
# Uses the latent bank from Stage 1 (trained on accel + HR) to score each
# new window by how far it deviates from the learned personal baseline
# in the 16-dim latent space.
#
# The final combined score (ensemble of Stage 2 + Stage 3) feeds the
# same state machine to produce: 0=OK, 1=WARNING, 2=ALARM.


import numpy as np
from sklearn.neighbors import NearestNeighbors


W_DETERMINISTIC = 0.50
W_ML            = 0.50


class LatentKNNScorer:
    """
    kNN scorer on the 16-dim latent space from Stage 1.
    Higher distance = more anomalous = more likely a seizure.
    """

    def __init__(self, k: int = 5, metric: str = 'euclidean'):
        self.k         = k
        self.metric    = metric
        self.knn       = None
        self.threshold = None

    def fit(self, latent_bank: np.ndarray):
        """latent_bank: (n_windows, 16) from create_latent_bank() in Stage 1."""
        self.knn = NearestNeighbors(
            n_neighbors=self.k, metric=self.metric, algorithm='ball_tree'
        )
        self.knn.fit(latent_bank)
        return self

    def score(self, latent_vectors: np.ndarray) -> np.ndarray:
        """Returns raw mean kNN distances, shape (n_windows,)."""
        if self.knn is None:
            raise RuntimeError("Call .fit(latent_bank) first.")
        distances, _ = self.knn.kneighbors(latent_vectors)
        return distances.mean(axis=1)

    def calibrate_threshold(self, latent_bank: np.ndarray,
                            percentile: float = 95.0) -> float:
        """
        Set threshold = Nth percentile of the training data's self-scored distances.
        Windows scoring above this are considered anomalous.
        """
        baseline_scores = self.score(latent_bank)
        self.threshold  = float(np.percentile(baseline_scores, percentile))
        print(f"  [kNN] Threshold ({percentile}th pct): {self.threshold:.4f}")
        return self.threshold

    def normalise_scores(self, raw_scores: np.ndarray,
                         clip_multiplier: float = 3.0) -> np.ndarray:
        """Map raw kNN distances → [0, 1]."""
        if self.threshold is None:
            raise RuntimeError("Call .calibrate_threshold() first.")
        return np.clip(raw_scores, 0, self.threshold * clip_multiplier) / \
               (self.threshold * clip_multiplier)


def run_ml_inference(rows: list, mlp, scaler,
                     knn_scorer: LatentKNNScorer) -> np.ndarray:
    """
    Score windows using trained autoencoder + kNN.
    impute_hr() must have been called on rows before this.
    Returns normalised ML anomaly scores [0, 1], shape (len(rows),).
    """
    from part1_autoencoder import extract_features, encode

    feats      = np.array([extract_features(r) for r in rows], dtype=np.float32)
    norm_feats = scaler.transform(feats).astype(np.float32)
    latent     = encode(norm_feats, mlp)
    raw_scores = knn_scorer.score(latent)
    return knn_scorer.normalise_scores(raw_scores)


def ensemble_score(det_smooth: np.ndarray, ml_score: np.ndarray,
                   w_det: float = W_DETERMINISTIC,
                   w_ml:  float = W_ML) -> np.ndarray:
    """Weighted average of deterministic + ML scores, both in [0, 1]."""
    return np.clip(w_det * det_smooth + w_ml * ml_score, 0.0, 1.0)
