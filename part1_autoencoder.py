# ---------------------
# Stage 1: Autoencoder
# ---------------------
#
# CSV format (per spec):
#   col[0]      = dataTime       → timestamp
#   col[1]      = alarmState     → IGNORED
#   col[2]      = hr             → used (BPM, forward-filled if -1)
#   col[3]      = o2sat          → IGNORED (always -1)
#   col[4..128] = accel*125      → 125 milli-g readings
#
# The autoencoder should train on the baseline windows passed in by the caller.
# In the main pipeline, those windows can be filtered using a seizure log so the
# latent bank better reflects subject-specific normal accel + HR patterns.
# It compresses each window into a 16-dim latent vector used by Stage 3 (kNN).
#
# No ground-truth labels are used for training — only the raw sensor signals.
#
# Install: pip install scikit-learn scipy numpy
# Architecture: 20 features → 64 → 32 → 16 (latent) → 32 → 64 → 20


import numpy as np
import warnings
import glob

from scipy.signal import welch
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor


# ─────────────────────────────────────────────
# 1.  DATA LOADER
# ─────────────────────────────────────────────

def load_csv(filepath: str) -> list:
    """
    Parse one session CSV.

    Reads:   col[0] timestamp, col[2] hr, col[4..128] accel (125 values)
    Ignores: col[1] alarmState, col[3] o2sat

    Returns list of dicts: ts, hr, accel_mg
    (No 'alarm' field — alarmState is not read)
    """
    rows = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) < 5:
                continue
            try:
                rows.append({
                    'ts':       parts[0].strip(),
                    # col[1] alarmState → skipped
                    'hr':       float(parts[2].strip()),
                    # col[3] o2sat → skipped
                    'accel_mg': np.array([float(x) for x in parts[4:] if x.strip()]),
                })
            except (ValueError, IndexError):
                continue
    rows.sort(key=lambda r: r['ts'])
    return rows


def load_all_csvs(filepaths: list) -> list:
    """Load and time-sort multiple session CSVs."""
    all_rows = []
    for fp in filepaths:
        all_rows.extend(load_csv(fp))
    all_rows.sort(key=lambda r: r['ts'])
    return all_rows


def impute_hr(rows: list, fallback_bpm: float = 90.0) -> list:
    """
    Forward-fill missing HR (hr = -1 = sensor dropout, ~2.9% of windows).
    HR changes slowly so the previous reading is the best estimate.
    Operates in-place. Returns same list.
    """
    last_valid = fallback_bpm
    for r in rows:
        if r['hr'] <= 0:
            r['hr'] = last_valid
        else:
            last_valid = r['hr']
    return rows


# ─────────────────────────────────────────────
# 2.  FEATURE EXTRACTION  (accel + HR)
# ─────────────────────────────────────────────

def extract_features(row: dict) -> np.ndarray:
    """
    Extract 20 features from one window (18 accel-derived + 2 HR-derived).

    Accel features [0–17]:
      std_mg, p2p_mg, rms_mg, mean_mg,
      spike_count (>1200 mg), spike_max, p90_mg,
      jerk_std, jerk_max, jerk_mean,
      zero_cross,
      spec_energy, spec_peak_f, band_low, band_mid, band_high,
      accel_skew, accel_kurt

    HR features [18–19]:
      hr_bpm    – raw BPM (always valid after impute_hr)
      hr_norm   – (hr - 90) / 30, normalised to a target resting range
    """
    a  = row['accel_mg']
    hr = row['hr']

    std_mg      = float(np.std(a))
    p2p_mg      = float(np.max(a) - np.min(a))
    rms_mg      = float(np.sqrt(np.mean(a ** 2)))
    mean_mg     = float(np.mean(a))
    spike_count = float(np.sum(a > 1200))
    spike_max   = float(np.max(a))
    p90_mg      = float(np.percentile(a, 90))

    jerk      = np.diff(a)
    jerk_abs  = np.abs(jerk)
    jerk_std  = float(np.std(jerk))
    jerk_max  = float(jerk_abs.max()) if len(jerk) else 0.0
    jerk_mean = float(jerk_abs.mean()) if len(jerk) else 0.0

    centered   = a - mean_mg
    zero_cross = float(np.sum(np.diff(np.sign(centered)) != 0))

    freqs, psd  = welch(a, fs=25.0, nperseg=min(64, len(a)))
    spec_energy = float(np.sum(psd))
    spec_peak_f = float(freqs[np.argmax(psd)])
    band_low    = float(np.sum(psd[(freqs >= 0) & (freqs <  2)]))
    band_mid    = float(np.sum(psd[(freqs >= 2) & (freqs <  8)]))
    band_high   = float(np.sum(psd[(freqs >= 8) & (freqs < 12)]))

    accel_skew  = float(skew(a))
    accel_kurt  = float(kurtosis(a))

    hr_bpm  = float(hr)
    hr_norm = float(np.clip((hr - 90.0) / 30.0, -1.0, 2.0))

    return np.array([
        std_mg, p2p_mg, rms_mg, mean_mg,
        spike_count, spike_max, p90_mg,
        jerk_std, jerk_max, jerk_mean,
        zero_cross,
        spec_energy, spec_peak_f,
        band_low, band_mid, band_high,
        accel_skew, accel_kurt,
        hr_bpm, hr_norm,
    ], dtype=np.float32)


def compute_features(rows: list) -> np.ndarray:
    """
    Extract 20-feature matrix from all rows.
    Returns shape (n_windows, 20).
    """
    return np.array([extract_features(r) for r in rows], dtype=np.float32)


# ─────────────────────────────────────────────
# 3.  NORMALISATION
# ─────────────────────────────────────────────

def normalize_features(features: np.ndarray):
    """
    Z-score normalise all 20 features.
    Returns (normalised_array, fitted_scaler).
    The scaler MUST be saved and applied identically to all new data.
    """
    scaler = StandardScaler()
    norm   = scaler.fit_transform(features)
    return norm.astype(np.float32), scaler


# ─────────────────────────────────────────────
# 4.  AUTOENCODER
# ─────────────────────────────────────────────

def train_autoencoder(norm_features: np.ndarray, max_iter: int = 300):
    """
    Train autoencoder on the feature matrix supplied by the caller.
    In the main pipeline, this can be a seizure-log-filtered baseline set.
    Architecture: 20 → 64 → 32 → 16 → 32 → 64 → 20
    Returns fitted MLPRegressor.
    """
    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 32, 16, 32, 64),
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        max_iter=max_iter,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        verbose=False,
    )
    print(f"  Training on {norm_features.shape[0]:,} windows "
          f"({norm_features.shape[1]} features: 18 accel + 2 HR)...")
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mlp.fit(norm_features, norm_features)
    print(f"  Done. Loss: {mlp.loss_:.5f}  Iterations: {mlp.n_iter_}")
    return mlp


# ─────────────────────────────────────────────
# 5.  ENCODER
# ─────────────────────────────────────────────

def encode(norm_features: np.ndarray, mlp: MLPRegressor) -> np.ndarray:
    """
    Forward pass through encoder half only (layers 0–2 → 16-dim bottleneck).
    Returns shape (n_windows, 16).
    """
    h = norm_features.copy().astype(np.float64)
    for i in range(3):          # layers 0 (20→64), 1 (64→32), 2 (32→16)
        h = h @ mlp.coefs_[i] + mlp.intercepts_[i]
        h = np.maximum(0, h)    # ReLU
    return h.astype(np.float32)


def create_latent_bank(norm_features: np.ndarray, mlp: MLPRegressor) -> np.ndarray:
    """
    Encode all training windows → latent bank (personal normal reference).
    Returns shape (n_windows, 16).
    """
    return encode(norm_features, mlp)


# ─────────────────────────────────────────────
# 6.  QUICK DEMO
# ─────────────────────────────────────────────

if __name__ == '__main__':
    csv_files = sorted(glob.glob('data/*-data.csv'))
    if not csv_files:
        print("No CSVs found in ./data/")
        exit()

    print(f"Loading {len(csv_files)} file(s)...")
    all_rows = load_all_csvs(csv_files)
    all_rows = impute_hr(all_rows)
    print(f"  Total windows: {len(all_rows):,}")

    print("\nExtracting features...")
    feats = compute_features(all_rows)
    norm_feats, scaler = normalize_features(feats)
    print(f"  Feature matrix: {feats.shape}")

    print("\nTraining autoencoder...")
    mlp = train_autoencoder(norm_feats)

    print("\nBuilding latent bank...")
    latent_bank = create_latent_bank(norm_feats, mlp)
    print(f"  Shape: {latent_bank.shape}")
    print("Stage 1 complete.")
