# ---------------------
# Stage 2: Deterministic Algorithm
# ---------------------
#
# CSV format (per spec):
#   col[0]      = dataTime       → timestamp
#   col[1]      = alarmState     → IGNORED
#   col[2]      = hr             → used
#   col[3]      = o2sat          → IGNORED
#   col[4..128] = accel*125      → 125 milli-g readings
#
# No ground-truth labels used anywhere in this file.
# The algorithm scores each window purely from sensor signals (accel + HR).
#
# Detects four seizure-pattern categories used in this project:
#   Tonic-clonic : high accel std + HR ≥ 100 BPM
#   Tonic        : muscle rigidity → elevated jerk_mean + HR spike
#   Myoclonic    : isolated high jerk_max burst
#   Focal        : near-normal accel, sustained HR ≥ 105 BPM for ≥ 6 windows
#
# Thresholds calibrated from a development dataset of wearable sessions.


import numpy as np
import glob
import os


# ─────────────────────────────────────────────
# THRESHOLDS
# ─────────────────────────────────────────────

# Accel (milli-g) — resting ≈ 1000 mg
STD_WARN_MG        = 80.0    # std where concern begins
STD_ALARM_MG       = 150.0   # strong motor activity
SPIKE_THRESH_MG    = 1200.0  # single sample > 1.2g
SPIKE_COUNT_ALARM  = 8       # spikes per window for full spike score
JERK_MYOCLONIC_MG  = 400.0   # jerk_max threshold — myoclonic burst
JERK_MEAN_ELEV     = 25.0    # jerk_mean above this = elevated roughness
                               # (normal mean=15.9, hard seizures mean=30.5)

# HR — target profile used during threshold development
# Normal resting: mean=93, p75=104, p95=113
# Seizure:        mean=107, p25=103, p75=115
HR_ELEVATED        = 100     # BPM: elevated
HR_HIGH            = 115     # BPM: strongly elevated
HR_FOCAL_THRESH    = 105     # BPM: sustained above this → focal path
HR_FOCAL_WINDOWS   = 6       # consecutive windows with HR ≥ focal thresh

# Scoring weights (sum to 1.0)
W_ACCEL  = 0.40
W_JERK   = 0.20
W_HR     = 0.40
MOTOR_PEAK_GAIN = 0.85     # preserve short, high-magnitude motor bursts

# State machine
WARN_THRESH        = 0.35
ALARM_THRESH       = 0.50
MIN_ALARM_WINDOWS  = 3       # ~15 seconds sustained → ALARM
CLEAR_THRESH       = 0.20
INSTANT_WARN_THRESH  = 0.55
INSTANT_ALARM_THRESH = 0.80

EMA_ALPHA          = 0.45


# ─────────────────────────────────────────────
# 1.  DATA LOADER
# ─────────────────────────────────────────────

def load_csv(filepath: str) -> list:
    """
    Parse one session CSV.

    Reads:   col[0] timestamp, col[2] hr, col[4..128] accel (125 values)
    Ignores: col[1] alarmState, col[3] o2sat

    Returns list of dicts: ts, hr, accel_mg
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
                    # col[1] alarmState → not read
                    'hr':       float(parts[2].strip()),
                    # col[3] o2sat → not read
                    'accel_mg': np.array([float(x) for x in parts[4:] if x.strip()]),
                })
            except (ValueError, IndexError):
                continue
    rows.sort(key=lambda r: r['ts'])
    return rows


def load_all_csvs(filepaths: list) -> list:
    all_rows = []
    for fp in filepaths:
        all_rows.extend(load_csv(fp))
    all_rows.sort(key=lambda r: r['ts'])
    return all_rows


def impute_hr(rows: list, fallback_bpm: float = 90.0) -> list:
    """Forward-fill missing HR (hr ≤ 0 = sensor dropout). In-place."""
    last_valid = fallback_bpm
    for r in rows:
        if r['hr'] <= 0:
            r['hr'] = last_valid
        else:
            last_valid = r['hr']
    return rows


# ─────────────────────────────────────────────
# 2.  PER-WINDOW FEATURES
# ─────────────────────────────────────────────

def window_features(row: dict) -> dict:
    """
    Compute accel + HR features for one window.

    Includes 1-second burst maxima so short startle-like events are less likely
    to be washed out by the full 5-second window average.
    """
    a  = row['accel_mg']
    hr = row['hr']

    jerk      = np.diff(a)
    jerk_abs  = np.abs(jerk)

    centered  = a - a.mean()
    chunk_size = max(1, min(25, len(a)))
    chunk_stds = []
    chunk_spike_counts = []
    chunk_jerk_means = []
    chunk_jerk_maxes = []
    for start in range(0, len(a), chunk_size):
        chunk = a[start : start + chunk_size]
        if len(chunk) == 0:
            continue
        chunk_stds.append(float(np.std(chunk)))
        chunk_spike_counts.append(int(np.sum(chunk > SPIKE_THRESH_MG)))
        chunk_jerk = np.diff(chunk)
        chunk_jerk_abs = np.abs(chunk_jerk)
        chunk_jerk_means.append(
            float(chunk_jerk_abs.mean()) if len(chunk_jerk_abs) else 0.0
        )
        chunk_jerk_maxes.append(
            float(chunk_jerk_abs.max()) if len(chunk_jerk_abs) else 0.0
        )

    return dict(
        std_mg      = float(np.std(a)),
        spike_count = int(np.sum(a > SPIKE_THRESH_MG)),
        jerk_max    = float(jerk_abs.max()) if len(jerk) else 0.0,
        jerk_mean   = float(jerk_abs.mean()) if len(jerk) else 0.0,
        burst_std_1s_max = max(chunk_stds) if chunk_stds else 0.0,
        burst_spike_count_1s_max = max(chunk_spike_counts) if chunk_spike_counts else 0,
        burst_jerk_mean_1s_max = max(chunk_jerk_means) if chunk_jerk_means else 0.0,
        burst_jerk_max_1s_max = max(chunk_jerk_maxes) if chunk_jerk_maxes else 0.0,
        zero_cross  = int(np.sum(np.diff(np.sign(centered)) != 0)),
        hr          = hr,
    )


# ─────────────────────────────────────────────
# 3.  COMPONENT SCORES  [0, 1]
# ─────────────────────────────────────────────

def accel_component(std_mg: float, spike_count: int) -> float:
    """Motor intensity — tonic-clonic / strong tonic."""
    std_part   = min(max(std_mg - STD_WARN_MG, 0.0) / (STD_ALARM_MG - STD_WARN_MG), 1.0)
    spike_part = min(spike_count / SPIKE_COUNT_ALARM, 1.0)
    return 0.60 * std_part + 0.40 * spike_part


def jerk_component(jerk_max: float, jerk_mean: float) -> float:
    """Myoclonic bursts + hard focal/tonic cases via elevated jerk_mean."""
    myo_part  = min(max(jerk_max - JERK_MYOCLONIC_MG, 0.0) / (JERK_MYOCLONIC_MG * 2), 1.0)
    mean_part = min(max(jerk_mean - JERK_MEAN_ELEV, 0.0) / JERK_MEAN_ELEV, 1.0)
    return max(myo_part, mean_part * 0.6)


def hr_component(hr: float, focal_flag: bool) -> float:
    """
    HR score — equal weight to accel (W_HR = 0.40).
    HR is always valid (imputed) so no neutral fallback needed.
    HR < 100: small negative contribution (suppresses false alarms during play).
    HR ≥ 100: scales up to 1.0 at HR_HIGH (115 BPM).
    Focal path: boosted when sustained tachycardia detected.
    """
    if hr < HR_ELEVATED:
        return max((hr - HR_ELEVATED) / HR_ELEVATED * 0.1, -0.05)
    base = min((hr - HR_ELEVATED) / (HR_HIGH - HR_ELEVATED), 1.0)
    if focal_flag:
        return min(base * 1.4, 1.0)
    return base


def compute_score_details(row: dict, focal_flag: bool) -> dict:
    """
    Compute component scores for one window.

    The weighted score still captures sustained multi-signal events, while the
    motor peak term preserves short, isolated bursts that would otherwise be
    diluted by averaging across accel + HR.
    """
    wf = window_features(row)
    accel_score = accel_component(wf['std_mg'], wf['spike_count'])
    jerk_score  = jerk_component(wf['jerk_max'], wf['jerk_mean'])
    hr_score    = hr_component(wf['hr'], focal_flag)
    weighted_score = (
        W_ACCEL * accel_score
        + W_JERK * jerk_score
        + W_HR * hr_score
    )
    motor_peak_score = max(accel_score, jerk_score)
    raw_score = float(np.clip(max(weighted_score, MOTOR_PEAK_GAIN * motor_peak_score), 0.0, 1.0))

    return dict(
        **wf,
        accel_score=float(accel_score),
        jerk_score=float(jerk_score),
        hr_score=float(hr_score),
        motor_peak_score=float(motor_peak_score),
        weighted_score=float(weighted_score),
        raw_score=raw_score,
    )


def compute_raw_score(row: dict, focal_flag: bool) -> float:
    """Combined raw anomaly score [0, 1]."""
    return compute_score_details(row, focal_flag)['raw_score']


# ─────────────────────────────────────────────
# 4.  FOCAL PATH  (sustained HR)
# ─────────────────────────────────────────────

def compute_focal_flags(rows: list) -> list:
    """
    True for window i if HR ≥ HR_FOCAL_THRESH in ALL of the last
    HR_FOCAL_WINDOWS consecutive windows (~30 seconds of sustained tachycardia).
    Detects focal seizures where accel looks near-normal.
    """
    n       = len(rows)
    hr_high = [r['hr'] >= HR_FOCAL_THRESH for r in rows]
    flags   = [False] * n
    for i in range(n):
        start  = max(0, i - HR_FOCAL_WINDOWS + 1)
        window = hr_high[start : i + 1]
        if len(window) >= HR_FOCAL_WINDOWS and all(window):
            flags[i] = True
    return flags


# ─────────────────────────────────────────────
# 5.  EMA SMOOTHER
# ─────────────────────────────────────────────

def smooth_scores(raw: np.ndarray, alpha: float = EMA_ALPHA) -> np.ndarray:
    """Exponential moving average — requires sustained anomaly to accumulate."""
    ema    = np.empty_like(raw)
    ema[0] = raw[0]
    for i in range(1, len(raw)):
        ema[i] = alpha * raw[i] + (1.0 - alpha) * ema[i - 1]
    return np.maximum(raw, ema)


# ─────────────────────────────────────────────
# 6.  STATE MACHINE
# ─────────────────────────────────────────────

def run_state_machine(smoothed: np.ndarray,
                      warn_thresh:    float = WARN_THRESH,
                      alarm_thresh:   float = ALARM_THRESH,
                      min_alarm_wins: int   = MIN_ALARM_WINDOWS,
                      clear_thresh:   float = CLEAR_THRESH,
                      instant_scores: np.ndarray = None,
                      instant_warn_thresh: float = INSTANT_WARN_THRESH,
                      instant_alarm_thresh: float = INSTANT_ALARM_THRESH) -> np.ndarray:
    """
    3-state FSM: 0=OK, 1=WARNING, 2=ALARM
    Requires MIN_ALARM_WINDOWS consecutive windows above alarm_thresh to fire,
    but can immediately raise WARNING/ALARM for very strong one-window bursts.
    """
    states = np.zeros(len(smoothed), dtype=int)
    state  = 0
    count  = 0
    for i, s in enumerate(smoothed):
        instant = s if instant_scores is None else instant_scores[i]

        if state == 0:
            if instant >= instant_alarm_thresh:
                state, count = 2, 0
            elif instant >= instant_warn_thresh or s >= warn_thresh:
                state, count = 1, 0
        elif state == 1:
            if instant >= instant_alarm_thresh:
                state, count = 2, 0
            elif s < clear_thresh and instant < instant_warn_thresh:
                state, count = 0, 0
            elif s >= alarm_thresh:
                count += 1
                if count >= min_alarm_wins:
                    state = 2
            else:
                count = 0
        elif state == 2:
            if s < clear_thresh and instant < instant_warn_thresh:
                state, count = 0, 0
        states[i] = state
    return states


# ─────────────────────────────────────────────
# 7.  FULL PIPELINE
# ─────────────────────────────────────────────

def run_pipeline(rows: list,
                 warn_thresh:    float = WARN_THRESH,
                 alarm_thresh:   float = ALARM_THRESH,
                 min_alarm_wins: int   = MIN_ALARM_WINDOWS,
                 clear_thresh:   float = CLEAR_THRESH,
                 ema_alpha:      float = EMA_ALPHA) -> list:
    """
    Run all deterministic stages on a list of window rows.
    impute_hr() must be called before this.

    Adds to each row dict in-place:
        accel_std    – intra-window std (mg)
        spike_count  – samples above 1200 mg
        jerk_max     – largest single jerk (mg)
        jerk_mean    – mean jerk magnitude (mg)
        focal_flag   – True if in a sustained-HR-elevation run
        raw_score    – per-window combined score [0, 1]
        smooth_score – EMA-smoothed score [0, 1]
        pred_state   – 0=OK, 1=WARNING, 2=ALARM
    """
    focal_flags    = compute_focal_flags(rows)
    score_details  = [compute_score_details(r, focal_flags[i])
                      for i, r in enumerate(rows)]
    raw_arr        = np.array([d['raw_score'] for d in score_details], dtype=float)
    smoothed       = smooth_scores(raw_arr, alpha=ema_alpha)
    states      = run_state_machine(smoothed, warn_thresh=warn_thresh,
                                    alarm_thresh=alarm_thresh,
                                    min_alarm_wins=min_alarm_wins,
                                    clear_thresh=clear_thresh,
                                    instant_scores=raw_arr)
    for i, r in enumerate(rows):
        details = score_details[i]
        r['accel_std']       = details['std_mg']
        r['spike_count']     = details['spike_count']
        r['jerk_max']        = details['jerk_max']
        r['jerk_mean']       = details['jerk_mean']
        r['burst_std_1s_max'] = details['burst_std_1s_max']
        r['burst_spike_count_1s_max'] = details['burst_spike_count_1s_max']
        r['burst_jerk_mean_1s_max'] = details['burst_jerk_mean_1s_max']
        r['burst_jerk_max_1s_max'] = details['burst_jerk_max_1s_max']
        r['focal_flag']      = focal_flags[i]
        r['accel_score']     = details['accel_score']
        r['jerk_score']      = details['jerk_score']
        r['hr_score']        = details['hr_score']
        r['motor_peak_score'] = details['motor_peak_score']
        r['weighted_score']  = details['weighted_score']
        r['raw_score']       = float(raw_arr[i])
        r['smooth_score']    = float(smoothed[i])
        r['pred_state']      = int(states[i])
    return rows


# ─────────────────────────────────────────────
# 8.  SEIZURE TYPE CLASSIFIER
# ─────────────────────────────────────────────

def classify_seizure_type(row: dict) -> str:
    """
    Heuristic label for a flagged window.
    Used for caregiver logs only — not part of alarm logic.
    """
    if row.get('pred_state', 0) == 0:
        return ''
    std  = row.get('accel_std', 0)
    hr   = row.get('hr', 0)
    jmax = row.get('jerk_max', 0)
    jmn  = row.get('jerk_mean', 0)
    foc  = row.get('focal_flag', False)

    if std >= 120 and hr >= 100:
        return 'tonic-clonic'
    if jmax >= JERK_MYOCLONIC_MG and std < 80:
        return 'myoclonic'
    if jmn >= JERK_MEAN_ELEV and hr >= HR_ELEVATED and std < 50 and jmax < JERK_MYOCLONIC_MG:
        return 'tonic'
    if foc and std < 60:
        return 'focal'
    if std >= 50:
        return 'tonic-clonic'
    return 'focal'


# ─────────────────────────────────────────────
# 9.  QUICK DEMO
# ─────────────────────────────────────────────

if __name__ == '__main__':
    csv_files = sorted(glob.glob('data/*-data.csv'))
    if not csv_files:
        print("No CSVs found in ./data/")
        exit()

    for fp in csv_files:
        name = os.path.basename(fp)
        rows = load_csv(fp)
        rows = impute_hr(rows)
        rows = run_pipeline(rows)

        print(f"\n── {name}  ({len(rows)} windows) " + "─" * 30)
        print(f"{'Timestamp':<22} {'HR':>5} {'Std':>7} {'JrkM':>6} "
              f"{'Foc':>4} {'Raw':>6} {'Smth':>6} {'State':>6}  Type")
        print("─" * 78)
        for r in rows:
            state = {0:'OK', 1:'WARN', 2:'ALARM'}.get(r['pred_state'], '?')
            flag  = ' ◄' if r['pred_state'] >= 1 else ''
            stype = classify_seizure_type(r)
            foc   = 'Y' if r['focal_flag'] else '-'
            print(f"{r['ts']:<22} {r['hr']:>5.0f} {r['accel_std']:>7.1f} "
                  f"{r['jerk_mean']:>6.1f} {foc:>4} {r['raw_score']:>6.2f} "
                  f"{r['smooth_score']:>6.3f} {state:>6}{flag}  {stype}")
