# ---------------------
# main_pipeline.py  —  Full seizure detection pipeline
# ---------------------
#
# CSV format (per spec):
#   col[0]      = dataTime       → used (timestamp)
#   col[1]      = alarmState     → IGNORED
#   col[2]      = hr             → used (BPM)
#   col[3]      = o2sat          → IGNORED (always -1)
#   col[4..128] = accel*125      → used (125 milli-g readings)
#
# The detector itself runs from accel + HR sensor signals.
# An optional seizure log can be used to keep seizure-adjacent windows out of
# baseline training and to compare detections against reference event logs.
#
# Stage 1: Autoencoder learns subject-specific normal accel + HR patterns
# Stage 2: Deterministic rules across 4 seizure types
# Stage 3: kNN anomaly scoring in latent space + ensemble with Stage 2
# Stage 4: Supervised calibrator learns from seizure-log labels
#
# Install: pip install scikit-learn scipy numpy
# Usage:   python main_pipeline.py
# Data:    place CSV files in ./data/
# Logs:    optionally add seizure timestamps to ./seizure_log.csv


import contextlib
import csv
import glob
import hashlib
import json
import os
import sys
from bisect import bisect_left, bisect_right
from datetime import datetime, timedelta

import numpy as np

from part1_autoencoder import (
    load_csv, load_all_csvs, impute_hr,
    compute_features, normalize_features,
    train_autoencoder, create_latent_bank,
)
from part2_deterministic import (
    run_pipeline, run_state_machine, classify_seizure_type,
    WARN_THRESH, ALARM_THRESH, MIN_ALARM_WINDOWS, CLEAR_THRESH, STD_ALARM_MG,
)
from part3_knn_detector import (
    LatentKNNScorer, ensemble_score, run_ml_inference,
)
from part4_supervised_calibrator import (
    fit_supervised_calibrator,
)


DATA_DIR       = 'data'
SEIZURE_LOG    = 'seizure_log.csv'
RESULTS_DIR    = 'results'
STAGE4_TUNING_CACHE = 'stage4_tuning_cache.json'
TS_FORMAT      = '%Y-%m-%d %H:%M:%S'
AE_MAX_ITER    = 300
KNN_K          = 5
KNN_PERCENTILE = 95
EXCLUDE_MARGIN = 60
STAGE4_WARN_EVENT_TARGET   = 0.80
STAGE4_ALARM_EVENT_TARGET  = 0.70
STAGE4_TC_ALARM_EVENT_TARGET = 1.00
STAGE4_MAX_NORMAL_FLAG_RATE = 0.15
STAGE4_MAX_NORMAL_ALARM_RATE = 0.005
STAGE4_SOFT_NORMAL_ALARM_RATE = 0.07
STAGE4_MIN_WARN_ALARM_GAP  = 0.08
STAGE4_THRESHOLD_GRID_SIZE = 10
STAGE4_MIN_ALARM_WINDOWS   = max(MIN_ALARM_WINDOWS + 1, 4)
STAGE4_CLEAR_RATIO         = 0.72
STAGE4_INSTANT_WARN_OFFSET = 0.18
STAGE4_INSTANT_ALARM_OFFSET = 0.22
STAGE4_INSTANT_WARN_CAP    = 0.97
STAGE4_INSTANT_ALARM_CAP   = 0.995
STAGE4_PROGRESS_EVERY      = 25
STAGE4_WARN_CANDIDATE_MIN  = 0.05
STAGE4_WARN_QUANTILE_MIN   = 0.35
STAGE4_ALARM_CANDIDATE_MIN = 0.18
STAGE4_ALARM_QUANTILE_MIN  = 0.30
STAGE4_MIN_ALARM_WINDOWS_CANDIDATES = (2, 3)
STAGE4_INSTANT_WARN_OFFSET_CANDIDATES = (0.12, 0.18)
STAGE4_INSTANT_ALARM_OFFSET_CANDIDATES = (0.12, 0.18)
STAGE4_GATE_STRONG_SUP_MARGIN = 0.16
STAGE4_GATE_ENSEMBLE_THRESH = 0.35
STAGE4_GATE_SMOOTH_THRESH = 0.32
STAGE4_GATE_ML_THRESH = 0.58
STAGE4_GATE_MOTOR_THRESH = 0.45
STAGE4_TC_GATE_STD_THRESH = 110.0
STAGE4_TC_GATE_HR_THRESH = 95.0
STAGE4_TC_GATE_SMOOTH_THRESH = 0.42
STAGE4_TC_GATE_ENSEMBLE_THRESH = 0.38
STAGE4_TC_GATE_ACCEL_THRESH = 0.45
STAGE4_TC_GATE_SPIKE_COUNT = 4
STAGE4_TC_OVERRIDE_ENSEMBLE_THRESH = 0.52
STAGE4_TC_OVERRIDE_SMOOTH_THRESH = 0.52
STAGE4_TC_OVERRIDE_ML_THRESH = 0.66
STAGE4_TC_OVERRIDE_MOTOR_THRESH = 0.62
STAGE4_TC_OVERRIDE_SPIKE_COUNT = 6
STAGE4_TC_OVERRIDE_SUP_MARGIN = 0.08
STAGE4_TC_HOLD_MIN_SUPPORT = 3
STAGE4_TC_PROMOTION_MIN_SUPPORT = 4
STAGE4_TC_LOW_SUP_FLOOR = 0.20
STAGE4_TC_MIN_RUN_HOLD = 2
STAGE4_TC_MIN_RUN_PROMOTION = 3
STAGE4_TC_LOW_HR_SUPPRESS = 85.0
STAGE4_TC_LOW_HR_SUP_MAX = 0.45
STAGE4_TC_LOW_HR_EXTRA_SUPPORT = 1
STAGE4_TC_LOW_HR_EXTRA_RUN = 1
STAGE4_STARTLE_RESCUE_SUP_FLOOR = 0.52
STAGE4_STARTLE_RESCUE_ML_THRESH = 0.36
STAGE4_STARTLE_RESCUE_ENSEMBLE_THRESH = 0.26
STAGE4_STARTLE_RESCUE_MOTOR_THRESH = 0.34
STAGE4_STARTLE_RESCUE_BURST_STD_THRESH = 75.0
STAGE4_STARTLE_RESCUE_BURST_JERK_MEAN_THRESH = 24.0
STAGE4_STARTLE_RESCUE_BURST_JERK_MAX_THRESH = 260.0
STAGE4_STARTLE_SURPRISE_RATIO_THRESH = 1.45
STAGE4_STARTLE_PEAK_SUP_FLOOR = 0.22
STAGE4_STARTLE_PEAK_ML_THRESH = 0.24
STAGE4_STARTLE_PEAK_ENSEMBLE_THRESH = 0.18
STAGE4_STARTLE_PEAK_BURST_STD_THRESH = 45.0
STAGE4_STARTLE_PEAK_BURST_JERK_MEAN_THRESH = 20.0
STAGE4_STARTLE_PEAK_SURPRISE_THRESH = 1.80
STAGE4_MAX_FALSE_ALARM_EPISODES_PER_HOUR = 2.0
CAREGIVER_ALARM_COOLDOWN_SECONDS = 90
CAREGIVER_MIN_ALARM_EPISODE_WINDOWS = 2
CAREGIVER_SHORT_ALARM_STRONG_SUP = 0.82
CAREGIVER_SHORT_ALARM_ML_THRESH = 0.42
CAREGIVER_SHORT_ALARM_ENSEMBLE_THRESH = 0.42
CAREGIVER_SHORT_ALARM_BURST_STD_THRESH = 115.0
CAREGIVER_SHORT_ALARM_BURST_JERK_MAX_THRESH = 320.0
CAREGIVER_LONG_CONFIRM_WINDOWS = 8
CAREGIVER_LONG_CONFIRM_SECONDS = 180
CAREGIVER_LONG_CONFIRM_SUP = 0.58
CAREGIVER_LONG_CONFIRM_ML = 0.48
CAREGIVER_LONG_CONFIRM_BURST_WINDOWS = 2
CAREGIVER_LONG_CONFIRM_FOCAL_WINDOWS = 2
CAREGIVER_WARN_PROMOTION_MIN_WINDOWS = 4
CAREGIVER_WARN_PROMOTION_MAX_WINDOWS = 6
CAREGIVER_WARN_PROMOTION_MAX_SECONDS = 45
CAREGIVER_WARN_PROMOTION_SUP = 0.30
CAREGIVER_WARN_PROMOTION_STRONG_SUP = 0.44
CAREGIVER_WARN_PROMOTION_ML = 0.34
CAREGIVER_WARN_PROMOTION_ENSEMBLE = 0.24
CAREGIVER_WARN_PROMOTION_BURST_WINDOWS = 3
CAREGIVER_WARN_PROMOTION_FOCAL_WINDOWS = 4
CAREGIVER_WARN_PROMOTION_WINDOW_SUP = 0.28
CAREGIVER_WARN_PROMOTION_MIN_PEAK_SEP_SECONDS = 20
CAREGIVER_WARN_PROMOTION_MAX_PER_CLUSTER = 2
CAREGIVER_STARTLE_CLUSTER_WINDOWS = 2
CAREGIVER_STARTLE_SHORT_CLUSTER_SUP = 0.22
CAREGIVER_STARTLE_SHORT_CLUSTER_ML = 0.24
CAREGIVER_SINGLE_WINDOW_STARTLE_SUP = 0.30
CAREGIVER_SINGLE_WINDOW_STARTLE_ML = 0.30
STARTLE_BASELINE_HISTORY_WINDOWS = 12
STARTLE_BASELINE_MIN_HISTORY = 4
STARTLE_SURPRISE_RATIO_CAP = 6.0
REPORT_VERBOSE_ROWS = False
REPORT_CONTEXT_WINDOWS = 1
REPORT_MAX_ROWS_PER_SESSION = 180
USE_STAGE4_TUNING_CACHE = True
STAGE4_TUNING_CACHE_VERSION = '2026-04-07-startle-surprise-v5'


class Tee:
    """Mirror printed output to both the terminal and a report file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def build_report_path() -> str:
    """Create a unique timestamped text report path under ./results/."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    base_path = os.path.join(RESULTS_DIR, f'pipeline_report_{stamp}')
    candidate = f'{base_path}.txt'
    suffix = 1
    while os.path.exists(candidate):
        candidate = f'{base_path}_{suffix}.txt'
        suffix += 1
    return candidate


def build_stage4_tuning_cache_path() -> str:
    """Stable cache path used to reuse Stage 4 holdout tuning across reruns."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return os.path.join(RESULTS_DIR, STAGE4_TUNING_CACHE)


def build_stage4_tuning_signature(csv_files: list,
                                  seizure_log_path: str) -> str:
    """Hash the data/log inputs plus tuning-version settings for cache reuse."""
    file_info = []
    for path in sorted(csv_files):
        try:
            stat = os.stat(path)
            file_info.append({
                'path': os.path.basename(path),
                'size': int(stat.st_size),
                'mtime': int(stat.st_mtime),
            })
        except OSError:
            file_info.append({
                'path': os.path.basename(path),
                'size': -1,
                'mtime': -1,
            })

    log_info = {
        'present': False,
        'path': os.path.basename(seizure_log_path),
        'size': -1,
        'mtime': -1,
    }
    if os.path.exists(seizure_log_path):
        try:
            stat = os.stat(seizure_log_path)
            log_info.update({
                'present': True,
                'size': int(stat.st_size),
                'mtime': int(stat.st_mtime),
            })
        except OSError:
            pass

    settings = {
        'cache_version': STAGE4_TUNING_CACHE_VERSION,
        'warn_target': STAGE4_WARN_EVENT_TARGET,
        'alarm_target': STAGE4_ALARM_EVENT_TARGET,
        'tc_target': STAGE4_TC_ALARM_EVENT_TARGET,
        'max_normal_alarm_rate': STAGE4_MAX_NORMAL_ALARM_RATE,
        'max_false_alarm_episodes_per_hour': STAGE4_MAX_FALSE_ALARM_EPISODES_PER_HOUR,
        'grid_size': STAGE4_THRESHOLD_GRID_SIZE,
        'min_alarm_windows_candidates': list(STAGE4_MIN_ALARM_WINDOWS_CANDIDATES),
        'instant_warn_offset_candidates': list(STAGE4_INSTANT_WARN_OFFSET_CANDIDATES),
        'instant_alarm_offset_candidates': list(STAGE4_INSTANT_ALARM_OFFSET_CANDIDATES),
    }
    payload = {
        'files': file_info,
        'log': log_info,
        'settings': settings,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode('utf-8')
    ).hexdigest()


def load_stage4_tuning_cache(signature: str) -> dict | None:
    """Load previously tuned Stage 4 thresholds when inputs have not changed."""
    cache_path = build_stage4_tuning_cache_path()
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    if payload.get('signature') != signature:
        return None

    tuning = payload.get('tuning')
    if not isinstance(tuning, dict):
        return None

    tuning = tuning.copy()
    state_params = tuning.get('state_params')
    if not isinstance(state_params, dict):
        state_params = derive_stage4_state_params(
            float(tuning.get('warn_threshold', WARN_THRESH)),
            float(tuning.get('alarm_threshold', ALARM_THRESH)),
        )
    tuning['state_params'] = state_params
    tuning['mode'] = f"cached-{tuning.get('mode', 'holdout')}"
    return tuning


def save_stage4_tuning_cache(signature: str, tuning: dict) -> None:
    """Persist the last successful Stage 4 tuning result for fast reruns."""
    cache_path = build_stage4_tuning_cache_path()
    serializable = {
        key: value
        for key, value in tuning.items()
        if isinstance(value, (int, float, str, bool, list, dict)) or value is None
    }
    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump({
                'signature': signature,
                'saved_at': datetime.now().strftime(TS_FORMAT),
                'tuning': serializable,
            }, f, indent=2)
    except OSError:
        pass


def parse_timestamp(ts: str) -> datetime:
    """Parse sensor/log timestamps in the same 24-hour format as the CSV data."""
    return datetime.strptime(ts.strip(), TS_FORMAT)


def event_time(event):
    """Return the datetime for a seizure-log event record or raw timestamp."""
    return event['time'] if isinstance(event, dict) else event


def event_note(event) -> str:
    """Return the caregiver note stored on a seizure-log event, if present."""
    return str(event.get('note', '')).strip() if isinstance(event, dict) else ''


def event_type(event) -> str:
    """Return the normalized subtype stored on a seizure-log event."""
    if isinstance(event, dict):
        return str(event.get('event_type', 'seizure')).strip() or 'seizure'
    return 'seizure'


def event_is_startle(event) -> bool:
    """True when the seizure-log note identifies the event as a startle."""
    return bool(event.get('is_startle', False)) if isinstance(event, dict) else False


def event_is_tonic_clonic(event) -> bool:
    """True when the seizure-log subtype is tonic-clonic."""
    return str(event_type(event)).strip().lower() == 'tonic-clonic'


def normalize_event_note(note: str) -> dict:
    """Normalize the caregiver note into a coarse seizure subtype."""
    clean_note = str(note or '').strip()
    lower_note = clean_note.lower()
    is_startle = 'startle' in lower_note

    if is_startle:
        normalized_type = 'startle'
    elif 'tc' in lower_note or 'tonic' in lower_note or 'clonic' in lower_note:
        normalized_type = 'tonic-clonic'
    elif 'focal' in lower_note:
        normalized_type = 'focal'
    else:
        normalized_type = 'seizure'

    return {
        'note': clean_note,
        'event_type': normalized_type,
        'is_startle': is_startle,
    }


def load_seizure_log(filepath: str) -> list:
    """
    Read logged seizure timestamps from CSV.

    Expected format:
        seizure_time,note
        YYYY-MM-DD HH:MM:SS,event note

    The first column is the timestamp. The second column, when present, is
    kept as an event note so we can track seizure subtypes like "startle".
    """
    seizure_events = []
    with open(filepath, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue

            raw_ts = row[0].strip()
            if not raw_ts or raw_ts.startswith('#'):
                continue

            try:
                parsed = normalize_event_note(row[1] if len(row) > 1 else '')
                seizure_events.append({
                    'time': parse_timestamp(raw_ts),
                    **parsed,
                })
            except ValueError:
                if raw_ts.lower() in {'seizure_time', 'timestamp', 'event_time', 'ts'}:
                    continue
                print(f"  [WARN] Skipping seizure-log row with invalid timestamp: {raw_ts!r}")

    seizure_events.sort(key=event_time)
    return seizure_events


def label_rows_from_seizure_log(rows: list,
                                seizure_events: list,
                                margin_seconds: int = EXCLUDE_MARGIN):
    """
    Label each row as either seizure-highlighted or normal behavior.

    A row is treated as seizure-highlighted when it falls within
    +/- margin_seconds of any logged seizure timestamp. Every other parsed row is
    treated as normal behavior.
    """
    if not seizure_events:
        for row in rows:
            row['log_is_seizure'] = False
            row['log_is_normal'] = True
            row['log_label'] = 'normal'
            row['log_event_note'] = ''
            row['log_event_type'] = 'normal'
            row['log_is_startle'] = False
        return 0, 0

    margin = timedelta(seconds=margin_seconds)
    seizure_window_count = 0
    unparsed_count = 0
    seizure_idx = 0

    for row in rows:
        try:
            row_ts = parse_timestamp(row['ts'])
        except (KeyError, ValueError):
            row['log_is_seizure'] = False
            row['log_is_normal'] = False
            row['log_label'] = 'unparsed'
            row['log_event_note'] = ''
            row['log_event_type'] = 'unparsed'
            row['log_is_startle'] = False
            unparsed_count += 1
            continue

        while seizure_idx < len(seizure_events) and event_time(seizure_events[seizure_idx]) < row_ts - margin:
            seizure_idx += 1

        is_near_seizure = False
        matched_event = None
        matched_abs_seconds = None
        check_idx = max(0, seizure_idx - 1)
        while check_idx < len(seizure_events):
            current_event = seizure_events[check_idx]
            delta = event_time(current_event) - row_ts
            if delta > margin:
                break
            abs_seconds = abs(delta.total_seconds())
            if abs_seconds <= margin_seconds:
                is_near_seizure = True
                if matched_abs_seconds is None or abs_seconds < matched_abs_seconds:
                    matched_event = current_event
                    matched_abs_seconds = abs_seconds
            check_idx += 1

        row['log_is_seizure'] = is_near_seizure
        row['log_is_normal'] = not is_near_seizure
        row['log_label'] = 'seizure' if is_near_seizure else 'normal'
        row['log_event_note'] = event_note(matched_event) if matched_event else ''
        row['log_event_type'] = event_type(matched_event) if matched_event else 'normal'
        row['log_is_startle'] = event_is_startle(matched_event) if matched_event else False
        seizure_window_count += int(is_near_seizure)

    return seizure_window_count, unparsed_count


def normal_reference_rows(rows: list) -> list:
    """Return only rows currently labelled as normal behavior."""
    return [row for row in rows if row.get('log_is_normal', True)]


def exclude_logged_windows(rows: list,
                           seizure_events: list,
                           margin_seconds: int = EXCLUDE_MARGIN):
    """
    Remove rows within +/- margin_seconds of any logged seizure timestamp.

    Returns (filtered_rows, excluded_count, unparsed_count).
    """
    seizure_window_count, unparsed_count = label_rows_from_seizure_log(
        rows,
        seizure_events,
        margin_seconds=margin_seconds,
    )
    filtered_rows = normal_reference_rows(rows)
    return filtered_rows, seizure_window_count, unparsed_count


def summarize_logged_events(rows: list,
                            seizure_events: list,
                            state_key: str = 'final_state',
                            match_margin_seconds: int = EXCLUDE_MARGIN) -> list:
    """
    Compare detected window states against logged seizure times for one session.

    A logged event counts as matched if any window within +/- match_margin_seconds
    reaches WARNING or ALARM. ALARM matching is counted separately.
    """
    if not rows or not seizure_events:
        return []

    parsed_rows = []
    for row in rows:
        try:
            parsed_rows.append((parse_timestamp(row['ts']), int(row.get(state_key, 0))))
        except (KeyError, TypeError, ValueError):
            continue

    if not parsed_rows:
        return []

    margin = timedelta(seconds=match_margin_seconds)
    start_ts = parsed_rows[0][0] - margin
    end_ts = parsed_rows[-1][0] + margin
    relevant_events = [
        event for event in seizure_events
        if start_ts <= event_time(event) <= end_ts
    ]

    matches = []
    for event in relevant_events:
        event_ts = event_time(event)
        covered = False
        warn_match = False
        alarm_match = False
        for row_ts, state in parsed_rows:
            if abs((row_ts - event_ts).total_seconds()) <= match_margin_seconds:
                covered = True
                if state >= 1:
                    warn_match = True
                if state >= 2:
                    alarm_match = True
                if warn_match and alarm_match:
                    break
        matches.append({
            'time': event_ts,
            'note': event_note(event),
            'event_type': event_type(event),
            'is_startle': event_is_startle(event),
            'covered': covered,
            'warn_match': warn_match,
            'alarm_match': alarm_match,
        })

    return matches


def build_state_clusters(rows: list,
                         min_state: int,
                         state_key: str = 'final_state',
                         states: np.ndarray = None,
                         supervised_scores: np.ndarray = None,
                         cooldown_seconds: int = CAREGIVER_ALARM_COOLDOWN_SECONDS) -> list:
    """Group nearby >=min_state windows into caregiver-facing clusters."""
    if not rows:
        return []

    parsed_rows = []
    for idx, row in enumerate(rows):
        try:
            state = int(states[idx]) if states is not None else int(row.get(state_key, 0))
            sup_score = (
                float(supervised_scores[idx])
                if supervised_scores is not None else
                float(row.get('supervised_score', 0.0))
            )
            parsed_rows.append({
                'row_idx': idx,
                'time': parse_timestamp(row['ts']),
                'state': state,
                'log_is_normal': bool(row.get('log_is_normal')),
                'log_is_seizure': bool(row.get('log_is_seizure')),
                'supervised_score': sup_score,
                'legacy_state': int(row.get('legacy_state', 0)),
                'ensemble_score': float(row.get('ensemble_score', 0.0)),
                'ml_score': float(row.get('ml_score', 0.0)),
                'smooth_score': float(row.get('smooth_score', 0.0)),
                'burst_std_1s_max': float(row.get('burst_std_1s_max', row.get('accel_std', 0.0))),
                'burst_jerk_mean_1s_max': float(row.get('burst_jerk_mean_1s_max', row.get('jerk_mean', 0.0))),
                'burst_jerk_max_1s_max': float(row.get('burst_jerk_max_1s_max', row.get('jerk_max', 0.0))),
                'burst_std_surprise': float(row.get('burst_std_surprise', 1.0)),
                'burst_jerk_mean_surprise': float(row.get('burst_jerk_mean_surprise', 1.0)),
                'burst_jerk_max_surprise': float(row.get('burst_jerk_max_surprise', 1.0)),
                'focal_flag': bool(row.get('focal_flag', False)),
            })
        except (KeyError, TypeError, ValueError):
            continue

    if not parsed_rows:
        return []

    cooldown = timedelta(seconds=cooldown_seconds)
    clusters = []
    current = None
    last_alarm_time = None

    def flush_cluster():
        if current is not None:
            clusters.append(current.copy())

    for row in parsed_rows:
        if row['state'] < min_state:
            continue
        if (
            current is None
            or last_alarm_time is None
            or row['time'] - last_alarm_time > cooldown
        ):
            flush_cluster()
            current = {
                'start': row['time'],
                'end': row['time'],
                'row_indices': [],
                'windows': 0,
                'normal_windows': 0,
                'seizure_windows': 0,
                'max_supervised_score': 0.0,
                'max_legacy_state': 0,
                'max_ensemble_score': 0.0,
                'max_ml_score': 0.0,
                'max_smooth_score': 0.0,
                'max_burst_std_1s': 0.0,
                'max_burst_jerk_mean_1s': 0.0,
                'max_burst_jerk_max_1s': 0.0,
                'max_burst_std_surprise': 1.0,
                'max_burst_jerk_mean_surprise': 1.0,
                'max_burst_jerk_max_surprise': 1.0,
                'any_focal_flag': False,
            }
        current['end'] = row['time']
        current['row_indices'].append(int(row['row_idx']))
        current['windows'] += 1
        current['normal_windows'] += int(row['log_is_normal'])
        current['seizure_windows'] += int(row['log_is_seizure'])
        current['max_supervised_score'] = max(current['max_supervised_score'], row['supervised_score'])
        current['max_legacy_state'] = max(current['max_legacy_state'], row['legacy_state'])
        current['max_ensemble_score'] = max(current['max_ensemble_score'], row['ensemble_score'])
        current['max_ml_score'] = max(current['max_ml_score'], row['ml_score'])
        current['max_smooth_score'] = max(current['max_smooth_score'], row['smooth_score'])
        current['max_burst_std_1s'] = max(current['max_burst_std_1s'], row['burst_std_1s_max'])
        current['max_burst_jerk_mean_1s'] = max(current['max_burst_jerk_mean_1s'], row['burst_jerk_mean_1s_max'])
        current['max_burst_jerk_max_1s'] = max(current['max_burst_jerk_max_1s'], row['burst_jerk_max_1s_max'])
        current['max_burst_std_surprise'] = max(current['max_burst_std_surprise'], row['burst_std_surprise'])
        current['max_burst_jerk_mean_surprise'] = max(current['max_burst_jerk_mean_surprise'], row['burst_jerk_mean_surprise'])
        current['max_burst_jerk_max_surprise'] = max(current['max_burst_jerk_max_surprise'], row['burst_jerk_max_surprise'])
        current['any_focal_flag'] = current['any_focal_flag'] or row['focal_flag']
        last_alarm_time = row['time']

    flush_cluster()

    for cluster in clusters:
        cluster['duration_seconds'] = max(
            0.0,
            (cluster['end'] - cluster['start']).total_seconds(),
        )

    return clusters


def build_alarm_clusters(rows: list,
                         state_key: str = 'final_state',
                         states: np.ndarray = None,
                         supervised_scores: np.ndarray = None,
                         cooldown_seconds: int = CAREGIVER_ALARM_COOLDOWN_SECONDS) -> list:
    """Group nearby ALARM windows into caregiver-facing alert clusters."""
    return build_state_clusters(
        rows,
        min_state=2,
        state_key=state_key,
        states=states,
        supervised_scores=supervised_scores,
        cooldown_seconds=cooldown_seconds,
    )


def summarize_cluster_support(rows: list,
                              supervised_scores: np.ndarray,
                              row_indices: list,
                              alarm_threshold: float) -> dict:
    """Episode-level evidence summary used for promotion/suppression rules."""
    burst_windows = 0
    strong_burst_windows = 0
    focal_windows = 0
    legacy_warn_windows = 0
    legacy_alarm_windows = 0
    startle_windows = 0

    for idx in row_indices:
        row = rows[int(idx)]
        sup_score = float(supervised_scores[int(idx)])
        burst_std = float(row.get('burst_std_1s_max', row.get('accel_std', 0.0)))
        burst_jerk_mean = float(row.get('burst_jerk_mean_1s_max', row.get('jerk_mean', 0.0)))
        burst_jerk_max = float(row.get('burst_jerk_max_1s_max', row.get('jerk_max', 0.0)))
        burst_std_surprise = float(row.get('burst_std_surprise', 1.0))
        burst_jerk_mean_surprise = float(row.get('burst_jerk_mean_surprise', 1.0))
        burst_jerk_max_surprise = float(row.get('burst_jerk_max_surprise', 1.0))
        legacy_state = int(row.get('legacy_state', 0))
        ensemble_score = float(row.get('ensemble_score', 0.0))
        smooth_score = float(row.get('smooth_score', 0.0))
        ml_score = float(row.get('ml_score', 0.0))
        focal_flag = bool(row.get('focal_flag', False))
        burst_surprise = float(burst_jerk_mean_surprise)

        burst_like = (
            burst_std >= STAGE4_STARTLE_RESCUE_BURST_STD_THRESH
            or burst_jerk_mean >= STAGE4_STARTLE_RESCUE_BURST_JERK_MEAN_THRESH
            or burst_jerk_max >= STAGE4_STARTLE_RESCUE_BURST_JERK_MAX_THRESH
        )
        strong_burst = (
            burst_std >= CAREGIVER_SHORT_ALARM_BURST_STD_THRESH
            or burst_jerk_max >= CAREGIVER_SHORT_ALARM_BURST_JERK_MAX_THRESH
        )
        legacy_warn = legacy_state >= 1 and (
            ensemble_score >= STAGE4_GATE_ENSEMBLE_THRESH
            or smooth_score >= STAGE4_GATE_SMOOTH_THRESH
            or ml_score >= STAGE4_STARTLE_RESCUE_ML_THRESH
            or focal_flag
        )
        legacy_alarm = legacy_state >= 2 and (
            ensemble_score >= CAREGIVER_SHORT_ALARM_ENSEMBLE_THRESH
            or smooth_score >= CAREGIVER_SHORT_ALARM_ENSEMBLE_THRESH
            or ml_score >= CAREGIVER_SHORT_ALARM_ML_THRESH
            or focal_flag
        )

        burst_windows += int(burst_like)
        strong_burst_windows += int(strong_burst)
        focal_windows += int(focal_flag)
        legacy_warn_windows += int(legacy_warn)
        legacy_alarm_windows += int(legacy_alarm)
        startle_windows += int(
            burst_surprise >= STAGE4_STARTLE_SURPRISE_RATIO_THRESH
        )

    return {
        'burst_windows': int(burst_windows),
        'strong_burst_windows': int(strong_burst_windows),
        'focal_windows': int(focal_windows),
        'legacy_warn_windows': int(legacy_warn_windows),
        'legacy_alarm_windows': int(legacy_alarm_windows),
        'startle_windows': int(startle_windows),
        'promotion_sup_floor': max(CAREGIVER_WARN_PROMOTION_SUP, alarm_threshold - 0.08),
    }


def build_report_row_indices(rows: list) -> list:
    """Compact report mode: show the most relevant rows instead of every window."""
    if REPORT_VERBOSE_ROWS or not rows:
        return list(range(len(rows)))

    interesting = set()
    for idx, row in enumerate(rows):
        if (
            int(row.get('final_state', 0)) >= 2
            or bool(row.get('log_is_seizure'))
            or (
                int(row.get('final_state', 0)) >= 1
                and (
                    bool(row.get('focal_flag'))
                    or float(row.get('supervised_score', 0.0)) >= 0.35
                    or max(
                        float(row.get('burst_std_surprise', 1.0)),
                        float(row.get('burst_jerk_mean_surprise', 1.0)),
                        float(row.get('burst_jerk_max_surprise', 1.0)),
                    ) >= STAGE4_STARTLE_SURPRISE_RATIO_THRESH
                )
            )
        ):
            lo = max(0, idx - REPORT_CONTEXT_WINDOWS)
            hi = min(len(rows), idx + REPORT_CONTEXT_WINDOWS + 1)
            interesting.update(range(lo, hi))

    ordered = sorted(interesting)
    if len(ordered) <= REPORT_MAX_ROWS_PER_SESSION:
        return ordered

    seizure_rows = [
        idx for idx in ordered
        if bool(rows[idx].get('log_is_seizure'))
    ]
    alarm_rows = [
        idx for idx in ordered
        if int(rows[idx].get('final_state', 0)) >= 2 and idx not in seizure_rows
    ]
    ranked_alarm_rows = sorted(
        alarm_rows,
        key=lambda idx: (
            float(rows[idx].get('supervised_score', 0.0)),
            float(rows[idx].get('ml_score', 0.0)),
            float(rows[idx].get('ensemble_score', 0.0)),
        ),
        reverse=True,
    )

    keep = []
    seen = set()
    for idx in seizure_rows:
        if idx not in seen:
            keep.append(idx)
            seen.add(idx)

    remaining = max(REPORT_MAX_ROWS_PER_SESSION - len(keep), 0)
    for idx in ranked_alarm_rows[:remaining]:
        if idx not in seen:
            keep.append(idx)
            seen.add(idx)

    if len(keep) < REPORT_MAX_ROWS_PER_SESSION:
        for idx in ordered:
            if idx in seen:
                continue
            keep.append(idx)
            seen.add(idx)
            if len(keep) >= REPORT_MAX_ROWS_PER_SESSION:
                break

    return sorted(keep)


def select_warn_promotion_indices(rows: list,
                                  row_indices: list,
                                  supervised_scores: np.ndarray,
                                  alarm_threshold: float) -> list:
    """Choose one or two well-separated peak windows inside a strong WARN cluster."""
    candidates = []
    min_sup = CAREGIVER_WARN_PROMOTION_WINDOW_SUP
    min_gap = timedelta(seconds=CAREGIVER_WARN_PROMOTION_MIN_PEAK_SEP_SECONDS)

    for idx in row_indices:
        row = rows[int(idx)]
        sup_score = float(supervised_scores[int(idx)])
        burst_surprise = float(row.get('burst_jerk_mean_surprise', 1.0))
        if (
            sup_score < min_sup
            and burst_surprise < STAGE4_STARTLE_SURPRISE_RATIO_THRESH
        ):
            continue
        try:
            row_time = parse_timestamp(row['ts'])
        except (KeyError, ValueError):
            continue
        candidates.append({
            'idx': int(idx),
            'time': row_time,
            'score': (
                burst_surprise,
                sup_score,
                float(row.get('burst_jerk_max_1s_max', row.get('jerk_max', 0.0))),
                float(row.get('burst_std_1s_max', row.get('accel_std', 0.0))),
                float(row.get('ml_score', 0.0)),
                float(row.get('ensemble_score', 0.0)),
            ),
        })

    candidates.sort(key=lambda item: item['score'], reverse=True)

    chosen = []
    for candidate in candidates:
        if any(abs(candidate['time'] - picked['time']) < min_gap for picked in chosen):
            continue
        chosen.append(candidate)
        if len(chosen) >= CAREGIVER_WARN_PROMOTION_MAX_PER_CLUSTER:
            break

    return [item['idx'] for item in chosen]


def summarize_alarm_episodes(rows: list,
                             seizure_events: list,
                             state_key: str = 'final_state',
                             states: np.ndarray = None,
                             supervised_scores: np.ndarray = None,
                             match_margin_seconds: int = EXCLUDE_MARGIN) -> list:
    """
    Collapse ALARM windows into caregiver-facing alert episodes.

    Uses a cooldown so nearby ALARMs count as one caregiver alert episode.
    """
    clusters = build_alarm_clusters(
        rows,
        state_key=state_key,
        states=states,
        supervised_scores=supervised_scores,
        cooldown_seconds=CAREGIVER_ALARM_COOLDOWN_SECONDS,
    )
    if not clusters:
        return []

    margin = timedelta(seconds=match_margin_seconds)
    session_start = clusters[0]['start'] - margin
    session_end = clusters[-1]['end'] + margin
    relevant_events = [
        event for event in seizure_events
        if session_start <= event_time(event) <= session_end
    ]

    episodes = []
    for cluster in clusters:
        matched_events = []
        for event in relevant_events:
            event_ts = event_time(event)
            if cluster['start'] - margin <= event_ts <= cluster['end'] + margin:
                matched_events.append(event)

        nearest_event = None
        if matched_events:
            nearest_event = min(
                matched_events,
                key=lambda event: abs((event_time(event) - cluster['start']).total_seconds()),
            )

        episodes.append({
            'start': cluster['start'],
            'end': cluster['end'],
            'windows': cluster['windows'],
            'duration_seconds': cluster['duration_seconds'],
            'matched': bool(matched_events),
            'matched_count': len(matched_events),
            'event_type': event_type(nearest_event) if nearest_event else '',
            'note': event_note(nearest_event) if nearest_event else '',
            'normal_windows': cluster['normal_windows'],
            'seizure_windows': cluster['seizure_windows'],
        })

    return episodes


def contiguous_true_run_lengths(flags: np.ndarray) -> np.ndarray:
    """Return contiguous-True run lengths for each position in a boolean mask."""
    flags = np.asarray(flags, dtype=bool)
    run_lengths = np.zeros(flags.shape[0], dtype=np.int32)
    idx = 0
    n = flags.shape[0]
    while idx < n:
        if not flags[idx]:
            idx += 1
            continue
        end = idx
        while end < n and flags[end]:
            end += 1
        run_lengths[idx:end] = end - idx
        idx = end
    return run_lengths


def _rolling_median(rows: list, start: int, end: int, key: str, fallback: float) -> float:
    """Small helper for backward-looking baseline estimates."""
    values = [
        float(rows[idx].get(key, fallback))
        for idx in range(start, end)
        if np.isfinite(float(rows[idx].get(key, fallback)))
    ]
    return float(np.median(values)) if values else float(fallback)


def _surprise_ratio(value: float, baseline: float, floor: float) -> float:
    """Bounded ratio used for startle surprise against recent baseline."""
    denom = max(float(baseline), float(floor))
    return float(np.clip(float(value) / denom, 0.0, STARTLE_SURPRISE_RATIO_CAP))


def _ratio_to_unit(ratio: float, strong_ratio: float = 2.2) -> float:
    """Map a surprise ratio to [0, 1] for scoring."""
    return float(np.clip((float(ratio) - 1.0) / max(strong_ratio - 1.0, 1e-6), 0.0, 1.0))


def add_startle_context_features(rows: list) -> list:
    """
    Add baseline-relative startle features using only prior windows.

    The app can reproduce these features online because they depend only on the
    recent history within the current session, not any future data.
    """
    if not rows:
        return rows

    history = STARTLE_BASELINE_HISTORY_WINDOWS
    min_history = STARTLE_BASELINE_MIN_HISTORY

    for idx, row in enumerate(rows):
        start = max(0, idx - history)
        history_count = idx - start
        burst_std = float(row.get('burst_std_1s_max', row.get('accel_std', 0.0)))
        burst_jerk_mean = float(row.get('burst_jerk_mean_1s_max', row.get('jerk_mean', 0.0)))
        burst_jerk_max = float(row.get('burst_jerk_max_1s_max', row.get('jerk_max', 0.0)))
        ml_score = float(row.get('ml_score', 0.0))
        smooth_score = float(row.get('smooth_score', 0.0))

        if history_count >= min_history:
            base_burst_std = _rolling_median(rows, start, idx, 'burst_std_1s_max', burst_std)
            base_burst_jerk_mean = _rolling_median(rows, start, idx, 'burst_jerk_mean_1s_max', burst_jerk_mean)
            base_burst_jerk_max = _rolling_median(rows, start, idx, 'burst_jerk_max_1s_max', burst_jerk_max)
            base_ml = _rolling_median(rows, start, idx, 'ml_score', ml_score)
            base_smooth = _rolling_median(rows, start, idx, 'smooth_score', smooth_score)
        else:
            base_burst_std = burst_std
            base_burst_jerk_mean = burst_jerk_mean
            base_burst_jerk_max = burst_jerk_max
            base_ml = ml_score
            base_smooth = smooth_score

        burst_std_surprise = _surprise_ratio(burst_std, base_burst_std, 15.0)
        burst_jerk_mean_surprise = _surprise_ratio(burst_jerk_mean, base_burst_jerk_mean, 6.0)
        burst_jerk_max_surprise = _surprise_ratio(burst_jerk_max, base_burst_jerk_max, 60.0)
        ml_surprise = float(np.clip(ml_score - base_ml, 0.0, 1.0))
        smooth_surprise = float(np.clip(smooth_score - base_smooth, 0.0, 1.0))

        burst_surprise_component = max(
            _ratio_to_unit(burst_std_surprise),
            _ratio_to_unit(burst_jerk_mean_surprise),
            _ratio_to_unit(burst_jerk_max_surprise),
        )
        absolute_burst_component = max(
            np.clip(burst_std / max(STAGE4_STARTLE_RESCUE_BURST_STD_THRESH, 1.0), 0.0, 1.0),
            np.clip(burst_jerk_mean / max(STAGE4_STARTLE_RESCUE_BURST_JERK_MEAN_THRESH, 1.0), 0.0, 1.0),
            np.clip(burst_jerk_max / max(STAGE4_STARTLE_RESCUE_BURST_JERK_MAX_THRESH, 1.0), 0.0, 1.0),
        )
        support_component = max(
            float(bool(row.get('focal_flag', False))),
            0.70 if int(row.get('legacy_state', 0)) >= 2 else 0.0,
            0.45 if int(row.get('legacy_state', 0)) >= 1 else 0.0,
            float(np.clip((float(row.get('ensemble_score', 0.0)) - STAGE4_STARTLE_RESCUE_ENSEMBLE_THRESH) / 0.30, 0.0, 1.0)),
        )
        startle_score = float(np.clip(
            0.42 * burst_surprise_component
            + 0.22 * absolute_burst_component
            + 0.18 * ml_surprise
            + 0.08 * smooth_surprise
            + 0.10 * support_component,
            0.0,
            1.0,
        ))

        row['burst_std_surprise'] = burst_std_surprise
        row['burst_jerk_mean_surprise'] = burst_jerk_mean_surprise
        row['burst_jerk_max_surprise'] = burst_jerk_max_surprise
        row['ml_surprise'] = ml_surprise
        row['smooth_surprise'] = smooth_surprise
        row['startle_score'] = startle_score

    return rows


def apply_legacy_detector(rows: list, scaler, mlp, knn_scorer) -> list:
    """
    Run the existing Stage 2 + Stage 3 pipeline and attach its scores to rows.

    This keeps the original design intact so Stage 4 can learn from those
    outputs instead of replacing them wholesale.
    """
    rows = run_pipeline(rows)
    ml_scores = run_ml_inference(rows, mlp, scaler, knn_scorer)
    det_smooth = np.array([r['smooth_score'] for r in rows], dtype=float)
    legacy_ensemble = ensemble_score(det_smooth, ml_scores)
    legacy_states = run_state_machine(
        legacy_ensemble,
        warn_thresh=WARN_THRESH,
        alarm_thresh=ALARM_THRESH,
        min_alarm_wins=MIN_ALARM_WINDOWS,
        clear_thresh=CLEAR_THRESH,
        instant_scores=legacy_ensemble,
    )

    for i, row in enumerate(rows):
        row['ml_score'] = float(ml_scores[i])
        row['ensemble_score'] = float(legacy_ensemble[i])
        row['legacy_state'] = int(legacy_states[i])

    return rows


def derive_stage4_state_params(warn_threshold: float,
                               alarm_threshold: float,
                               min_alarm_wins: int = STAGE4_MIN_ALARM_WINDOWS,
                               instant_warn_offset: float = STAGE4_INSTANT_WARN_OFFSET,
                               instant_alarm_offset: float = STAGE4_INSTANT_ALARM_OFFSET) -> dict:
    """Stage-4-specific state-machine settings tuned to clear faster than Stage 2."""
    clear_thresh = min(
        max(CLEAR_THRESH, warn_threshold * STAGE4_CLEAR_RATIO),
        max(CLEAR_THRESH, warn_threshold - 0.05),
    )
    instant_warn_thresh = min(STAGE4_INSTANT_WARN_CAP, warn_threshold + instant_warn_offset)
    instant_alarm_thresh = min(
        STAGE4_INSTANT_ALARM_CAP,
        max(alarm_threshold + instant_alarm_offset, instant_warn_thresh + 0.08),
    )
    return {
        'min_alarm_wins': int(min_alarm_wins),
        'clear_thresh': float(clear_thresh),
        'instant_warn_thresh': float(instant_warn_thresh),
        'instant_alarm_thresh': float(instant_alarm_thresh),
        'instant_warn_offset': float(instant_warn_offset),
        'instant_alarm_offset': float(instant_alarm_offset),
    }


def build_stage4_state_param_candidates(warn_threshold: float,
                                        alarm_threshold: float) -> list:
    """Small search space of caregiver-alert policies for Stage 4."""
    candidates = []
    for min_alarm_wins in STAGE4_MIN_ALARM_WINDOWS_CANDIDATES:
        for instant_warn_offset in STAGE4_INSTANT_WARN_OFFSET_CANDIDATES:
            for instant_alarm_offset in STAGE4_INSTANT_ALARM_OFFSET_CANDIDATES:
                candidates.append(
                    derive_stage4_state_params(
                        warn_threshold,
                        alarm_threshold,
                        min_alarm_wins=min_alarm_wins,
                        instant_warn_offset=instant_warn_offset,
                        instant_alarm_offset=instant_alarm_offset,
                    )
                )
    return candidates


def stage4_strong_supervised_alarm_thresh(alarm_threshold: float,
                                          state_params: dict) -> float:
    """Higher supervised-score bar used to bypass legacy alarm corroboration."""
    return float(min(
        STAGE4_INSTANT_ALARM_CAP,
        max(
            alarm_threshold + STAGE4_GATE_STRONG_SUP_MARGIN,
            state_params['instant_alarm_thresh'] - 0.04,
        ),
    ))


def has_tonic_clonic_alarm_signature(row: dict) -> bool:
    """
    High-confidence tonic-clonic signature from the deterministic features.

    This is intentionally stricter than the broad seizure-type classifier
    because it is used to force caregiver-facing ALARM behaviour.
    """
    std = float(row.get('accel_std', 0.0))
    hr = float(row.get('hr', 0.0))
    smooth = float(row.get('smooth_score', 0.0))
    ensemble = float(row.get('ensemble_score', 0.0))
    accel_score = float(row.get('accel_score', 0.0))
    motor_peak_score = float(row.get('motor_peak_score', 0.0))
    spike_count = int(row.get('spike_count', 0))

    strong_motion_hr = (
        std >= STAGE4_TC_GATE_STD_THRESH
        and hr >= STAGE4_TC_GATE_HR_THRESH
        and max(accel_score, motor_peak_score) >= STAGE4_TC_GATE_ACCEL_THRESH
    )
    strong_motor_waveform = (
        std >= STD_ALARM_MG
        and smooth >= STAGE4_TC_GATE_SMOOTH_THRESH
        and ensemble >= STAGE4_TC_GATE_ENSEMBLE_THRESH
    )
    spike_heavy_pattern = (
        std >= STAGE4_TC_GATE_STD_THRESH
        and spike_count >= STAGE4_TC_GATE_SPIKE_COUNT
        and ensemble >= STAGE4_TC_GATE_ENSEMBLE_THRESH
    )

    return bool(strong_motion_hr or strong_motor_waveform or spike_heavy_pattern)


def tonic_clonic_override_support(row: dict,
                                  supervised_score: float,
                                  alarm_threshold: float,
                                  strong_sup_thresh: float) -> dict:
    """Corroboration summary used to keep the tonic-clonic override narrow."""
    legacy_state = int(row.get('legacy_state', 0))
    ensemble_score = float(row.get('ensemble_score', 0.0))
    smooth_score = float(row.get('smooth_score', 0.0))
    ml_score = float(row.get('ml_score', 0.0))
    accel_score = float(row.get('accel_score', 0.0))
    jerk_score = float(row.get('jerk_score', 0.0))
    motor_peak_score = float(row.get('motor_peak_score', 0.0))
    spike_count = int(row.get('spike_count', 0))

    support = 0
    has_warn_support = legacy_state >= 1 and (
        ensemble_score >= STAGE4_GATE_ENSEMBLE_THRESH
        or smooth_score >= STAGE4_GATE_SMOOTH_THRESH
    )
    if has_warn_support:
        support += 1
    if ensemble_score >= STAGE4_TC_OVERRIDE_ENSEMBLE_THRESH:
        support += 1
    if smooth_score >= STAGE4_TC_OVERRIDE_SMOOTH_THRESH:
        support += 1
    if ml_score >= STAGE4_TC_OVERRIDE_ML_THRESH:
        support += 1
    if max(accel_score, jerk_score, motor_peak_score) >= STAGE4_TC_OVERRIDE_MOTOR_THRESH:
        support += 1
    if spike_count >= STAGE4_TC_OVERRIDE_SPIKE_COUNT:
        support += 1
    if supervised_score >= max(strong_sup_thresh, alarm_threshold + STAGE4_TC_OVERRIDE_SUP_MARGIN):
        support += 1

    return {
        'legacy_state': legacy_state,
        'has_warn_support': has_warn_support,
        'strong_legacy_alarm': legacy_state >= 2,
        'support_count': support,
    }


def apply_caregiver_alarm_gate(rows: list,
                               supervised_scores: np.ndarray,
                               states: np.ndarray,
                               alarm_threshold: float,
                               state_params: dict) -> np.ndarray:
    """
    Demote unsupported ALARM windows to WARN.

    Caregiver-facing alerts should not come from Stage 4 alone unless the
    supervised score is clearly stronger than the base ALARM threshold. This
    keeps the startle-aware model while making ALARM require corroboration from
    the legacy detector or a very strong supervised spike.
    """
    gated_states = np.asarray(states, dtype=np.int32).copy()
    strong_sup_thresh = stage4_strong_supervised_alarm_thresh(alarm_threshold, state_params)
    tc_signature_flags = np.array(
        [has_tonic_clonic_alarm_signature(row) for row in rows],
        dtype=bool,
    )
    tc_run_lengths = contiguous_true_run_lengths(tc_signature_flags)

    for idx, row in enumerate(rows):
        sup_score = float(supervised_scores[idx])
        legacy_state = int(row.get('legacy_state', 0))
        hr = float(row.get('hr', 0.0))
        ensemble_score = float(row.get('ensemble_score', 0.0))
        smooth_score = float(row.get('smooth_score', 0.0))
        ml_score = float(row.get('ml_score', 0.0))
        accel_score = float(row.get('accel_score', 0.0))
        jerk_score = float(row.get('jerk_score', 0.0))
        motor_peak_score = float(row.get('motor_peak_score', 0.0))
        burst_std_1s = float(row.get('burst_std_1s_max', row.get('accel_std', 0.0)))
        burst_jerk_mean_1s = float(row.get('burst_jerk_mean_1s_max', row.get('jerk_mean', 0.0)))
        burst_jerk_max_1s = float(row.get('burst_jerk_max_1s_max', row.get('jerk_max', 0.0)))
        burst_std_surprise = float(row.get('burst_std_surprise', 1.0))
        burst_jerk_mean_surprise = float(row.get('burst_jerk_mean_surprise', 1.0))
        burst_jerk_max_surprise = float(row.get('burst_jerk_max_surprise', 1.0))
        tc_signature = bool(tc_signature_flags[idx])
        tc_run_length = int(tc_run_lengths[idx])
        low_sup_tc = sup_score < STAGE4_TC_LOW_SUP_FLOOR
        strong_motor_signal = max(accel_score, jerk_score, motor_peak_score) >= STAGE4_GATE_MOTOR_THRESH
        startle_surprise = float(burst_jerk_mean_surprise)
        jerk_startle_surprise = float(burst_jerk_mean_surprise)
        supported_legacy_warn = legacy_state >= 1 and (
            ensemble_score >= STAGE4_GATE_ENSEMBLE_THRESH
            or smooth_score >= STAGE4_GATE_SMOOTH_THRESH
        )
        low_hr_tc_penalty = (
            hr < STAGE4_TC_LOW_HR_SUPPRESS
            and not bool(row.get('focal_flag', False))
            and sup_score < STAGE4_TC_LOW_HR_SUP_MAX
        )
        hold_support_req = STAGE4_TC_HOLD_MIN_SUPPORT + int(low_hr_tc_penalty) * STAGE4_TC_LOW_HR_EXTRA_SUPPORT
        promotion_support_req = STAGE4_TC_PROMOTION_MIN_SUPPORT + int(low_hr_tc_penalty) * STAGE4_TC_LOW_HR_EXTRA_SUPPORT
        hold_run_req = (STAGE4_TC_MIN_RUN_HOLD if low_sup_tc else 1) + int(low_hr_tc_penalty) * STAGE4_TC_LOW_HR_EXTRA_RUN
        promotion_run_req = (STAGE4_TC_MIN_RUN_PROMOTION if low_sup_tc else 1) + int(low_hr_tc_penalty) * STAGE4_TC_LOW_HR_EXTRA_RUN
        corroborated_legacy_alarm = legacy_state >= 2 and (
            ensemble_score >= STAGE4_TC_OVERRIDE_ENSEMBLE_THRESH
            or smooth_score >= STAGE4_TC_OVERRIDE_SMOOTH_THRESH
            or (ml_score >= STAGE4_STARTLE_RESCUE_ML_THRESH and strong_motor_signal)
            or bool(row.get('focal_flag', False))
        )
        tc_support = tonic_clonic_override_support(
            row,
            sup_score,
            alarm_threshold,
            strong_sup_thresh,
        )
        tc_alarm_hold = tc_signature and (
            tc_run_length >= hold_run_req
            and (
                corroborated_legacy_alarm
                or (
                    supported_legacy_warn
                    and tc_support['support_count'] >= hold_support_req
                )
                or tc_support['support_count'] >= promotion_support_req
            )
        )
        tc_promotion = tc_signature and (
            tc_run_length >= promotion_run_req
            and (
                corroborated_legacy_alarm
                or (
                    supported_legacy_warn
                    and tc_support['support_count'] >= promotion_support_req
                )
                or tc_support['support_count'] > promotion_support_req
            )
        )
        startle_rescue_alarm = (
            not tc_signature
            and legacy_state >= 1
            and sup_score >= max(STAGE4_STARTLE_RESCUE_SUP_FLOOR, alarm_threshold - 0.40)
            and startle_surprise >= STAGE4_STARTLE_SURPRISE_RATIO_THRESH
            and (
                ml_score >= STAGE4_STARTLE_RESCUE_ML_THRESH
                or ensemble_score >= STAGE4_STARTLE_RESCUE_ENSEMBLE_THRESH
                or bool(row.get('focal_flag', False))
            )
            and (
                max(accel_score, jerk_score, motor_peak_score) >= STAGE4_STARTLE_RESCUE_MOTOR_THRESH
                or burst_std_1s >= STAGE4_STARTLE_RESCUE_BURST_STD_THRESH
                or burst_jerk_mean_1s >= STAGE4_STARTLE_RESCUE_BURST_JERK_MEAN_THRESH
                or burst_jerk_max_1s >= STAGE4_STARTLE_RESCUE_BURST_JERK_MAX_THRESH
                or startle_surprise >= STAGE4_STARTLE_SURPRISE_RATIO_THRESH
                or float(row.get('jerk_mean', 0.0)) >= 30.0
                or bool(row.get('focal_flag', False))
            )
        )
        startle_peak_alarm = (
            not tc_signature
            and jerk_startle_surprise >= STAGE4_STARTLE_PEAK_SURPRISE_THRESH
            and (
                legacy_state >= 1
                or bool(row.get('focal_flag', False))
                or sup_score >= CAREGIVER_SINGLE_WINDOW_STARTLE_SUP
                or ml_score >= CAREGIVER_SINGLE_WINDOW_STARTLE_ML
            )
            and (
                sup_score >= (
                    CAREGIVER_SINGLE_WINDOW_STARTLE_SUP
                    if legacy_state == 0 and not bool(row.get('focal_flag', False))
                    else STAGE4_STARTLE_PEAK_SUP_FLOOR
                )
                or ml_score >= (
                    CAREGIVER_SINGLE_WINDOW_STARTLE_ML
                    if legacy_state == 0 and not bool(row.get('focal_flag', False))
                    else max(STAGE4_STARTLE_PEAK_ML_THRESH, STAGE4_STARTLE_RESCUE_ML_THRESH - 0.10)
                )
            )
            and (
                ml_score >= (
                    CAREGIVER_SINGLE_WINDOW_STARTLE_ML
                    if legacy_state == 0 and not bool(row.get('focal_flag', False))
                    else STAGE4_STARTLE_PEAK_ML_THRESH
                )
                or (
                    legacy_state >= 1
                    and ensemble_score >= STAGE4_STARTLE_PEAK_ENSEMBLE_THRESH
                )
                or bool(row.get('focal_flag', False))
            )
            and (
                burst_jerk_mean_1s >= STAGE4_STARTLE_PEAK_BURST_JERK_MEAN_THRESH
                or burst_jerk_max_1s >= STAGE4_STARTLE_RESCUE_BURST_JERK_MAX_THRESH
                or bool(row.get('focal_flag', False))
            )
        )

        if gated_states[idx] < 2:
            if tc_promotion or startle_rescue_alarm or startle_peak_alarm:
                gated_states[idx] = 2
            continue

        strong_supervised_alarm = (
            sup_score >= strong_sup_thresh
            and (
                ml_score >= STAGE4_GATE_ML_THRESH
                or strong_motor_signal
                or bool(row.get('focal_flag', False))
            )
        )

        if not (
            corroborated_legacy_alarm
            or supported_legacy_warn
            or strong_supervised_alarm
            or tc_alarm_hold
            or startle_rescue_alarm
            or startle_peak_alarm
        ):
            gated_states[idx] = 1

    return gated_states


def apply_caregiver_episode_policy(rows: list,
                                   supervised_scores: np.ndarray,
                                   states: np.ndarray,
                                   alarm_threshold: float,
                                   state_params: dict) -> np.ndarray:
    """
    Apply caregiver-facing alert policy on top of window-level ALARM states.

    This merges nearby ALARM bursts into one caregiver alert episode and
    suppresses very short, weak episodes that are unlikely to be clinically
    meaningful alerts.
    """
    policy_states = np.asarray(states, dtype=np.int32).copy()
    strong_sup_thresh = stage4_strong_supervised_alarm_thresh(alarm_threshold, state_params)
    clusters = build_alarm_clusters(
        rows,
        states=policy_states,
        supervised_scores=supervised_scores,
        cooldown_seconds=CAREGIVER_ALARM_COOLDOWN_SECONDS,
    )

    for cluster in clusters:
        support = summarize_cluster_support(
            rows,
            supervised_scores,
            cluster['row_indices'],
            alarm_threshold,
        )
        strong_supervised_cluster = cluster['max_supervised_score'] >= max(
            CAREGIVER_SHORT_ALARM_STRONG_SUP,
            strong_sup_thresh,
        )
        strong_legacy_cluster = (
            cluster['max_legacy_state'] >= 2
            and (
                cluster['max_ensemble_score'] >= CAREGIVER_SHORT_ALARM_ENSEMBLE_THRESH
                or cluster['max_ml_score'] >= CAREGIVER_SHORT_ALARM_ML_THRESH
                or cluster['max_smooth_score'] >= CAREGIVER_SHORT_ALARM_ENSEMBLE_THRESH
                or cluster['any_focal_flag']
            )
        )
        strong_burst_cluster = (
            cluster['max_burst_std_1s'] >= CAREGIVER_SHORT_ALARM_BURST_STD_THRESH
            or cluster['max_burst_jerk_max_1s'] >= CAREGIVER_SHORT_ALARM_BURST_JERK_MAX_THRESH
        )
        startle_like_cluster = (
            cluster['max_supervised_score'] >= max(STAGE4_STARTLE_RESCUE_SUP_FLOOR, alarm_threshold - 0.18)
            and (
                cluster['max_burst_jerk_mean_surprise'] >= STAGE4_STARTLE_SURPRISE_RATIO_THRESH
                or cluster['max_burst_jerk_mean_1s'] >= STAGE4_STARTLE_RESCUE_BURST_JERK_MEAN_THRESH
                or cluster['max_burst_jerk_max_1s'] >= STAGE4_STARTLE_RESCUE_BURST_JERK_MAX_THRESH
            )
            and support['startle_windows'] >= CAREGIVER_STARTLE_CLUSTER_WINDOWS
            and (support['legacy_warn_windows'] >= 1 or cluster['any_focal_flag'])
        )
        short_startle_peak_cluster = (
            cluster['windows'] <= CAREGIVER_MIN_ALARM_EPISODE_WINDOWS
            and cluster['max_burst_jerk_mean_surprise'] >= STAGE4_STARTLE_PEAK_SURPRISE_THRESH
            and cluster['max_supervised_score'] >= (
                CAREGIVER_SINGLE_WINDOW_STARTLE_SUP
                if cluster['windows'] == 1
                else CAREGIVER_STARTLE_SHORT_CLUSTER_SUP
            )
            and (
                cluster['max_ml_score'] >= (
                    CAREGIVER_SINGLE_WINDOW_STARTLE_ML
                    if cluster['windows'] == 1
                    else CAREGIVER_STARTLE_SHORT_CLUSTER_ML
                )
                or (
                    cluster['windows'] > 1
                    and cluster['max_ensemble_score'] >= STAGE4_STARTLE_PEAK_ENSEMBLE_THRESH
                )
                or cluster['any_focal_flag']
            )
            and (
                cluster['max_burst_jerk_mean_1s'] >= STAGE4_STARTLE_PEAK_BURST_JERK_MEAN_THRESH
                or cluster['max_burst_jerk_max_1s'] >= STAGE4_STARTLE_RESCUE_BURST_JERK_MAX_THRESH
                or cluster['any_focal_flag']
            )
        )
        long_cluster = (
            cluster['windows'] >= CAREGIVER_LONG_CONFIRM_WINDOWS
            or cluster['duration_seconds'] >= CAREGIVER_LONG_CONFIRM_SECONDS
        )
        long_cluster_confirmed = (
            not long_cluster
            or (
                cluster['max_supervised_score'] >= CAREGIVER_LONG_CONFIRM_SUP
                or cluster['max_ml_score'] >= CAREGIVER_LONG_CONFIRM_ML
                or support['strong_burst_windows'] >= CAREGIVER_LONG_CONFIRM_BURST_WINDOWS
                or (
                    support['focal_windows'] >= CAREGIVER_LONG_CONFIRM_FOCAL_WINDOWS
                    and cluster['max_supervised_score'] >= max(alarm_threshold - 0.02, 0.0)
                )
            )
        )
        qualifies = (
            cluster['windows'] >= CAREGIVER_MIN_ALARM_EPISODE_WINDOWS
            or strong_supervised_cluster
            or strong_legacy_cluster
            or (strong_burst_cluster and cluster['max_supervised_score'] >= max(alarm_threshold - 0.10, 0.0))
            or startle_like_cluster
            or short_startle_peak_cluster
        )
        if qualifies and long_cluster_confirmed:
            continue
        policy_states[np.asarray(cluster['row_indices'], dtype=np.int32)] = 1

    warn_clusters = build_state_clusters(
        rows,
        min_state=1,
        states=policy_states,
        supervised_scores=supervised_scores,
        cooldown_seconds=CAREGIVER_ALARM_COOLDOWN_SECONDS,
    )
    for cluster in warn_clusters:
        row_indices = np.asarray(cluster['row_indices'], dtype=np.int32)
        if row_indices.size == 0 or np.any(policy_states[row_indices] >= 2):
            continue

        support = summarize_cluster_support(
            rows,
            supervised_scores,
            cluster['row_indices'],
            alarm_threshold,
        )
        if (
            cluster['windows'] < CAREGIVER_WARN_PROMOTION_MIN_WINDOWS
            or cluster['windows'] > CAREGIVER_WARN_PROMOTION_MAX_WINDOWS
            or cluster['duration_seconds'] > CAREGIVER_WARN_PROMOTION_MAX_SECONDS
        ):
            continue

        startle_like_warn_cluster = (
            support['legacy_warn_windows'] >= 2
            and (
                (
                    support['startle_windows'] >= CAREGIVER_STARTLE_CLUSTER_WINDOWS
                    and cluster['max_supervised_score'] >= CAREGIVER_WARN_PROMOTION_STRONG_SUP
                )
                or (
                    cluster['max_supervised_score'] >= max(
                        support['promotion_sup_floor'],
                        CAREGIVER_WARN_PROMOTION_STRONG_SUP,
                    )
                    and (
                        (
                            support['burst_windows'] >= CAREGIVER_WARN_PROMOTION_BURST_WINDOWS
                            and (
                                cluster['max_ml_score'] >= CAREGIVER_WARN_PROMOTION_ML
                                or cluster['max_ensemble_score'] >= CAREGIVER_WARN_PROMOTION_ENSEMBLE
                                or cluster['max_burst_jerk_mean_surprise'] >= STAGE4_STARTLE_SURPRISE_RATIO_THRESH
                            )
                        )
                        or (
                            support['focal_windows'] >= CAREGIVER_WARN_PROMOTION_FOCAL_WINDOWS
                            and cluster['max_burst_jerk_mean_surprise'] >= STAGE4_STARTLE_SURPRISE_RATIO_THRESH
                        )
                    )
                )
                or (
                    support['strong_burst_windows'] >= 1
                    and cluster['max_burst_jerk_mean_surprise'] >= STAGE4_STARTLE_SURPRISE_RATIO_THRESH
                    and cluster['max_supervised_score'] >= CAREGIVER_WARN_PROMOTION_STRONG_SUP
                )
            )
        )
        if not startle_like_warn_cluster:
            continue

        promotion_indices = select_warn_promotion_indices(
            rows,
            cluster['row_indices'],
            supervised_scores,
            alarm_threshold,
        )
        for idx in promotion_indices:
            policy_states[int(idx)] = 2

    return policy_states


def build_prepared_sessions(csv_files: list,
                            seizure_events: list,
                            scaler,
                            mlp,
                            knn_scorer) -> list:
    """Load, label, and score each session through the legacy pipeline once."""
    prepared = []
    for fp in csv_files:
        rows = load_csv(fp)
        rows = impute_hr(rows)
        session_seizure_windows, session_unparsed_count = label_rows_from_seizure_log(
            rows,
            seizure_events,
            margin_seconds=EXCLUDE_MARGIN,
        )
        rows = apply_legacy_detector(rows, scaler, mlp, knn_scorer)
        rows = add_startle_context_features(rows)
        parsed_indices = []
        parsed_times = []
        for idx, row in enumerate(rows):
            try:
                parsed_indices.append(idx)
                parsed_times.append(parse_timestamp(row['ts']))
            except (KeyError, ValueError):
                continue

        normal_mask = np.array([bool(row.get('log_is_normal')) for row in rows], dtype=bool)
        seizure_mask = np.array([bool(row.get('log_is_seizure')) for row in rows], dtype=bool)

        event_windows = []
        if parsed_times and seizure_events:
            margin = timedelta(seconds=EXCLUDE_MARGIN)
            session_start = parsed_times[0] - margin
            session_end = parsed_times[-1] + margin
            relevant_events = [
                event for event in seizure_events
                if session_start <= event_time(event) <= session_end
            ]
            for event in relevant_events:
                event_ts = event_time(event)
                start = bisect_left(parsed_times, event_ts - margin)
                end = bisect_right(parsed_times, event_ts + margin)
                event_windows.append({
                    'time': event_ts,
                    'note': event_note(event),
                    'event_type': event_type(event),
                    'is_startle': event_is_startle(event),
                    'start': int(start),
                    'end': int(end),
                })

        if len(parsed_times) >= 2:
            duration_hours = max(
                (parsed_times[-1] - parsed_times[0]).total_seconds() / 3600.0,
                len(parsed_times) * 5.0 / 3600.0,
            )
        else:
            duration_hours = max(len(rows) * 5.0 / 3600.0, 1.0 / 60.0)

        prepared.append({
            'path': fp,
            'name': os.path.basename(fp),
            'rows': rows,
            'seizure_windows': int(session_seizure_windows),
            'unparsed_count': int(session_unparsed_count),
            'parsed_indices': np.asarray(parsed_indices, dtype=np.int32),
            'normal_mask': normal_mask,
            'seizure_mask': seizure_mask,
            'event_windows': event_windows,
            'duration_hours': float(duration_hours),
        })
    return prepared


def build_holdout_stage4_scores(prepared_sessions: list) -> list:
    """
    Score each session with a calibrator trained on all of the other sessions.

    This gives us leave-one-session-out probabilities for threshold tuning.
    """
    scored_sessions = []
    for holdout_idx, holdout_session in enumerate(prepared_sessions):
        training_rows = []
        for train_idx, session in enumerate(prepared_sessions):
            if train_idx == holdout_idx:
                continue
            training_rows.extend(
                r for r in session['rows']
                if r.get('log_label') in {'normal', 'seizure'}
            )

        n_train_normal = sum(1 for row in training_rows if row.get('log_is_normal'))
        n_train_seizure = sum(1 for row in training_rows if row.get('log_is_seizure'))
        if not training_rows or not n_train_normal or not n_train_seizure:
            continue

        fold_calibrator, _ = fit_supervised_calibrator(training_rows)
        scored_sessions.append({
            **holdout_session,
            'scores': fold_calibrator.score_rows(holdout_session['rows']),
        })

    return scored_sessions


def evaluate_stage4_thresholds(scored_sessions: list,
                               seizure_times: list,
                               warn_threshold: float,
                               alarm_threshold: float,
                               state_params: dict = None) -> dict:
    """Evaluate Stage 4 thresholds on held-out session scores."""
    params = state_params or derive_stage4_state_params(warn_threshold, alarm_threshold)
    totals = {
        'warn_threshold': float(warn_threshold),
        'alarm_threshold': float(alarm_threshold),
        'state_params': params,
        'total_windows': 0,
        'flagged_windows': 0,
        'alarm_windows': 0,
        'normal_windows': 0,
        'normal_flagged': 0,
        'normal_alarms': 0,
        'seizure_windows': 0,
        'seizure_flagged': 0,
        'seizure_alarms': 0,
        'session_hours': 0.0,
        'alarm_episodes': 0,
        'true_alarm_episodes': 0,
        'false_alarm_episodes': 0,
        'logged_events': 0,
        'logged_tonic_clonic_events': 0,
        'matched_warn': 0,
        'matched_alarm': 0,
        'matched_alarm_tonic_clonic': 0,
    }

    for session in scored_sessions:
        scores = np.asarray(session['scores'], dtype=float)
        if scores.size == 0:
            continue

        states = run_state_machine(
            scores,
            warn_thresh=warn_threshold,
            alarm_thresh=alarm_threshold,
            min_alarm_wins=params['min_alarm_wins'],
            clear_thresh=params['clear_thresh'],
            instant_scores=scores,
            instant_warn_thresh=params['instant_warn_thresh'],
            instant_alarm_thresh=params['instant_alarm_thresh'],
        )
        states = apply_caregiver_alarm_gate(
            session['rows'],
            scores,
            states,
            alarm_threshold,
            params,
        )
        states = apply_caregiver_episode_policy(
            session['rows'],
            scores,
            states,
            alarm_threshold,
            params,
        )

        rows = session['rows']
        flagged_mask = states >= 1
        alarm_mask = states >= 2
        normal_mask = session['normal_mask']
        seizure_mask = session['seizure_mask']
        totals['total_windows'] += len(rows)
        totals['flagged_windows'] += int(np.sum(flagged_mask))
        totals['alarm_windows'] += int(np.sum(alarm_mask))

        session_normal_windows = int(np.sum(normal_mask))
        session_seizure_windows = int(np.sum(seizure_mask))
        session_normal_flagged = int(np.sum(flagged_mask & normal_mask))
        session_normal_alarms = int(np.sum(alarm_mask & normal_mask))
        session_seizure_flagged = int(np.sum(flagged_mask & seizure_mask))
        session_seizure_alarms = int(np.sum(alarm_mask & seizure_mask))

        totals['normal_windows'] += session_normal_windows
        totals['normal_flagged'] += session_normal_flagged
        totals['normal_alarms'] += session_normal_alarms
        totals['seizure_windows'] += session_seizure_windows
        totals['seizure_flagged'] += session_seizure_flagged
        totals['seizure_alarms'] += session_seizure_alarms
        totals['session_hours'] += float(session.get('duration_hours', 0.0))

        alarm_episodes = summarize_alarm_episodes(
            rows,
            seizure_times,
            states=states,
            match_margin_seconds=EXCLUDE_MARGIN,
        )
        totals['alarm_episodes'] += len(alarm_episodes)
        totals['true_alarm_episodes'] += sum(1 for episode in alarm_episodes if episode['matched'])
        totals['false_alarm_episodes'] += sum(1 for episode in alarm_episodes if not episode['matched'])

        parsed_indices = session['parsed_indices']
        for event_window in session['event_windows']:
            window_indices = parsed_indices[event_window['start']:event_window['end']]
            if window_indices.size == 0:
                totals['logged_events'] += 1
                totals['logged_tonic_clonic_events'] += int(
                    str(event_window.get('event_type', '')).strip().lower() == 'tonic-clonic'
                )
                continue

            totals['logged_events'] += 1
            is_tonic_clonic_event = str(event_window.get('event_type', '')).strip().lower() == 'tonic-clonic'
            totals['logged_tonic_clonic_events'] += int(is_tonic_clonic_event)
            if np.any(flagged_mask[window_indices]):
                totals['matched_warn'] += 1
            if np.any(alarm_mask[window_indices]):
                totals['matched_alarm'] += 1
                if is_tonic_clonic_event:
                    totals['matched_alarm_tonic_clonic'] += 1

    totals['normal_flag_rate'] = (
        totals['normal_flagged'] / totals['normal_windows']
        if totals['normal_windows'] else 0.0
    )
    totals['normal_alarm_rate'] = (
        totals['normal_alarms'] / totals['normal_windows']
        if totals['normal_windows'] else 0.0
    )
    totals['false_alarm_episode_rate'] = (
        totals['false_alarm_episodes'] / totals['session_hours']
        if totals['session_hours'] else 0.0
    )
    totals['event_warn_recall'] = (
        totals['matched_warn'] / totals['logged_events']
        if totals['logged_events'] else 0.0
    )
    totals['event_alarm_recall'] = (
        totals['matched_alarm'] / totals['logged_events']
        if totals['logged_events'] else 0.0
    )
    totals['event_alarm_recall_tonic_clonic'] = (
        totals['matched_alarm_tonic_clonic'] / totals['logged_tonic_clonic_events']
        if totals['logged_tonic_clonic_events'] else 0.0
    )
    totals['seizure_alarm_rate'] = (
        totals['seizure_alarms'] / totals['seizure_windows']
        if totals['seizure_windows'] else 0.0
    )
    return totals


def tune_stage4_thresholds(scored_sessions: list, seizure_times: list) -> dict:
    """
    Tune Stage 4 thresholds from hold-out scores.

    Preference order:
      1. preserve tonic-clonic caregiver ALARM recall
      2. meet caregiver-facing ALARM event recall target
      3. stay within a caregiver-usable false-episode budget
      4. keep WARN performance reasonable as a secondary goal
      5. otherwise fall back to the lowest-penalty tradeoff
    """
    score_arrays = [
        np.asarray(session['scores'], dtype=np.float32)
        for session in scored_sessions
        if len(session.get('scores', []))
    ]
    if not score_arrays:
        params = derive_stage4_state_params(WARN_THRESH, ALARM_THRESH)
        return {
            'mode': 'fallback-no-scores',
            'warn_threshold': float(WARN_THRESH),
            'alarm_threshold': float(ALARM_THRESH),
            'state_params': params,
            'normal_flag_rate': 0.0,
            'normal_alarm_rate': 0.0,
            'false_alarm_episode_rate': 0.0,
            'event_warn_recall': 0.0,
            'event_alarm_recall': 0.0,
            'event_alarm_recall_tonic_clonic': 0.0,
            'logged_events': 0,
            'logged_tonic_clonic_events': 0,
        }

    all_scores = np.concatenate(score_arrays)
    if all_scores.size == 0:
        params = derive_stage4_state_params(WARN_THRESH, ALARM_THRESH)
        return {
            'mode': 'fallback-no-scores',
            'warn_threshold': float(WARN_THRESH),
            'alarm_threshold': float(ALARM_THRESH),
            'state_params': params,
            'normal_flag_rate': 0.0,
            'normal_alarm_rate': 0.0,
            'false_alarm_episode_rate': 0.0,
            'event_warn_recall': 0.0,
            'event_alarm_recall': 0.0,
            'event_alarm_recall_tonic_clonic': 0.0,
            'logged_events': 0,
            'logged_tonic_clonic_events': 0,
        }

    warn_candidates = np.unique(np.clip(np.concatenate([
        np.linspace(STAGE4_WARN_CANDIDATE_MIN, 0.85, STAGE4_THRESHOLD_GRID_SIZE, dtype=np.float32),
        np.quantile(
            all_scores,
            np.linspace(STAGE4_WARN_QUANTILE_MIN, 0.97, STAGE4_THRESHOLD_GRID_SIZE),
        ),
    ]), 0.0, 1.0))
    alarm_candidates = np.unique(np.clip(np.concatenate([
        np.linspace(STAGE4_ALARM_CANDIDATE_MIN, 0.96, STAGE4_THRESHOLD_GRID_SIZE, dtype=np.float32),
        np.quantile(
            all_scores,
            np.linspace(STAGE4_ALARM_QUANTILE_MIN, 0.995, STAGE4_THRESHOLD_GRID_SIZE),
        ),
    ]), 0.0, 1.0))

    n_policy_candidates = (
        len(STAGE4_MIN_ALARM_WINDOWS_CANDIDATES)
        * len(STAGE4_INSTANT_WARN_OFFSET_CANDIDATES)
        * len(STAGE4_INSTANT_ALARM_OFFSET_CANDIDATES)
    )
    total_pairs = sum(
        1
        for warn_threshold in np.sort(warn_candidates)
        for alarm_threshold in np.sort(alarm_candidates)
        if alarm_threshold >= warn_threshold + STAGE4_MIN_WARN_ALARM_GAP
    ) * n_policy_candidates
    print(f"  Holdout threshold search: {len(warn_candidates)} WARN x "
          f"{len(alarm_candidates)} ALARM candidates x "
          f"{n_policy_candidates} "
          f"policies ({total_pairs} valid combinations)")

    feasible = []
    fallback = []
    evaluated_pairs = 0
    for warn_threshold in np.sort(warn_candidates):
        for alarm_threshold in np.sort(alarm_candidates):
            if alarm_threshold < warn_threshold + STAGE4_MIN_WARN_ALARM_GAP:
                continue

            for state_params in build_stage4_state_param_candidates(
                float(warn_threshold),
                float(alarm_threshold),
            ):
                evaluated_pairs += 1
                if evaluated_pairs == 1 or evaluated_pairs % STAGE4_PROGRESS_EVERY == 0:
                    print(f"    Tuning progress: {evaluated_pairs}/{total_pairs} pairs")

                metrics = evaluate_stage4_thresholds(
                    scored_sessions,
                    seizure_times,
                    float(warn_threshold),
                    float(alarm_threshold),
                    state_params=state_params,
                )
                tc_target_met = (
                    metrics['logged_tonic_clonic_events'] == 0
                    or metrics['event_alarm_recall_tonic_clonic'] >= STAGE4_TC_ALARM_EVENT_TARGET
                )
                meets_targets = (
                    tc_target_met
                    and metrics['event_alarm_recall'] >= STAGE4_ALARM_EVENT_TARGET
                    and metrics['false_alarm_episode_rate'] <= STAGE4_MAX_FALSE_ALARM_EPISODES_PER_HOUR
                    and metrics['normal_alarm_rate'] <= STAGE4_SOFT_NORMAL_ALARM_RATE
                )
                if meets_targets:
                    feasible.append((
                        -metrics['event_alarm_recall_tonic_clonic'],
                        -metrics['event_alarm_recall'],
                        metrics['false_alarm_episode_rate'],
                        metrics['normal_alarm_rate'],
                        metrics['normal_flag_rate'],
                        -metrics['event_warn_recall'],
                        metrics['state_params']['min_alarm_wins'],
                        metrics['state_params']['instant_alarm_thresh'],
                        -metrics['alarm_threshold'],
                        -metrics['warn_threshold'],
                        metrics,
                    ))
                    continue

                tc_alarm_deficit = max(
                    0.0,
                    STAGE4_TC_ALARM_EVENT_TARGET - metrics['event_alarm_recall_tonic_clonic'],
                ) if metrics['logged_tonic_clonic_events'] else 0.0
                alarm_deficit = max(0.0, STAGE4_ALARM_EVENT_TARGET - metrics['event_alarm_recall'])
                alarm_fp_excess = max(
                    0.0,
                    metrics['normal_alarm_rate'] - STAGE4_MAX_NORMAL_ALARM_RATE,
                )
                false_alarm_episode_excess = max(
                    0.0,
                    metrics['false_alarm_episode_rate'] - STAGE4_MAX_FALSE_ALARM_EPISODES_PER_HOUR,
                )
                warn_deficit = max(0.0, STAGE4_WARN_EVENT_TARGET - metrics['event_warn_recall'])
                flag_fp_excess = max(
                    0.0,
                    metrics['normal_flag_rate'] - STAGE4_MAX_NORMAL_FLAG_RATE,
                )
                penalty = (
                    12.0 * tc_alarm_deficit
                    + 9.0 * alarm_deficit
                    + 3.0 * alarm_fp_excess
                    + 4.0 * false_alarm_episode_excess
                    + 1.5 * warn_deficit
                    + 0.5 * flag_fp_excess
                )
                fallback.append((
                    penalty,
                    -metrics['event_alarm_recall_tonic_clonic'],
                    -metrics['event_alarm_recall'],
                    metrics['normal_alarm_rate'],
                    metrics['false_alarm_episode_rate'],
                    metrics['normal_flag_rate'],
                    -metrics['event_warn_recall'],
                    metrics['state_params']['min_alarm_wins'],
                    metrics['state_params']['instant_alarm_thresh'],
                    -metrics['alarm_threshold'],
                    -metrics['warn_threshold'],
                    metrics,
                ))

    if feasible:
        best = min(feasible, key=lambda item: item[:10])[10]
        best['mode'] = 'holdout-feasible'
        return best

    best = min(fallback, key=lambda item: item[:11])[11]
    best['mode'] = 'holdout-fallback'
    return best


def main():
    # ── 0. Find data ─────────────────────────────────────────────────────
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, '*-data.csv')))
    if not csv_files:
        print(f"\n[ERROR] No CSV files found in ./{DATA_DIR}/")
        print("        Place the session CSV files there and re-run.")
        return

    print(f"\n{'='*64}")
    print("  Seizure Detection Pipeline  —  Wearable HR + Accel")
    print(f"{'='*64}")
    print(f"  Using: accel (125 mg readings) + HR (BPM)")
    print(f"  Ignoring: alarmState, o2sat")
    print(f"  Caregiver alerts: ALARM only (WARN is internal early warning)")
    print(f"  Session files ({len(csv_files)}):")
    for f in csv_files:
        print(f"    • {os.path.basename(f)}")
    print()

    # ── 1. Load all data + impute HR ─────────────────────────────────────
    print("[Stage 1] Loading data + forward-filling HR gaps...")
    all_rows = load_all_csvs(csv_files)
    all_rows = impute_hr(all_rows)
    print(f"  Total windows : {len(all_rows):,}")

    seizure_times = []
    if os.path.exists(SEIZURE_LOG):
        print(f"\n[Stage 1] Loading seizure log from ./{SEIZURE_LOG}...")
        seizure_times = load_seizure_log(SEIZURE_LOG)
        n_startle_events = sum(1 for event in seizure_times if event_is_startle(event))
        print(f"  Logged seizures: {len(seizure_times):,}  "
              f"(startle={n_startle_events:,})")
    else:
        print(f"\n[Stage 1] No ./{SEIZURE_LOG} found, so training will use all windows.")

    seizure_window_count, unparsed_count = label_rows_from_seizure_log(
        all_rows,
        seizure_times,
        margin_seconds=EXCLUDE_MARGIN,
    )
    train_rows = normal_reference_rows(all_rows)
    print(f"  Windows treated as NORMAL for training : {len(train_rows):,}")
    print(f"  Windows highlighted by seizure log     : {seizure_window_count:,}")
    if unparsed_count:
        print(f"  [WARN] Could not label {unparsed_count:,} windows due to timestamp parse issues.")
    if not train_rows:
        print("  [ERROR] Seizure-log labelling left no normal windows for training.")
        return

    # ── 2. Extract features ──────────────────────────────────────────────
    print("\n[Stage 1] Extracting features (18 accel + 2 HR = 20 total)...")
    feats = compute_features(train_rows)
    norm_feats, scaler = normalize_features(feats)
    print(f"  Training feature matrix: {feats.shape}")

    # ── 3. Train autoencoder ─────────────────────────────────────────────
    print("\n[Stage 1] Training autoencoder (20→64→32→16→32→64→20)...")
    mlp = train_autoencoder(norm_feats, max_iter=AE_MAX_ITER)

    # ── 4. Build latent bank ─────────────────────────────────────────────
    print("\n[Stage 1] Building latent bank...")
    latent_bank = create_latent_bank(norm_feats, mlp)
    print(f"  Latent bank: {latent_bank.shape}")

    # ── 5. Fit kNN ───────────────────────────────────────────────────────
    print("\n[Stage 3] Fitting kNN scorer...")
    knn_scorer = LatentKNNScorer(k=KNN_K)
    knn_scorer.fit(latent_bank)
    knn_scorer.calibrate_threshold(latent_bank, percentile=KNN_PERCENTILE)

    print("\n[Stage 4] Preparing labelled sessions for supervised calibration...")
    prepared_sessions = build_prepared_sessions(
        csv_files,
        seizure_times,
        scaler,
        mlp,
        knn_scorer,
    )
    calibrator_rows = [
        row
        for session in prepared_sessions
        for row in session['rows']
        if row.get('log_label') in {'normal', 'seizure'}
    ]

    n_cal_seizure = sum(1 for r in calibrator_rows if r.get('log_is_seizure'))
    n_cal_normal = sum(1 for r in calibrator_rows if r.get('log_is_normal'))
    if not n_cal_seizure or not n_cal_normal:
        print("  [ERROR] Stage 4 needs both normal and seizure-labelled windows.")
        print(f"          Found normal={n_cal_normal:,}, seizure={n_cal_seizure:,}.")
        return

    stage4_cache_signature = build_stage4_tuning_signature(csv_files, SEIZURE_LOG)
    stage4_tuning = None
    if USE_STAGE4_TUNING_CACHE:
        stage4_tuning = load_stage4_tuning_cache(stage4_cache_signature)

    if stage4_tuning is not None:
        print("[Stage 4] Using cached holdout tuning...")
    else:
        print("[Stage 4] Tuning thresholds from leave-one-session-out scores...")
        holdout_scored_sessions = build_holdout_stage4_scores(prepared_sessions)
        stage4_tuning = tune_stage4_thresholds(holdout_scored_sessions, seizure_times)
        if USE_STAGE4_TUNING_CACHE:
            save_stage4_tuning_cache(stage4_cache_signature, stage4_tuning)

    print("[Stage 4] Fitting final supervised calibrator on all labelled windows...")
    calibrator, calibrator_summary = fit_supervised_calibrator(calibrator_rows)
    calibrator.warn_threshold = float(stage4_tuning['warn_threshold'])
    calibrator.alarm_threshold = float(stage4_tuning['alarm_threshold'])
    stage4_state_params = stage4_tuning['state_params']

    print(f"  Labelled windows: {calibrator_summary['n_rows']:,}  "
          f"(normal={calibrator_summary['n_normal']:,}, "
          f"seizure={calibrator_summary['n_seizure']:,})")
    print(f"  Startle-labelled windows: {calibrator_summary['n_startle']:,}")
    print(f"  Tonic-clonic-labelled windows: {calibrator_summary['n_tonic_clonic']:,}")
    print(f"  Class weight   : normal={calibrator_summary['class_weight'][0]:.2f}, "
          f"seizure={calibrator_summary['class_weight'][1]:.2f}")
    print(f"  Holdout targets: tonic-clonic ALARM >= {100*STAGE4_TC_ALARM_EVENT_TARGET:.0f}%  |  "
          f"caregiver ALARM recall >= {100*STAGE4_ALARM_EVENT_TARGET:.0f}%  |  "
          f"soft normal ALARM <= {100*STAGE4_SOFT_NORMAL_ALARM_RATE:.1f}%  |  "
          f"false ALARM episodes/hr <= {STAGE4_MAX_FALSE_ALARM_EPISODES_PER_HOUR:.1f}  |  "
          f"WARN recall guide >= {100*STAGE4_WARN_EVENT_TARGET:.0f}%")
    print(f"  Threshold mode : {stage4_tuning['mode']}")
    print(f"  WARN threshold : {calibrator.warn_threshold:.3f}  "
          f"(holdout event recall={100*stage4_tuning['event_warn_recall']:.1f}%)")
    print(f"  ALARM threshold: {calibrator.alarm_threshold:.3f}  "
          f"(holdout alarm recall={100*stage4_tuning['event_alarm_recall']:.1f}%)")
    if stage4_tuning.get('logged_tonic_clonic_events', 0):
        print(f"  Holdout tonic-clonic ALARM recall: "
              f"{100*stage4_tuning['event_alarm_recall_tonic_clonic']:.1f}%")
    print(f"  Holdout normal ALARM rate: {100*stage4_tuning['normal_alarm_rate']:.2f}%")
    print(f"  Holdout false ALARM episodes/hr: {stage4_tuning['false_alarm_episode_rate']:.2f}")
    print(f"  Holdout normal flag rate : {100*stage4_tuning['normal_flag_rate']:.1f}%")
    print(f"  Stage-4 state machine   : min_alarm_wins={stage4_state_params['min_alarm_wins']}, "
          f"clear={stage4_state_params['clear_thresh']:.3f}, "
          f"instant_warn={stage4_state_params['instant_warn_thresh']:.3f}, "
          f"instant_alarm={stage4_state_params['instant_alarm_thresh']:.3f}")
    print(f"  Caregiver ALARM gate   : corroborated legacy ALARM, or strong supervised >= "
          f"{stage4_strong_supervised_alarm_thresh(calibrator.alarm_threshold, stage4_state_params):.3f}, "
          f"or corroborated tonic-clonic override, or startle rescue")
    print(f"  Caregiver episode rule : cooldown={CAREGIVER_ALARM_COOLDOWN_SECONDS}s, "
          f"min_alarm_windows={CAREGIVER_MIN_ALARM_EPISODE_WINDOWS}, "
          f"long_confirm={CAREGIVER_LONG_CONFIRM_WINDOWS}+wins, "
          f"warn_promotion<={CAREGIVER_WARN_PROMOTION_MAX_SECONDS}s")
    report_mode = (
        f"verbose rows"
        if REPORT_VERBOSE_ROWS else
        f"compact rows (max {REPORT_MAX_ROWS_PER_SESSION}/session)"
    )
    print(f"  Report mode            : {report_mode}")
    if USE_STAGE4_TUNING_CACHE:
        print(f"  Stage-4 tuning cache   : {build_stage4_tuning_cache_path()}")
    top_features = ", ".join(
        f"{name}={weight:+.2f}" for name, weight in calibrator_summary['top_features']
    )
    print(f"  Top calibrator weights: {top_features}")

    # ── 6. Per-session detection ─────────────────────────────────────────
    print(f"\n{'='*64}")
    print("  Detection results per session")
    print(f"{'='*64}")

    total_windows  = 0
    total_flagged  = 0
    total_alarms   = 0
    total_normal_windows = 0
    total_normal_flagged = 0
    total_normal_alarms = 0
    total_seizure_windows = 0
    total_seizure_flagged = 0
    total_seizure_alarms = 0
    total_alarm_episodes = 0
    total_true_alarm_episodes = 0
    total_false_alarm_episodes = 0
    total_logged_events = 0
    total_matched_warn  = 0
    total_matched_alarm = 0

    for session_data in prepared_sessions:
        session = session_data['name']
        print(f"\n── {session} " + "─" * (56 - len(session)))

        rows = session_data['rows']
        session_seizure_windows = session_data['seizure_windows']
        session_unparsed_count = session_data['unparsed_count']

        # Stage 4: supervised final score on top of the legacy pipeline
        supervised_scores = calibrator.score_rows(rows)
        comb_states = run_state_machine(
            supervised_scores,
            warn_thresh=calibrator.warn_threshold,
            alarm_thresh=calibrator.alarm_threshold,
            min_alarm_wins=stage4_state_params['min_alarm_wins'],
            clear_thresh=stage4_state_params['clear_thresh'],
            instant_scores=supervised_scores,
            instant_warn_thresh=stage4_state_params['instant_warn_thresh'],
            instant_alarm_thresh=stage4_state_params['instant_alarm_thresh'],
        )
        comb_states = apply_caregiver_alarm_gate(
            rows,
            supervised_scores,
            comb_states,
            calibrator.alarm_threshold,
            stage4_state_params,
        )
        comb_states = apply_caregiver_episode_policy(
            rows,
            supervised_scores,
            comb_states,
            calibrator.alarm_threshold,
            stage4_state_params,
        )

        for i, r in enumerate(rows):
            r['supervised_score'] = float(supervised_scores[i])
            r['final_state'] = int(comb_states[i])

        # Print table
        report_indices = build_report_row_indices(rows)
        if report_indices:
            print(f"{'Timestamp':<22} {'HR':>5} {'Std':>7} {'JrkM':>5} "
                  f"{'Foc':>4} {'Det':>6} {'ML':>6} {'Ens':>6} {'Sup':>6} {'State':>6}  Type")
            print("─" * 82)
            if len(report_indices) < len(rows):
                print(f"  [compact] Showing {len(report_indices):,} of {len(rows):,} rows")

        for idx in report_indices:
            r = rows[idx]
            state = {0:'OK', 1:'WARN', 2:'ALARM'}.get(r['final_state'], '?')
            flag  = ' ◄' if r['final_state'] >= 1 else ''
            stype = classify_seizure_type({**r, 'pred_state': r['final_state']})
            foc   = 'Y' if r['focal_flag'] else '-'
            print(f"{r['ts']:<22} {r['hr']:>5.0f} {r['accel_std']:>7.1f} "
                  f"{r['jerk_mean']:>5.1f} {foc:>4} {r['smooth_score']:>6.3f} "
                  f"{r['ml_score']:>6.3f} {r['ensemble_score']:>6.3f} "
                  f"{r['supervised_score']:>6.3f} "
                  f"{state:>6}{flag}  {stype}")

        n_warn  = sum(1 for r in rows if r['final_state'] == 1)
        n_alarm = sum(1 for r in rows if r['final_state'] == 2)
        print(f"\n  Windows: {len(rows):,}  |  WARNING: {n_warn}  |  ALARM: {n_alarm}")

        session_normal_windows = sum(1 for r in rows if r.get('log_is_normal'))
        session_normal_flagged = sum(
            1 for r in rows if r.get('log_is_normal') and r['final_state'] >= 1
        )
        session_normal_alarms = sum(
            1 for r in rows if r.get('log_is_normal') and r['final_state'] == 2
        )
        session_seizure_flagged = sum(
            1 for r in rows if r.get('log_is_seizure') and r['final_state'] >= 1
        )
        session_seizure_alarms = sum(
            1 for r in rows if r.get('log_is_seizure') and r['final_state'] == 2
        )
        print(f"  Log labels: NORMAL {session_normal_windows:,}  |  "
              f"SEIZURE {session_seizure_windows:,}")
        if session_normal_windows:
            print(f"  Flagged NORMAL windows: {session_normal_flagged:,}  "
                  f"({100*session_normal_flagged/session_normal_windows:.1f}%)")
            print(f"  ALARM on NORMAL windows: {session_normal_alarms:,}  "
                  f"({100*session_normal_alarms/session_normal_windows:.1f}%)")
        if session_seizure_windows:
            print(f"  Flagged SEIZURE windows: {session_seizure_flagged:,}  "
                  f"({100*session_seizure_flagged/session_seizure_windows:.1f}%)")
            print(f"  ALARM on SEIZURE windows: {session_seizure_alarms:,}  "
                  f"({100*session_seizure_alarms/session_seizure_windows:.1f}%)")
        if session_unparsed_count:
            print(f"  [WARN] Unlabelled session windows: {session_unparsed_count:,}")

        alarm_episodes = summarize_alarm_episodes(rows, seizure_times, state_key='final_state')
        matched_alarm_episodes = sum(1 for episode in alarm_episodes if episode['matched'])
        false_alarm_episodes = len(alarm_episodes) - matched_alarm_episodes
        if alarm_episodes:
            print(f"  Caregiver ALARM episodes: {len(alarm_episodes)}  |  "
                  f"true-linked: {matched_alarm_episodes}  |  "
                  f"false-linked: {false_alarm_episodes}")
            false_episode_labels = []
            for episode in alarm_episodes:
                if episode['matched']:
                    continue
                false_episode_labels.append(
                    f"{episode['start'].strftime(TS_FORMAT)} "
                    f"({episode['windows']} windows, {episode['duration_seconds']:.0f}s)"
                )
            for label in false_episode_labels[:3]:
                print(f"    False ALARM episode: {label}")
            if len(false_episode_labels) > 3:
                print(f"    ... and {len(false_episode_labels) - 3} more false ALARM episodes")

        event_matches = summarize_logged_events(rows, seizure_times, state_key='final_state')
        if event_matches:
            matched_warn = sum(1 for m in event_matches if m['warn_match'])
            matched_alarm = sum(1 for m in event_matches if m['alarm_match'])
            missed_times = []
            for match in event_matches:
                if match['warn_match']:
                    continue
                label = match['time'].strftime(TS_FORMAT)
                if match.get('event_type') and match['event_type'] != 'seizure':
                    label += f" [{match['event_type']}]"
                elif match.get('note'):
                    label += f" [{match['note']}]"
                missed_times.append(label)
            print(f"  Logged seizures: {len(event_matches)}  |  "
                  f"Matched (>=WARN): {matched_warn}  |  "
                  f"Caregiver matched (ALARM): {matched_alarm}  |  "
                  f"Missed: {len(missed_times)}")
            for missed_ts in missed_times[:5]:
                print(f"    Missed log: {missed_ts}")
            if len(missed_times) > 5:
                print(f"    ... and {len(missed_times) - 5} more")

            total_logged_events += len(event_matches)
            total_matched_warn += matched_warn
            total_matched_alarm += matched_alarm

        total_windows += len(rows)
        total_flagged += n_warn + n_alarm
        total_alarms  += n_alarm
        total_normal_windows += session_normal_windows
        total_normal_flagged += session_normal_flagged
        total_normal_alarms += session_normal_alarms
        total_seizure_windows += session_seizure_windows
        total_seizure_flagged += session_seizure_flagged
        total_seizure_alarms += session_seizure_alarms
        total_alarm_episodes += len(alarm_episodes)
        total_true_alarm_episodes += matched_alarm_episodes
        total_false_alarm_episodes += false_alarm_episodes

    # ── 7. Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print("  OVERALL SUMMARY")
    print(f"{'='*64}")
    print(f"  Total windows processed : {total_windows:,}")
    print(f"  Windows flagged (≥WARN) : {total_flagged:,}  "
          f"({100*total_flagged/total_windows:.1f}%)")
    print(f"  Caregiver alerts (ALARM): {total_alarms:,}  "
          f"({100*total_alarms/total_windows:.1f}%)")
    print(f"  Caregiver ALARM episodes: {total_alarm_episodes:,}  "
          f"(true-linked={total_true_alarm_episodes:,}, false-linked={total_false_alarm_episodes:,})")
    if total_normal_windows:
        print(f"  Windows labelled NORMAL : {total_normal_windows:,}")
        print(f"  Flagged NORMAL windows  : {total_normal_flagged:,}  "
              f"({100*total_normal_flagged/total_normal_windows:.1f}%)")
        print(f"  ALARM on NORMAL windows : {total_normal_alarms:,}  "
              f"({100*total_normal_alarms/total_normal_windows:.1f}%)")
    if total_seizure_windows:
        print(f"  Windows labelled SEIZURE: {total_seizure_windows:,}")
        print(f"  Flagged SEIZURE windows : {total_seizure_flagged:,}  "
              f"({100*total_seizure_flagged/total_seizure_windows:.1f}%)")
        print(f"  ALARM on SEIZURE windows: {total_seizure_alarms:,}  "
              f"({100*total_seizure_alarms/total_seizure_windows:.1f}%)")
    if total_logged_events:
        print(f"  Logged seizures         : {total_logged_events:,}")
        print(f"  Matched at >=WARN       : {total_matched_warn:,}  "
              f"({100*total_matched_warn/total_logged_events:.1f}%)")
        print(f"  Caregiver matched (ALARM): {total_matched_alarm:,}  "
              f"({100*total_matched_alarm/total_logged_events:.1f}%)")
        print(f"  Missed logged seizures  : {total_logged_events - total_matched_warn:,}")
    print()


if __name__ == '__main__':
    report_path = build_report_path()
    with open(report_path, 'w', encoding='utf-8') as report_file:
        tee = Tee(sys.stdout, report_file)
        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
            print(f"[INFO] Saving report to ./{report_path}")
            main()
    print(f"[INFO] Report saved to ./{report_path}")
