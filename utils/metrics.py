"""Medical imaging and statistical validation metrics."""

import numpy as np


# ─── Image & Segmentation Metrics ─────────────────────────────────────

def dice_score(pred, target, threshold=0.5):
    """Dice similarity coefficient for binary segmentation.

    Args:
        pred: Predicted mask (probabilities or binary).
        target: Ground truth binary mask.
        threshold: Binarization threshold for pred.

    Returns:
        Dice coefficient in [0, 1].
    """
    pred_bin = (np.asarray(pred) > threshold).astype(bool)
    target_bin = np.asarray(target).astype(bool)
    intersection = np.sum(pred_bin & target_bin)
    return 2.0 * intersection / (np.sum(pred_bin) + np.sum(target_bin) + 1e-8)


def compute_suv(activity_bqml, weight_kg, injected_dose_bq, scan_time_s=0,
                injection_time_s=0, half_life_s=6586.2):
    """Compute Standardized Uptake Value with decay correction.

    Args:
        activity_bqml: Activity concentration in Bq/mL (from PET image).
        weight_kg: Patient weight in kg.
        injected_dose_bq: Injected dose in Bq.
        scan_time_s: Scan start time in seconds since midnight.
        injection_time_s: Injection time in seconds since midnight.
        half_life_s: Isotope half-life in seconds (default: F-18 = 6586.2s).

    Returns:
        SUV value(s) — same shape as activity_bqml.
    """
    elapsed = scan_time_s - injection_time_s
    decay_factor = np.exp(-np.log(2) * elapsed / half_life_s) if elapsed > 0 else 1.0
    corrected_dose = injected_dose_bq * decay_factor
    suv = activity_bqml * weight_kg * 1000 / corrected_dose  # *1000: kg→g for mL
    return suv


def total_perfusion_deficit(stress_map, rest_map=None, threshold=2.5):
    """Total Perfusion Deficit (TPD) for myocardial perfusion imaging.

    Fraction of myocardium with perfusion below threshold standard deviations
    from normal. Simplified version (clinical TPD uses polar map databases).

    Args:
        stress_map: Stress perfusion values (2D polar map or 3D volume).
        rest_map: Optional rest perfusion for ischemia calculation.
        threshold: SD threshold for abnormality.

    Returns:
        tpd: Fraction of abnormal segments [0, 1].
    """
    flat = np.asarray(stress_map).ravel()
    mean, std = flat.mean(), flat.std()
    abnormal = flat < (mean - threshold * std)
    return np.sum(abnormal) / len(flat)


# ─── Statistical Validation Metrics ───────────────────────────────────

def concordance_index(y_true, y_pred):
    """Harrell's C-statistic (concordance index).

    Probability that for a randomly chosen pair of subjects, the one with
    the higher predicted risk actually had the event.

    Args:
        y_true: Binary outcomes (0/1).
        y_pred: Predicted risk scores (higher = more risk).

    Returns:
        C-statistic in [0, 1]. 0.5 = random, 1.0 = perfect.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    concordant = 0
    discordant = 0

    for i in range(n):
        for j in range(i + 1, n):
            if y_true[i] != y_true[j]:
                if (y_true[i] > y_true[j] and y_pred[i] > y_pred[j]) or \
                   (y_true[j] > y_true[i] and y_pred[j] > y_pred[i]):
                    concordant += 1
                elif y_pred[i] != y_pred[j]:
                    discordant += 1

    total = concordant + discordant
    return concordant / total if total > 0 else 0.5


def net_reclassification_improvement(y_true, p_old, p_new, thresholds=None):
    """Net Reclassification Improvement (NRI).

    Measures how well a new model reclassifies subjects compared to an old model.

    Args:
        y_true: Binary outcomes.
        p_old: Predicted probabilities from old model.
        p_new: Predicted probabilities from new model.
        thresholds: Risk category boundaries (e.g., [0.06, 0.20]).
                    If None, computes continuous NRI.

    Returns:
        dict with 'nri', 'nri_events', 'nri_nonevents'.
    """
    y_true = np.asarray(y_true, dtype=bool)
    p_old = np.asarray(p_old)
    p_new = np.asarray(p_new)

    if thresholds is None:
        # Continuous NRI
        events = y_true
        nonevents = ~y_true

        up_events = np.sum(p_new[events] > p_old[events])
        down_events = np.sum(p_new[events] < p_old[events])
        nri_events = (up_events - down_events) / np.sum(events)

        up_nonevents = np.sum(p_new[nonevents] > p_old[nonevents])
        down_nonevents = np.sum(p_new[nonevents] < p_old[nonevents])
        nri_nonevents = (down_nonevents - up_nonevents) / np.sum(nonevents)
    else:
        # Categorical NRI
        bins = [-np.inf] + list(thresholds) + [np.inf]
        cat_old = np.digitize(p_old, bins) - 1
        cat_new = np.digitize(p_new, bins) - 1

        events = y_true
        nonevents = ~y_true

        up_events = np.sum(cat_new[events] > cat_old[events])
        down_events = np.sum(cat_new[events] < cat_old[events])
        nri_events = (up_events - down_events) / np.sum(events)

        up_nonevents = np.sum(cat_new[nonevents] > cat_old[nonevents])
        down_nonevents = np.sum(cat_new[nonevents] < cat_old[nonevents])
        nri_nonevents = (down_nonevents - up_nonevents) / np.sum(nonevents)

    return {
        'nri': nri_events + nri_nonevents,
        'nri_events': nri_events,
        'nri_nonevents': nri_nonevents,
    }


def integrated_discrimination_improvement(y_true, p_old, p_new):
    """Integrated Discrimination Improvement (IDI).

    Measures the improvement in discrimination slope between two models.

    Args:
        y_true: Binary outcomes.
        p_old: Predicted probabilities from old model.
        p_new: Predicted probabilities from new model.

    Returns:
        dict with 'idi', 'is_old' (discrimination slope old), 'is_new'.
    """
    y_true = np.asarray(y_true, dtype=bool)

    is_old = np.mean(p_old[y_true]) - np.mean(p_old[~y_true])
    is_new = np.mean(p_new[y_true]) - np.mean(p_new[~y_true])

    return {
        'idi': is_new - is_old,
        'is_old': is_old,
        'is_new': is_new,
    }


def brier_score(y_true, y_pred):
    """Brier score — mean squared error of probabilistic predictions.

    Args:
        y_true: Binary outcomes (0/1).
        y_pred: Predicted probabilities.

    Returns:
        Brier score (lower is better, 0 = perfect).
    """
    return np.mean((np.asarray(y_pred) - np.asarray(y_true)) ** 2)
