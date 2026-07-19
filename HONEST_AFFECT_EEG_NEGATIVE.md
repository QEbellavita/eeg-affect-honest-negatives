# Honest negative: subject-independent EEG affect (DEAP + DREAMER)

**Date:** 2026-06-14
**Trainers:** `train_deap_honest.py`, `train_dreamer_honest.py`
**Verdict:** NEGATIVE — hand-crafted EEG band-power + ECG/peripheral features + RandomForest do **not** beat the majority-class baseline when evaluated **subject-independently** (GroupKFold by subject). No usable model committed.

This is the honest counterpart to the leaky `ml-pipeline/amigos/train_dreamer.py`, whose good numbers come from a random `train_test_split` that puts the same subject's trials in both train and test (subject-identity leakage). Evaluated honestly, the signal isn't there with these features.

## Results (GroupKFold-5 by subject, `class_weight=balanced`)

| Dataset | Task | LOSO acc | Majority baseline | High-class F1 |
|---|---|---|---|---|
| DREAMER (23 subj, 14-ch EEG + 2-ch ECG, 1-5 SAM) | valence | 0.593 | 0.606 | 0.24 |
| DREAMER | arousal | 0.539 | 0.563 | 0.40 |
| DEAP (32 subj, 32-ch EEG + peripheral, mirror labels) | valence | 0.792 | 0.790 | 0.00 |
| DEAP | arousal | 0.768 | 0.767 | 0.00 |

- **DREAMER**: at/below baseline; it predicts the minority class but doesn't beat chance.
- **DEAP**: collapses entirely to majority class (high-class F1 = 0.00). Additionally, the **Kaggle mirror's labels are non-canonical** — valence mean 3.47 (min 0.00, max 9.00) vs real DEAP SAM valence ~5.4 on a 1-9 scale — so DEAP is doubly unreliable here. A canonical-source DEAP (official EULA download) could be retried; the archive is kept for that.

## Why this is the right (honest) outcome

Beating subject-independent baseline on EEG affect is research-grade work — differential entropy, functional connectivity, Riemannian / CSP features, domain adaptation, or deep models with subject-invariant training. Simple band-power + RF is the honest floor and it lands at chance. Reporting a high within-subject / random-split number instead would be exactly the leakage this program exists to eliminate.

## Reproducibility without raw data

Feature caches under `ml-pipeline/models/affect_eeg_negative/` (`deap_features_cache.npz`, `dreamer_features_cache.npz`) hold the extracted features + labels + subject groups, so the LOSO negative is reproducible without re-extracting from (or retaining) the multi-GB raw signals. The trainers auto-load the cache if present.
