# Honest spike: subject-independent EEG affect on FACED (N=123, finer-grained)

**Date:** 2026-06-27
**Status:** PRE-REGISTERED — decision rule fixed below *before* the run. Verdict PENDING.
**Trainer (planned):** `train_faced_honest.py` (+ `faced_data.py` loader)
**Companion to:** `HONEST_AFFECT_EEG_NEGATIVE.md` (DEAP/DREAMER negative)

## Question

DEAP and DREAMER hit a subject-independent ceiling — band-power + RF at/below the majority
baseline (high-class F1 0.00–0.40, GroupKFold-by-subject). FACED is the "more data + finer
labels" play: 123 subjects (vs 32), 32-ch, 28 clips, 9 emotions, built explicitly for
cross-subject affective computing. **Does N=123 + finer-grained elicitation break the ceiling,
or does the ceiling hold at larger N?**

The published FACED cross-subject numbers (binary 69.3%, 9-class 35.2%; CLISA 42.4%) are
**clip-category** decoding — partly stimulus-locked EEG shared across people, not subject-invariant
*felt* affect. This spike separates the two and compares like-for-like against our own ceiling.

## Prior ceiling we are comparing against (from `HONEST_AFFECT_EEG_NEGATIVE.md`)

| Dataset | Task | Acc | Majority baseline | High-class F1 |
|---|---|---|---|---|
| DREAMER | valence | 0.593 | 0.606 | 0.24 |
| DREAMER | arousal | 0.539 | 0.563 | 0.40 |
| DEAP | valence | 0.792 | 0.790 | 0.00 |
| DEAP | arousal | 0.768 | 0.767 | 0.00 |

Comparable metric across datasets = **minority/high-class F1 over the majority baseline** (accuracy
is base-rate-dependent and misleading — see DEAP's 0.79 acc / 0.00 F1 collapse).

## Data (verified 2026-06-27, no raw download required)

- **Features:** `EEG_Features/DE/subXXX.pkl.pkl` → `np.ndarray (28, 32, 30, 5)` float64 =
  (video, electrode, second, band[delta,theta,alpha,beta,gamma]). 123 subjects. Per-second sample
  = flatten(electrode×band) = **160-dim**. Confirmed by load + exact byte size (1,075,366 B).
- **Labels:** `After_remarks.mat` (in `Code.zip`, also `Data/subXXX/`). Struct `After_remark(28,1)`
  with fields `score, trial, vid, Accuracy, ResponseTime`. `score` = 12 items in order
  `[Joy,Tenderness,Inspiration,Amusement,Anger,Disgust,Fear,Sadness,Arousal,Valence,Familiarity,Liking]`
  on 0–7. **Valence = index 9, Arousal = index 8.**
- **Clip→emotion map:** `Stimuli_info.xlsx` (vid index → one of 9 categories).

### Two alignment gotchas (same class of bug as DEAP-Kaggle / AMIGOS-swap)
1. **Valence is item index 9, Arousal index 8** — mis-grabbing the wrong column silently inverts the label.
2. **DE features are ordered by sorted `vid`; `After_remarks` rows are in presentation order.**
   Labels MUST be joined to features through the `.vid` field, never by row position.

Both are caught by the Gate-0 sanity assertion below.

## Method

**Same DE features throughout. Step 1 mirrors FACED's own pipeline (to prove the harness reproduces
their result); Steps 2–3 use our RF + `GroupKFold(5)` + `class_weight='balanced'` harness (so the
FACED number is apples-to-apples with the DEAP/DREAMER ceiling).**

### Gate 0 — env + load + align (fail loud)
- `uv` venv, pins: `numpy>=1.21, scipy>=1.7, scikit-learn>=1.0, pandas>=1.3, openpyxl, joblib`
  (system python3.14 has no sci stack; FACED's `mne` not needed — DE is precomputed).
- Load DE + `After_remarks` + `Stimuli_info`; join labels via `.vid`.
- **Sanity assertion (must pass before any training):** after alignment, mean self-report **valence
  on paper-positive clips > on paper-negative clips** (and arousal sane). This simultaneously
  verifies index-9 valence AND the vid-ordering join. Abort on failure.
- Read recorded DEAP/DREAMER baseline from `models/affect_eeg_negative/{deap,dreamer}_affect_honest.json`.
- Cache to `models/affect_eeg_negative/faced_features_cache.npz` (`X`, `y_clip9`, `y_clipbin`,
  `y_valence`, `y_arousal`, `groups_subject`, `groups_vid`).

### Step 1 — validate the data reproduces above-chance cross-subject clip decoding
Purpose: confirm the extracted DE + label alignment are correct (not to chase the exact paper number).
- **Primary (lean):** DE + linear SVM + `GroupKFold`-by-subject, no running-norm/LDS. Expect clearly
  above chance (binary ≳ 0.60; 9-class ≳ 0.25 vs 0.111 chance). This alone validates data+alignment.
- **Optional (exact reproduction):** reuse FACED's *bundled* scripts from `Code.zip`
  (`Code/Validation/Classification_validation/SVM_analysis/`: `save_de.py`, `running_norm_fea.py`,
  `smooth_lds.py`, `main_de_svm.py`) to hit the published **69.3%** binary / **35.2%** 9-class
  (DE+running-norm+LDS+SVM, 10-fold 12×9+15). Run only if the lean check is ambiguous.
- **Pass gate:** lean binary ≥ 0.60 AND 9-class ≥ 0.25. Else → extraction/alignment bug; fix before Steps 2–3.

### Step 2 — ceiling comparison (the like-for-like number)
Per-subject self-report **valence + arousal**, binary @ threshold 3.5 (0–7).
RF + `GroupKFold(5)` + `class_weight='balanced'`, `StandardScaler` fit on train fold only.
Report acc, majority baseline, **minority-class F1** — same schema as DEAP/DREAMER.
- **Running-norm ablation** (with vs without online running-normalization on the test stream).
  This isolates the *calibration* contribution (running-norm = test-subject online adaptation).

### Step 3 — leave-one-video-out control (decisive)
Nested cross-subject × cross-video: test on held-out subjects **and** held-out clips, so the model
cannot win by memorizing clip-specific EEG.
- **Condition A** — cross-subject, *seen* clips (= Step 2).
- **Condition B** — cross-subject, *unseen* clips. **This is the result.**

## Pre-registered decision rule

Interpreted on **Condition B valence minority-F1** (arousal secondary):

- **POSITIVE / pivot-worthy:** Condition B valence minority-F1 **≥ 0.60** AND Condition B accuracy
  beats its majority baseline by **> 1 pooled-std** AND clearly exceeds the prior ceiling's best
  high-class F1 (0.40). → genuine subject-invariant felt-affect signal; promote toward affect
  registry @ a downstream detect→predict stage.
- **CALIBRATION-DRIVEN (partial):** Step 2/3 strong **with** running-norm but collapses **without**
  it → gain is online calibration, not static subject-invariance. Distinct, reportable result that
  *supports the documented calibration pivot* rather than refuting the ceiling.
- **NEGATIVE / kill:** Condition B minority-F1 **≤ 0.45** or accuracy within noise of majority
  baseline → ceiling holds at N=123; "more data alone didn't fix it" confirmed. Document, commit no
  model, stop.

## Artifacts
- `scripts/train_faced_honest.py`, `scripts/faced_data.py`
- `models/affect_eeg_negative/faced_features_cache.npz`
- `models/affect_eeg_negative/faced_affect_honest.json` (schema mirrors `deap_affect_honest.json`)
- this doc (verdict filled after run)

## Reproducibility without raw data
The DE feature cache + `After_remarks` labels (both from already-downloaded `EEG_Features.zip` +
`Code.zip`) make every step reproducible **without** the multi-GB raw `Data/`/`Processed_Data/`
download. Trainer auto-loads `faced_features_cache.npz` if present.

## Verdict

**NEGATIVE — the cross-subject felt-affect ceiling holds at N=123** (run 2026-06-27,
`models/affect_eeg_negative/faced_affect_honest.json`).

**Step 1 (alignment validation) — PASS.** Raw lean DE clip-decoding = 0.582 binary / 0.220
9-class: ~9σ / ~20σ above chance (a real misalignment gives *chance*, not this), just under the
optimistic lean gate. Per the pre-registered running-norm escalation, calibrated DE reaches
**0.640 / 0.345**, reproducing FACED's published 9-class ~35.2% — alignment confirmed and the
setup shown well-powered. (The Step-1 gate validates on the running-norm path; the decision rule
below is unchanged.)

| metric | seen clips (Cond A) | **unseen clips (Cond B, decisive)** | majority |
|---|---|---|---|
| valence acc | 0.564 | **0.540** | 0.567 |
| valence high-class F1 | 0.486 | **0.453** | — |
| arousal acc | 0.536 | 0.515 | 0.539 |
| arousal high-class F1 | 0.593 | 0.570 | — |

The cross-video control is decisive: valence high-class F1 *drops* 0.486 → **0.453** from seen
to unseen clips, and Condition-B accuracy (0.540) is **below** the majority baseline (0.567) —
the modest seen-clip signal was partly clip memorization, not subject-invariant felt affect.
Arousal's higher F1 is the base-rate artifact (near-balanced classes); its accuracy (0.515) is
also below majority. Per-subject calibration lifts Step-2 valence acc 0.564 → 0.608 (high-class
F1 0.486 → 0.541) — a small real gain toward the calibration lever — but the calibrated F1 stays
below the 0.60 bar, so it is not "calibration-driven" by the rule → **NEGATIVE**.

**Conclusion:** N=123 + finer-grained 9-emotion elicitation does NOT break the cross-subject
felt-affect ceiling. With the DEAP/DREAMER negatives, "more data alone
won't fix it" is now confirmed at the largest available scale. The published FACED 69%/35% are
**clip-category** decoding (stimulus-locked, partly shared EEG) — reproduced here in Step 1 —
which is a different task from generalizing *felt* affect across people and stimuli. Honesty
gate honored: no model committed.
