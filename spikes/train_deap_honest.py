#!/usr/bin/env python3
"""Honest DEAP affect trainer — REAL, leakage-free valence/arousal from EEG +
peripheral physiology, evaluated SUBJECT-INDEPENDENTLY (GroupKFold by subject).

DEAP (Koelstra et al. 2012, IEEE TAC): 32 subjects, 40 music-video trials each
(1280 trials total). Preprocessed Python format: each s01.dat..s32.dat is a
latin1 pickle with 'data' (40 trials, 40 channels, 8064 samples @128Hz) and
'labels' (40 trials, [valence, arousal, dominance, liking] on 1-9). Channels
0-31 are 32-ch Biosemi EEG; 32-39 are peripheral (hEOG, vEOG, zEMG, tEMG, GSR,
Resp, BVP/Plethysmograph, Temp).

Honest eval = GroupKFold BY SUBJECT — no person in both train and test; the
StandardScaler is fit on the TRAIN fold only. One feature vector per trial.
DEAP is HARD subject-independently: literature reports ~55-65% binary
valence/arousal. A number in that range is HONEST; the ~80-95% often quoted
comes from within-subject / random-trial splits that leak subject identity.

Labels binarised at the 1-9 midpoint: rating > 5 == "high".

REAL data only — refuses to run without the dataset. No synthetic fallback.

Run: python spikes/train_deap_honest.py
"""
import os
import sys
import json
import pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DDIR = os.path.join(HERE, "..", "..", "datasets", "deap", "data_preprocessed_python")
OUT = os.path.join(HERE, "..", "..", "datasets", "deap")
CACHE = os.path.join(OUT, "deap_features_cache.npz")

FS = 128                       # preprocessed sampling rate
BASELINE_S = 3                 # first 3s is pre-stimulus baseline → dropped
N_EEG = 32                     # channels 0..31 are EEG
# peripheral channel indices within the 40-channel array
CH_ZEMG, CH_TEMG, CH_GSR, CH_RESP, CH_BVP, CH_TEMP = 34, 35, 36, 37, 38, 39
BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13),
         "beta": (13, 30), "gamma": (30, 45)}
THRESH = 5.0                   # 1-9 midpoint; rating > 5 == high

if not os.path.isdir(DDIR):
    sys.exit(f"REFUSING TO RUN: real DEAP not found at {DDIR}. No fabrication.")

from scipy.signal import welch, butter, filtfilt, find_peaks
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report


# ---------------------------------------------------------------- feature blocks
def _bandpower(sig, fs, lo, hi):
    f, ps = welch(sig, fs=fs, nperseg=min(256, len(sig)))
    idx = (f >= lo) & (f <= hi)
    return float(np.trapezoid(ps[idx], f[idx])) if np.any(idx) else 0.0


def eeg_features(eeg):
    """Log band-power per EEG channel (32 ch x 5 bands) + frontal alpha asymmetry."""
    feats = []
    alpha = {}
    for ch in range(N_EEG):
        sig = eeg[ch].astype(np.float64)
        for name, (lo, hi) in BANDS.items():
            bp = _bandpower(sig, FS, lo, hi)
            feats.append(float(np.log1p(bp)))
            if name == "alpha":
                alpha[ch] = bp
    # frontal alpha asymmetry (right-left): AF4-AF3, F4-F3, F8-F7 (Biosemi indices)
    asym = []
    for r, l in [(17, 1), (19, 2), (20, 4)]:   # (AF4,AF3) (F4,F3) (F8,F7)
        asym.append(float(np.log1p(alpha.get(r, 0.0)) - np.log1p(alpha.get(l, 0.0))))
    return feats + asym


def _bp_filt(sig, lo, hi, fs, order=3):
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, sig)


def bvp_features(bvp, fs=FS):
    """HR/HRV from BVP (plethysmograph) peak detection — ECG proxy."""
    x = bvp.astype(np.float64).ravel()
    try:
        f = _bp_filt(x, 0.5, 4.0, fs)
    except Exception:
        f = x - np.mean(x)
    pk, _ = find_peaks(f, distance=int(0.4 * fs))      # <=150 bpm
    if len(pk) < 4:
        return [0.0] * 5
    rr = np.diff(pk) / fs * 1000.0
    rr = rr[(rr > 333) & (rr < 1500)]
    if len(rr) < 4:
        return [0.0] * 5
    hr = 60000.0 / rr
    drr = np.diff(rr)
    return [float(np.mean(hr)), float(np.std(hr)), float(np.std(rr)),
            float(np.sqrt(np.mean(drr ** 2))), float(np.mean(np.abs(drr) > 50) * 100)]


def gsr_features(gsr, fs=FS):
    x = gsr.astype(np.float64).ravel()
    slope = float(np.polyfit(np.arange(len(x)), x, 1)[0]) if len(x) > 1 else 0.0
    sm = np.convolve(x, np.ones(int(0.5 * fs)) / int(0.5 * fs), mode="same")
    pk, _ = find_peaks(sm, prominence=max(1e-6, 0.05 * np.std(sm)), distance=int(fs))
    return [float(np.mean(x)), float(np.std(x)), float(np.max(x) - np.min(x)),
            slope, float(len(pk))]


def resp_features(resp, fs=FS):
    x = resp.astype(np.float64).ravel()
    try:
        f = _bp_filt(x, 0.1, 0.5, fs)
    except Exception:
        f = x - np.mean(x)
    pk, _ = find_peaks(f, distance=int(1.5 * fs))
    rate = len(pk) / (len(x) / fs) * 60.0
    return [float(rate), float(np.std(x))]


def temp_features(temp):
    x = temp.astype(np.float64).ravel()
    slope = float(np.polyfit(np.arange(len(x)), x, 1)[0]) if len(x) > 1 else 0.0
    return [float(np.mean(x)), slope]


def emg_features(emg):
    x = emg.astype(np.float64).ravel()
    return [float(np.sqrt(np.mean(x ** 2))), float(np.std(x))]


EEG_NAMES = [f"eeg{ch}_{b}" for ch in range(N_EEG) for b in BANDS]
FEATURE_NAMES = (
    EEG_NAMES
    + ["alpha_asym_AF", "alpha_asym_F", "alpha_asym_F8F7"]
    + ["bvp_hr_mean", "bvp_hr_std", "bvp_sdnn", "bvp_rmssd", "bvp_pnn50"]
    + ["gsr_mean", "gsr_std", "gsr_range", "gsr_slope", "gsr_scr_peaks"]
    + ["resp_rate", "resp_std"]
    + ["temp_mean", "temp_slope"]
    + ["zemg_rms", "zemg_std", "temg_rms", "temg_std"]
)


def trial_features(data, trial):
    start = BASELINE_S * FS
    eeg = data[trial, :N_EEG, start:]
    per = data[trial, :, start:]
    return np.array(
        eeg_features(eeg)
        + bvp_features(per[CH_BVP]) + gsr_features(per[CH_GSR])
        + resp_features(per[CH_RESP]) + temp_features(per[CH_TEMP])
        + emg_features(per[CH_ZEMG]) + emg_features(per[CH_TEMG]),
        dtype=np.float64,
    )


# ------------------------------------------------------------------------- build
def build():
    X, val, aro, g = [], [], [], []
    files = sorted(f for f in os.listdir(DDIR) if f.endswith(".dat"))
    for fn in files:
        subj = fn[:-4]
        # Trusted source: official DEAP preprocessed_python (the only format it
        # ships in; manually downloaded by the user). Not untrusted input.
        with open(os.path.join(DDIR, fn), "rb") as fh:
            d = pickle.load(fh, encoding="latin1")
        data, labels = d["data"], d["labels"]
        n0 = len(X)
        for t in range(data.shape[0]):
            X.append(trial_features(data, t))
            val.append(1 if labels[t, 0] > THRESH else 0)
            aro.append(1 if labels[t, 1] > THRESH else 0)
            g.append(subj)
        print(f"  {subj}: {len(X) - n0:3d} trials", flush=True)
    return (np.stack(X), np.asarray(val), np.asarray(aro), np.asarray(g))


def load_or_build():
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        print(f"(loaded cached features from {CACHE})")
        return z["X"], z["val"], z["aro"], z["g"]
    print("Extracting real DEAP features (32 subjects x 40 trials @128Hz)...")
    X, val, aro, g = build()
    np.savez_compressed(CACHE, X=X, val=val, aro=aro, g=g)
    return X, val, aro, g


# -------------------------------------------------------------------------- eval
def loso(X, y, groups, classes, label):
    clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    gkf = GroupKFold(n_splits=5)
    accs, f1s, yt, yp = [], [], [], []
    for k, (tr, te) in enumerate(gkf.split(clean, y, groups)):
        clf = make_pipeline(
            StandardScaler(),
            RandomForestClassifier(n_estimators=400, max_depth=12, n_jobs=-1,
                                   class_weight="balanced", random_state=42),
        )
        clf.fit(clean[tr], y[tr])
        p = clf.predict(clean[te])
        a, f = accuracy_score(y[te], p), f1_score(y[te], p, average="macro")
        accs.append(a); f1s.append(f)
        yt.extend(y[te]); yp.extend(p)
        print(f"  [{label}] fold {k}: held={sorted(set(groups[te]))}  acc={a:.3f}  macroF1={f:.3f}")
    print(f"\n  == {label}: HONEST LOSO (GroupKFold by subject) ==")
    print(f"     accuracy = {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"     macro-F1 = {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(classification_report(yt, yp, target_names=[classes[c] for c in sorted(set(y.tolist()))],
                                zero_division=0))
    # majority-class baseline for honest comparison
    maj = max(np.bincount(y)) / len(y)
    print(f"     (majority-class baseline = {maj:.4f})")
    return {"acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
            "macro_f1_mean": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s)),
            "majority_baseline": float(maj), "n_trials": int(len(y)),
            "eval": "GroupKFold(5) by subject"}


def _final(X, y):
    Xc = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return make_pipeline(
        StandardScaler(),
        RandomForestClassifier(n_estimators=400, max_depth=12, n_jobs=-1,
                               class_weight="balanced", random_state=42),
    ).fit(Xc, y)


def main():
    X, val, aro, groups = load_or_build()
    print(f"\ntrials={len(X)}  features={X.shape[1]}  subjects={len(set(groups))}")
    print(f"valence high={int(val.sum())}/{len(val)}  arousal high={int(aro.sum())}/{len(aro)}\n")

    print("=" * 64 + "\nTASK 1 — BINARY VALENCE (rating > 5 == high)\n" + "=" * 64)
    r_val = loso(X, val, groups, {0: "low_valence", 1: "high_valence"}, "valence")
    print("\n" + "=" * 64 + "\nTASK 2 — BINARY AROUSAL (rating > 5 == high)\n" + "=" * 64)
    r_aro = loso(X, aro, groups, {0: "low_arousal", 1: "high_arousal"}, "arousal")

    import joblib
    joblib.dump(
        {"valence_model": _final(X, val), "arousal_model": _final(X, aro),
         "feature_names": FEATURE_NAMES, "feature_count": len(FEATURE_NAMES),
         "task": "binary_valence_arousal", "classes": ["low", "high"], "fs": FS,
         "threshold": THRESH, "honest_cv_valence": r_val, "honest_cv_arousal": r_aro,
         "leakage_free": True},
        os.path.join(OUT, "deap_affect_honest.pkl"),
    )
    with open(os.path.join(OUT, "deap_affect_honest.json"), "w") as f:
        json.dump({"valence": r_val, "arousal": r_aro,
                   "eval": "GroupKFold(5) by subject (subject-independent / LOSO-style)",
                   "n_features": len(FEATURE_NAMES), "leakage_free": True,
                   "synthetic_fallback": False, "label_threshold": THRESH,
                   "lit_reference": "DEAP subject-independent binary valence/arousal ~55-65%",
                   "caveat": "subject-independent is HARD; within-subject/random-trial splits "
                             "leak subject identity and inflate to ~80-95%"},
                  f, indent=2)
    print(f"\nsaved honest model + metrics to {OUT}")
    print(f"  valence : acc={r_val['acc_mean']:.3f} (baseline {r_val['majority_baseline']:.3f})")
    print(f"  arousal : acc={r_aro['acc_mean']:.3f} (baseline {r_aro['majority_baseline']:.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
