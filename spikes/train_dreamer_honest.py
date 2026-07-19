#!/usr/bin/env python3
"""Honest DREAMER affect trainer — REAL, leakage-free valence/arousal from EEG +
ECG, evaluated SUBJECT-INDEPENDENTLY (GroupKFold by subject).

REPLACES the leaky ml-pipeline/amigos/train_dreamer.py, which uses a random
train_test_split that puts trials from ONE subject in both train and test
(subject-identity leakage → inflated accuracy).

DREAMER (Katsigiannis & Ramzan 2018): 23 subjects, 18 film-clip trials each
(414 total). 14-ch Emotiv EEG @128Hz + 2-ch ECG @256Hz. Self-report valence /
arousal / dominance on 1-5 SAM. Stored as a single DREAMER.mat struct.

Honest eval = GroupKFold BY SUBJECT — no person in both train and test; scaler
fit on TRAIN fold only. One feature vector per trial (last 60s of stimulus).
Labels binarised at the 1-5 midpoint: rating > 3 == "high". Literature
subject-independent binary valence/arousal ~60-70%.

REAL data only — refuses to run without the dataset. No synthetic fallback.

Run: python spikes/train_dreamer_honest.py
"""
import os
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "datasets", "dreamer")
MAT = os.path.join(OUT, "DREAMER.mat")
CACHE = os.path.join(OUT, "dreamer_features_cache.npz")

EEG_FS = 128
ECG_FS = 256
USE_S = 60                      # last 60s of each stimulus
BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13),
         "beta": (13, 30), "gamma": (30, 45)}
THRESH = 3.0                   # 1-5 midpoint; rating > 3 == high

if not os.path.exists(MAT):
    sys.exit(f"REFUSING TO RUN: real DREAMER.mat not found at {MAT}. No fabrication.")

from scipy import io as sio
from scipy.signal import welch, butter, filtfilt, find_peaks
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report


def _bandpower(sig, fs, lo, hi):
    f, ps = welch(sig, fs=fs, nperseg=min(256, len(sig)))
    idx = (f >= lo) & (f <= hi)
    return float(np.trapezoid(ps[idx], f[idx])) if np.any(idx) else 0.0


def eeg_features(eeg, fs=EEG_FS):
    """eeg shape (channels, samples). Log band-power per channel + frontal asym."""
    feats = []
    alpha = []
    for ch in range(eeg.shape[0]):
        sig = eeg[ch].astype(np.float64)
        a = 0.0
        for name, (lo, hi) in BANDS.items():
            bp = _bandpower(sig, fs, lo, hi)
            feats.append(float(np.log1p(bp)))
            if name == "alpha":
                a = bp
        alpha.append(a)
    # Emotiv 14: AF3,F7,F3,FC5,T7,P7,O1,O2,P8,T8,FC6,F4,F8,AF4 (idx 0..13)
    # frontal alpha asymmetry right-left: AF4-AF3 (13,0), F4-F3 (11,2), F8-F7 (12,1)
    asym = [float(np.log1p(alpha[r]) - np.log1p(alpha[l]))
            for r, l in [(13, 0), (11, 2), (12, 1)] if r < len(alpha) and l < len(alpha)]
    return feats, asym


def _bp_filt(sig, lo, hi, fs, order=3):
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, sig)


def ecg_features(ecg, fs=ECG_FS):
    """HRV from ECG R-peaks. ecg shape (channels, samples); use channel 0."""
    x = np.asarray(ecg)[0].astype(np.float64).ravel() if ecg.ndim > 1 else ecg.astype(np.float64).ravel()
    try:
        f = _bp_filt(x, 5, 15, fs)
    except Exception:
        f = x - np.mean(x)
    f = np.abs(np.gradient(f))
    thr = np.mean(f) + 0.5 * np.std(f)
    pk, _ = find_peaks(f, height=thr, distance=int(0.33 * fs))
    if len(pk) < 4:
        return [0.0] * 6
    rr = np.diff(pk) / fs * 1000.0
    rr = rr[(rr > 300) & (rr < 2000)]
    if len(rr) < 4:
        return [0.0] * 6
    hr = 60000.0 / rr
    drr = np.diff(rr)
    lfhf = 0.0
    try:
        t = np.cumsum(rr) / 1000.0
        tt = np.arange(t[0], t[-1], 0.25)
        if len(tt) >= 16:
            ri = np.interp(tt, t, rr)
            fr, ps = welch(ri - ri.mean(), fs=4.0, nperseg=min(256, len(ri)))
            lf = np.trapezoid(ps[(fr >= 0.04) & (fr < 0.15)], fr[(fr >= 0.04) & (fr < 0.15)])
            hf = np.trapezoid(ps[(fr >= 0.15) & (fr < 0.40)], fr[(fr >= 0.15) & (fr < 0.40)])
            lfhf = float(lf / (hf + 1e-9))
    except Exception:
        lfhf = 0.0
    return [float(np.mean(hr)), float(np.std(hr)), float(np.std(rr)),
            float(np.sqrt(np.mean(drr ** 2))), float(np.mean(np.abs(drr) > 50) * 100),
            float(np.log1p(lfhf))]


def _tail(arr, fs, secs):
    """arr (samples, channels) or (channels, samples) → (channels, last secs)."""
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 1:
        a = a[None, :]
    if a.shape[0] > a.shape[1]:      # (samples, channels) → transpose
        a = a.T
    n = int(secs * fs)
    return a[:, -n:] if a.shape[1] > n else a


FEATURE_NAMES = None   # set after first trial (depends on EEG channel count)


def build():
    # Trusted source: official DREAMER.mat (the only format it ships in;
    # manually downloaded by the user). Not untrusted input.
    mat = sio.loadmat(MAT, struct_as_record=False, squeeze_me=True)
    dr = mat["DREAMER"]
    n_subj, n_vid = int(dr.noOfSubjects), int(dr.noOfVideoSequences)
    eeg_fs = int(dr.EEG_SamplingRate); ecg_fs = int(dr.ECG_SamplingRate)
    print(f"  subjects={n_subj} videos={n_vid} eeg_fs={eeg_fs} ecg_fs={ecg_fs}")
    X, val, aro, g = [], [], [], []
    global FEATURE_NAMES
    for si in range(n_subj):
        sd = dr.Data[si]
        for vi in range(n_vid):
            eeg = _tail(sd.EEG.stimuli[vi], eeg_fs, USE_S)
            ecg = _tail(sd.ECG.stimuli[vi], ecg_fs, USE_S)
            ef, asym = eeg_features(eeg, eeg_fs)
            feats = ef + asym + ecg_features(ecg, ecg_fs)
            if FEATURE_NAMES is None:
                nch = eeg.shape[0]
                FEATURE_NAMES = (
                    [f"eeg{c}_{b}" for c in range(nch) for b in BANDS]
                    + [f"alpha_asym_{i}" for i in range(len(asym))]
                    + ["ecg_hr_mean", "ecg_hr_std", "ecg_sdnn", "ecg_rmssd",
                       "ecg_pnn50", "ecg_lfhf_log"]
                )
            X.append(np.array(feats, dtype=np.float64))
            val.append(1 if float(sd.ScoreValence[vi]) > THRESH else 0)
            aro.append(1 if float(sd.ScoreArousal[vi]) > THRESH else 0)
            g.append(f"s{si + 1:02d}")
        print(f"  s{si + 1:02d}: {n_vid} trials", flush=True)
    return np.stack(X), np.asarray(val), np.asarray(aro), np.asarray(g)


def load_or_build():
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        global FEATURE_NAMES
        FEATURE_NAMES = list(z["names"])
        print(f"(loaded cached features from {CACHE})")
        return z["X"], z["val"], z["aro"], z["g"]
    print("Extracting real DREAMER features (23 subjects x 18 trials)...")
    X, val, aro, g = build()
    np.savez_compressed(CACHE, X=X, val=val, aro=aro, g=g, names=np.array(FEATURE_NAMES))
    return X, val, aro, g


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

    print("=" * 64 + "\nTASK 1 — BINARY VALENCE (rating > 3 == high)\n" + "=" * 64)
    r_val = loso(X, val, groups, {0: "low_valence", 1: "high_valence"}, "valence")
    print("\n" + "=" * 64 + "\nTASK 2 — BINARY AROUSAL (rating > 3 == high)\n" + "=" * 64)
    r_aro = loso(X, aro, groups, {0: "low_arousal", 1: "high_arousal"}, "arousal")

    import joblib
    joblib.dump(
        {"valence_model": _final(X, val), "arousal_model": _final(X, aro),
         "feature_names": list(FEATURE_NAMES), "feature_count": len(FEATURE_NAMES),
         "task": "binary_valence_arousal", "classes": ["low", "high"],
         "eeg_fs": EEG_FS, "ecg_fs": ECG_FS, "threshold": THRESH,
         "honest_cv_valence": r_val, "honest_cv_arousal": r_aro,
         "replaces": "amigos/train_dreamer.py (random split = subject-identity leakage)",
         "leakage_free": True},
        os.path.join(OUT, "dreamer_affect_honest.pkl"),
    )
    with open(os.path.join(OUT, "dreamer_affect_honest.json"), "w") as f:
        json.dump({"valence": r_val, "arousal": r_aro,
                   "eval": "GroupKFold(5) by subject (subject-independent / LOSO-style)",
                   "n_features": len(FEATURE_NAMES), "leakage_free": True,
                   "synthetic_fallback": False, "label_threshold": THRESH,
                   "replaces": "amigos/train_dreamer.py (random train_test_split leaks subject identity)",
                   "lit_reference": "DREAMER subject-independent binary valence/arousal ~60-70%"},
                  f, indent=2)
    print(f"\nsaved honest model + metrics to {OUT}")
    print(f"  valence : acc={r_val['acc_mean']:.3f} (baseline {r_val['majority_baseline']:.3f})")
    print(f"  arousal : acc={r_aro['acc_mean']:.3f} (baseline {r_aro['majority_baseline']:.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
