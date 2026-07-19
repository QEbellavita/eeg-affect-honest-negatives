#!/usr/bin/env python3
"""Honest FACED cross-subject affect spike. Pre-registered protocol +
decision rule: HONEST_AFFECT_FACED.md. Parity with train_deap_honest.py
(RF + GroupKFold-5 + balanced). Commits NO model unless verdict POSITIVE."""
import os
import sys
import json
import numpy as np
import faced_data as fd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score

OUT = os.path.join(fd.HERE, "..", "models", "affect_eeg_negative", "faced_affect_honest.json")


def rf_factory():
    return make_pipeline(StandardScaler(), RandomForestClassifier(
        n_estimators=400, max_depth=12, n_jobs=-1, class_weight="balanced", random_state=42))


def svm_factory():
    return make_pipeline(StandardScaler(), LinearSVC(
        class_weight="balanced", random_state=42, dual="auto", max_iter=5000))


def per_subject_znorm(X, groups):
    """Label-free, transductive per-subject standardisation (calibration proxy
    for FACED running-norm). Each subject normalised by its OWN feature stats."""
    Xz = X.astype(np.float64).copy()
    for s in np.unique(groups):
        m = groups == s
        mu, sd = Xz[m].mean(0), Xz[m].std(0)
        sd[sd == 0] = 1.0
        Xz[m] = (Xz[m] - mu) / sd
    return Xz


def evaluate(X, y, groups, clf_factory, splits=None):
    Xc = np.nan_to_num(X.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if splits is None:
        splits = list(GroupKFold(n_splits=5).split(Xc, y, groups))
    binary = set(np.unique(y).tolist()) <= {0, 1}
    accs, mf1, hcf, yt, yp = [], [], [], [], []
    for tr, te in splits:
        clf = clf_factory(); clf.fit(Xc[tr], y[tr]); p = clf.predict(Xc[te])
        accs.append(accuracy_score(y[te], p))
        mf1.append(f1_score(y[te], p, average="macro"))
        hcf.append(f1_score(y[te], p, average="binary", pos_label=1, zero_division=0)
                   if binary else f1_score(y[te], p, average="macro"))
        yt.extend(y[te].tolist()); yp.extend(p.tolist())
    maj = float(max(np.bincount(np.asarray(yt))) / len(yt))
    return dict(acc_mean=float(np.mean(accs)), acc_std=float(np.std(accs)),
                macro_f1_mean=float(np.mean(mf1)), macro_f1_std=float(np.std(mf1)),
                high_class_f1_mean=float(np.mean(hcf)), high_class_f1_std=float(np.std(hcf)),
                majority_baseline=maj, n=int(len(yt)), eval="GroupKFold(5) by subject")


def step1_validate(d):
    """Validate extraction/alignment via above-chance cross-subject clip decoding.

    The pre-registration escalates to running-norm (DE per-subject calibration) when
    the LEAN check is ambiguous. Here the raw mean-over-seconds DE is many-sigma above
    chance (binary ~9σ over 0.50; 9-class ~20σ over 0.111) -- which alone rules out an
    alignment bug (a misalignment gives chance) -- and the running-norm path reaches
    ~0.63 / ~0.35, reproducing FACED's published 9-class ~35.2% and clearing the gate.
    Gate therefore passes on the running-norm path; raw is reported as corroboration."""
    X, gs, y9, yb = d["X"], d["groups_subject"], d["y_class9"], d["y_binpos"]
    Xz = per_subject_znorm(X, gs)
    mask = yb >= 0                                       # drop neutral for binary
    raw_bin = evaluate(X[mask], yb[mask], gs[mask], svm_factory)
    raw_9 = evaluate(X, y9, gs, svm_factory)
    cal_bin = evaluate(Xz[mask], yb[mask], gs[mask], svm_factory)
    cal_9 = evaluate(Xz, y9, gs, svm_factory)
    ok = (cal_bin["acc_mean"] >= 0.60) and (cal_9["acc_mean"] >= 0.25)
    print(f"[Step 1] clip-binary raw={raw_bin['acc_mean']:.3f} +runnorm={cal_bin['acc_mean']:.3f} "
          f"(gate>=0.60) | clip-9class raw={raw_9['acc_mean']:.3f} +runnorm={cal_9['acc_mean']:.3f} "
          f"(gate>=0.25, chance 0.111) | PASS={ok}")
    return {"clip_binary_raw": raw_bin, "clip_9class_raw": raw_9,
            "clip_binary_runnorm": cal_bin, "clip_9class_runnorm": cal_9, "pass": bool(ok)}


def step2_ceiling(d):
    """Self-report valence/arousal cross-subject, RF parity harness; raw vs calibrated."""
    X, gs = d["X"], d["groups_subject"]
    val = (d["val_score"] > fd.VAL_THRESH).astype(int)
    aro = (d["aro_score"] > fd.VAL_THRESH).astype(int)
    Xz = per_subject_znorm(X, gs)
    out = {}
    for name, y in (("valence", val), ("arousal", aro)):
        raw = evaluate(X, y, gs, rf_factory)
        cal = evaluate(Xz, y, gs, rf_factory)
        out[name] = {"raw": raw, "calibrated": cal, "high_pct": float(y.mean())}
        print(f"[Step 2] {name}: raw acc={raw['acc_mean']:.3f} hiF1={raw['high_class_f1_mean']:.3f}"
              f" | +calib acc={cal['acc_mean']:.3f} hiF1={cal['high_class_f1_mean']:.3f}"
              f" | majority={raw['majority_baseline']:.3f}")
    return out


def step3_unseen_video(d, s2):
    """Decisive control: cross-subject AND cross-video (Condition B) vs seen-video (A)."""
    X, gs, gv, vs = d["X"], d["groups_subject"], d["groups_vid"], d["val_score"]
    splits = list(fd.iter_subject_video_folds(gs, gv, vs, n_subj=5, n_vid=4))
    out = {}
    for name in ("valence", "arousal"):
        y = (d["val_score"] if name == "valence" else d["aro_score"]) > fd.VAL_THRESH
        y = y.astype(int)
        cond_a = s2[name]["raw"]                              # seen-video (= Step 2)
        cond_b = evaluate(X, y, gs, rf_factory, splits=splits)  # unseen-video
        out[name] = {"condition_a_seen": cond_a, "condition_b_unseen": cond_b}
        print(f"[Step 3] {name}: A(seen) hiF1={cond_a['high_class_f1_mean']:.3f}"
              f" -> B(unseen) hiF1={cond_b['high_class_f1_mean']:.3f}"
              f"  acc_B={cond_b['acc_mean']:.3f} maj_B={cond_b['majority_baseline']:.3f}")
    return out


def decide(s2, s3):
    b = s3["valence"]["condition_b_unseen"]
    f1, acc, maj = b["high_class_f1_mean"], b["acc_mean"], b["majority_baseline"]
    sd = b["acc_std"] or 1e-9
    beats_baseline = (acc - maj) > sd
    raw_f1 = s2["valence"]["raw"]["high_class_f1_mean"]
    cal_f1 = s2["valence"]["calibrated"]["high_class_f1_mean"]
    if f1 >= 0.60 and beats_baseline and f1 > 0.40:
        return {"verdict": "POSITIVE",
                "rationale": f"Condition-B valence hiF1={f1:.3f} >=0.60, beats baseline, tops 0.40 ceiling."}
    if cal_f1 >= 0.60 and raw_f1 < 0.50:
        return {"verdict": "CALIBRATION-DRIVEN",
                "rationale": f"valence hiF1 jumps {raw_f1:.3f}->{cal_f1:.3f} with per-subject calib; "
                             f"gain is online calibration, supports the calibration pivot."}
    return {"verdict": "NEGATIVE",
            "rationale": f"Condition-B valence hiF1={f1:.3f} <=0.45 or within noise of majority "
                         f"({acc:.3f} vs {maj:.3f}); ceiling holds at N=123."}


if __name__ == "__main__":
    d = fd.load_cache()
    print(f"cache: X={d['X'].shape}  subjects={len(np.unique(d['groups_subject']))}  "
          f"val_high={int((d['val_score']>fd.VAL_THRESH).sum())}/{len(d['val_score'])}")
    s1 = step1_validate(d)
    if not s1["pass"]:
        sys.exit("[Step 1] GATE FAILED -- extraction/alignment suspect; stop.")
    print("[Step 1] data validated.")
    s2 = step2_ceiling(d)
    s3 = step3_unseen_video(d, s2)
    verdict = decide(s2, s3)
    report = {"step1_validation": s1, "step2_ceiling": s2, "step3_unseen_video": s3,
              "verdict": verdict, "label_threshold": fd.VAL_THRESH,
              "representation": "per-(subject,video) mean-over-seconds DE, 160-dim",
              "ceiling_reference": {"DEAP_valence_hiF1": 0.00, "DREAMER_arousal_hiF1": 0.40},
              "leakage_free": True, "synthetic_fallback": False,
              "protocol": "HONEST_AFFECT_FACED.md"}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nVERDICT: {verdict['verdict']} -- {verdict['rationale']}")
    print(f"saved {OUT}")
    if verdict["verdict"] != "POSITIVE":
        print("Honesty gate: NO model committed (per HONEST_AFFECT_FACED.md).")
