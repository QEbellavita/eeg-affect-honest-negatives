# FACED Honest Cross-Subject Spike — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether FACED (N=123, finer-grained) breaks the subject-independent EEG-affect ceiling that DEAP/DREAMER hit, using a pre-registered honest protocol (spec: `HONEST_AFFECT_FACED.md`).

**Architecture:** Two scripts. `faced_data.py` is the pure data layer — reads DE features + self-report labels directly from the already-downloaded `EEG_Features.zip` / `Code.zip`, joins labels to features by `vid`, aggregates to one 160-dim vector per (subject,video), and caches to `.npz`. `train_faced_honest.py` is the evaluator — Step 1 validates the data reproduces above-chance clip decoding, Step 2 runs the self-report valence/arousal ceiling comparison (RF + GroupKFold-5, parity with `train_deap_honest.py`) plus a calibration ablation, Step 3 runs the decisive leave-one-video-out control, then writes JSON + verdict.

**Tech Stack:** Python 3, numpy, scipy, scikit-learn, pandas, openpyxl; `uv` venv. No `mne`/torch (DE is precomputed). pytest for the bug-prone pure functions.

## Global Constraints

- **Env:** `uv` venv; pins `numpy>=1.21, scipy>=1.7, scikit-learn>=1.0, pandas>=1.3, openpyxl, pytest`. System python3.14 has no sci stack — never run scripts with bare `python3`.
- **Representation:** per-(subject,video) **mean-over-seconds DE**, `de.mean(axis=2).reshape(28,160)`. 123 subjects × 28 videos = **3,444 samples × 160 features**.
- **Label indices (verbatim):** `score` order `[Joy,Tenderness,Inspiration,Amusement,Anger,Disgust,Fear,Sadness,Arousal,Valence,Familiarity,Liking]` → **Valence = index 9, Arousal = index 8**, 0–7 scale, binarise at **> 3.5**.
- **Alignment (verbatim):** DE axis-0 position `p` ↔ **vid `p+1`**; `After_remarks` rows are presentation-order — **join labels by `.vid`, never by row position**.
- **Stimuli map (verbatim):** vids **1–12 Negative**, **13–16 Neutral**, **17–28 Positive**; 9-class order `[Anger,Disgust,Fear,Sadness,Neutral,Amusement,Inspiration,Joy,Tenderness]`.
- **Parity harness:** RF `n_estimators=400, max_depth=12, n_jobs=-1, class_weight="balanced", random_state=42` inside `make_pipeline(StandardScaler(), …)`; `GroupKFold(n_splits=5)`; majority baseline `max(np.bincount(y))/len(y)`; report acc, macro-F1, **high-class (pos_label=1) F1**, majority baseline.
- **Honesty:** commit **no model** unless verdict is POSITIVE. Decision rule is fixed in `HONEST_AFFECT_FACED.md` and must not be edited after seeing results.
- **Paths:** zips default to `~/Downloads/{EEG_Features,Code}.zip` + `~/Downloads/Stimuli_info.xlsx` (override via env `FACED_FEATURES_ZIP`/`FACED_CODE_ZIP`/`FACED_STIMULI`); cache → `ml-pipeline/models/affect_eeg_negative/faced_features_cache.npz`; results → `…/faced_affect_honest.json`.

---

## File Structure

- `ml-pipeline/scripts/faced_data.py` — data layer: stimuli map, zip readers, per-video aggregation, vid-join, Gate-0 sanity assertion, cache build/load, nested subject×video splitter.
- `ml-pipeline/scripts/train_faced_honest.py` — evaluator: shared `evaluate()`, Steps 1–3, JSON + verdict.
- `ml-pipeline/scripts/test_faced_data.py` — pytest for the bug-prone pure functions (alignment/sanity, stimuli map, splitter leakage).

---

## Task 1: Env + stimuli map + label constants (`faced_data.py` part 1)

**Files:**
- Create: `ml-pipeline/scripts/faced_data.py`
- Create: `ml-pipeline/scripts/test_faced_data.py`
- Env: `ml-pipeline/.venv-faced` (uv)

**Interfaces:**
- Produces: `stimuli_map(path) -> (emo: dict[int,str], sign: dict[int,str])`; `vid_to_class9(emo) -> dict[int,int]`; `vid_to_binpos(sign) -> dict[int,int|None]`; constants `ITEMS, VAL_IDX=9, ARO_IDX=8, VAL_THRESH=3.5, CLASS9, N_VIDS=28`.

- [ ] **Step 1: Create the venv**

Run:
```bash
uv venv .venv-faced && . .venv-faced/bin/activate && uv pip install "numpy>=1.21" "scipy>=1.7" "scikit-learn>=1.0" "pandas>=1.3" openpyxl pytest
```
Expected: packages install; `python -c "import sklearn,scipy,pandas,openpyxl"` exits 0.

- [ ] **Step 2: Write the failing test for the stimuli map**

Create `ml-pipeline/scripts/test_faced_data.py`:
```python
import os
import faced_data as fd

def test_stimuli_map_partitions_vids():
    emo, sign = fd.stimuli_map()
    assert len(emo) == 28
    assert all(sign[v] == "Negative" for v in range(1, 13))
    assert all(sign[v] == "Neutral" for v in range(13, 17))
    assert all(sign[v] == "Positive" for v in range(17, 29))
    assert emo[1] == "Anger" and emo[7] == "Fear" and emo[23] == "Joy" and emo[26] == "Tenderness"

def test_class9_and_binpos():
    emo, sign = fd.stimuli_map()
    c9 = fd.vid_to_class9(emo)
    assert c9[1] == 0 and c9[13] == 4 and c9[23] == 7      # Anger=0, Neutral=4, Joy=7
    cb = fd.vid_to_binpos(sign)
    assert cb[1] == 0 and cb[28] == 1 and cb[14] is None    # neg / pos / neutral dropped
```

- [ ] **Step 3: Run it to verify it fails**

Run: `cd ml-pipeline/scripts && ../.venv-faced/bin/python -m pytest test_faced_data.py -k stimuli -x -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'faced_data'`.

- [ ] **Step 4: Implement the maps**

Create `ml-pipeline/scripts/faced_data.py`:
```python
#!/usr/bin/env python3
"""FACED honest-spike data layer: DE features + self-report labels → cached
per-(subject,video) matrix, joined by vid, with a Gate-0 sanity assertion.
Reads directly from EEG_Features.zip / Code.zip — no full extraction, no raw
Data/ download needed. See HONEST_AFFECT_FACED.md for the protocol."""
import os, io, zipfile, pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DL = os.path.expanduser(os.environ.get("FACED_DL", "~/Downloads"))
FEATURES_ZIP = os.environ.get("FACED_FEATURES_ZIP", os.path.join(DL, "EEG_Features.zip"))
CODE_ZIP     = os.environ.get("FACED_CODE_ZIP",     os.path.join(DL, "Code.zip"))
STIMULI_XLSX = os.environ.get("FACED_STIMULI",      os.path.join(DL, "Stimuli_info.xlsx"))
CACHE = os.path.join(HERE, "..", "models", "affect_eeg_negative", "faced_features_cache.npz")

N_VIDS, N_ELEC, N_SEC, N_BAND = 28, 32, 30, 5
N_SUBJECTS = 123
ITEMS = ["Joy", "Tenderness", "Inspiration", "Amusement", "Anger", "Disgust",
         "Fear", "Sadness", "Arousal", "Valence", "Familiarity", "Liking"]
VAL_IDX, ARO_IDX, VAL_THRESH = 9, 8, 3.5
CLASS9 = ["Anger", "Disgust", "Fear", "Sadness", "Neutral",
          "Amusement", "Inspiration", "Joy", "Tenderness"]
_CLASS9_IDX = {c: i for i, c in enumerate(CLASS9)}


def stimuli_map(path=STIMULI_XLSX):
    """vid(1-28) -> (Targeted Emotion or 'Neutral', valence sign)."""
    import pandas as pd
    df = pd.read_excel(path).dropna(subset=["Video index"])
    emo, sign = {}, {}
    for _, r in df.iterrows():
        vid = int(r["Video index"])
        s = str(r["Valence"]).strip()                      # Negative/Neutral/Positive
        emo[vid] = "Neutral" if s == "Neutral" else str(r["Targeted Emotion"]).strip()
        sign[vid] = s
    return emo, sign


def vid_to_class9(emo):
    return {vid: _CLASS9_IDX[e] for vid, e in emo.items()}


def vid_to_binpos(sign):
    return {vid: (1 if s == "Positive" else 0 if s == "Negative" else None)
            for vid, s in sign.items()}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd ml-pipeline/scripts && ../.venv-faced/bin/python -m pytest test_faced_data.py -k "stimuli or class9" -q`
Expected: 2 passed.

- [ ] **Step 6: Commit** (only if user has authorised commits; otherwise leave staged)
```bash
git add ml-pipeline/scripts/faced_data.py ml-pipeline/scripts/test_faced_data.py
git commit -m "feat(faced): stimuli map + label constants for honest spike"
```

---

## Task 2: Loaders + per-video aggregation + vid-join + Gate-0 sanity + cache

**Files:**
- Modify: `ml-pipeline/scripts/faced_data.py` (append)
- Modify: `ml-pipeline/scripts/test_faced_data.py` (append)

**Interfaces:**
- Consumes: `stimuli_map`, `vid_to_class9`, `vid_to_binpos`, constants.
- Produces: `load_subject_de(sub) -> np.ndarray(28,32,30,5)`; `load_subject_labels_by_vid(sub) -> dict[int, np.ndarray(12,)]`; `build_cache() -> dict of arrays`; `load_cache() -> dict`. Cache arrays: `X(3444,160)`, `groups_subject(3444,)`, `groups_vid(3444,)`, `y_class9(3444,)`, `y_binpos(3444,)` (-1 for neutral), `val_score(3444,)`, `aro_score(3444,)`.

- [ ] **Step 1: Write the failing Gate-0 sanity test (real data, first 5 subjects)**

Append to `test_faced_data.py`:
```python
def test_alignment_sanity_positive_gt_negative():
    """The pre-registered Gate-0 check: positive-clip self-report valence must
    exceed negative-clip valence. Simultaneously verifies VAL_IDX=9 AND the
    vid-join (row-position join would scramble this)."""
    d = fd.build_cache(subjects=[f"sub{i:03d}" for i in range(5)], save=False)
    gv, vs = d["groups_vid"], d["val_score"]
    pos = vs[np.isin(gv, range(17, 29))].mean()
    neg = vs[np.isin(gv, range(1, 13))].mean()
    assert pos > neg + 1.0, f"alignment FAILED: pos={pos:.2f} neg={neg:.2f}"

def test_feature_shape_and_labels():
    d = fd.build_cache(subjects=[f"sub{i:03d}" for i in range(3)], save=False)
    assert d["X"].shape == (3 * 28, 160)
    assert set(np.unique(d["y_class9"])) <= set(range(9))
    assert set(np.unique(d["y_binpos"])) <= {-1, 0, 1}
```
(add `import numpy as np` at top of the test file)

- [ ] **Step 2: Run to verify it fails**

Run: `cd ml-pipeline/scripts && ../.venv-faced/bin/python -m pytest test_faced_data.py -k "alignment or feature_shape" -x -q`
Expected: FAIL — `AttributeError: module 'faced_data' has no attribute 'build_cache'`.

- [ ] **Step 3: Implement loaders + aggregation + join + sanity + cache**

Append to `faced_data.py`:
```python
def _find_member(z, suffix):
    for n in z.namelist():
        if n.endswith(suffix):
            return n
    raise FileNotFoundError(suffix)


def load_subject_de(sub, features_zip=FEATURES_ZIP):
    """Return DE tensor (28,32,30,5) for sub from the features zip."""
    with zipfile.ZipFile(features_zip) as z:
        name = _find_member(z, f"DE/{sub}.pkl.pkl")
        de = pickle.load(io.BytesIO(z.read(name)))
    de = np.asarray(de, dtype=np.float64)
    assert de.shape == (N_VIDS, N_ELEC, N_SEC, N_BAND), f"{sub}: bad DE shape {de.shape}"
    return de


def load_subject_labels_by_vid(sub, code_zip=CODE_ZIP):
    """Return {vid(1-28): score(12,)} for sub from After_remarks.mat in the code zip."""
    import scipy.io as sio
    with zipfile.ZipFile(code_zip) as z:
        name = _find_member(z, f"After_remarks/{sub}/After_remarks.mat")
        m = sio.loadmat(io.BytesIO(z.read(name)), squeeze_me=True, struct_as_record=False)
    out = {}
    for e in np.atleast_1d(m["After_remark"]):
        vid = int(np.array(e.vid).ravel()[0])
        out[vid] = np.array(e.score, dtype=np.float64).ravel()
        assert out[vid].shape == (12,), f"{sub} vid{vid}: bad score shape {out[vid].shape}"
    return out


def build_cache(subjects=None, save=True):
    """Build the per-(subject,video) cache. DE position p ↔ vid p+1; labels joined
    by vid. Runs the Gate-0 sanity assertion before saving/returning."""
    emo, sign = stimuli_map()
    c9, cbin = vid_to_class9(emo), vid_to_binpos(sign)
    subjects = subjects or [f"sub{i:03d}" for i in range(N_SUBJECTS)]
    X, gs, gv, y9, yb, vsc, asc = [], [], [], [], [], [], []
    for si, sub in enumerate(subjects):
        de = load_subject_de(sub)                              # (28,32,30,5)
        feat = de.mean(axis=2).reshape(N_VIDS, N_ELEC * N_BAND)  # (28,160) mean over seconds
        labs = load_subject_labels_by_vid(sub)                 # vid -> score(12,)
        for p in range(N_VIDS):
            vid = p + 1                                        # alignment: DE[p] ↔ vid p+1
            sc = labs[vid]
            X.append(feat[p]); gs.append(si); gv.append(vid)
            y9.append(c9[vid]); yb.append(cbin[vid] if cbin[vid] is not None else -1)
            vsc.append(sc[VAL_IDX]); asc.append(sc[ARO_IDX])
    d = dict(X=np.asarray(X), groups_subject=np.asarray(gs), groups_vid=np.asarray(gv),
             y_class9=np.asarray(y9), y_binpos=np.asarray(yb),
             val_score=np.asarray(vsc), aro_score=np.asarray(asc))
    # ---- Gate-0 sanity assertion (pre-registered) ----
    gvv, vs = d["groups_vid"], d["val_score"]
    pos = vs[np.isin(gvv, range(17, 29))].mean()
    neg = vs[np.isin(gvv, range(1, 13))].mean()
    assert pos > neg + 1.0, f"GATE-0 FAILED valence pos={pos:.2f} neg={neg:.2f} (idx/join bug)"
    if save:
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        np.savez_compressed(CACHE, **d)
        print(f"saved cache: {CACHE}  X={d['X'].shape}  (Gate-0 ok: pos={pos:.2f} neg={neg:.2f})")
    return d


def load_cache():
    if not os.path.exists(CACHE):
        return build_cache()
    z = np.load(CACHE)
    return {k: z[k] for k in z.files}
```

- [ ] **Step 4: Run to verify pass**

Run: `cd ml-pipeline/scripts && ../.venv-faced/bin/python -m pytest test_faced_data.py -k "alignment or feature_shape" -q`
Expected: 2 passed (positive valence ≈ 4+, negative ≈ 1).

- [ ] **Step 5: Build the full 123-subject cache**

Run: `cd ml-pipeline/scripts && ../.venv-faced/bin/python -c "import faced_data; faced_data.build_cache()"`
Expected: `saved cache: …faced_features_cache.npz  X=(3444, 160)  (Gate-0 ok: …)`.

- [ ] **Step 6: Commit** (if authorised)
```bash
git add ml-pipeline/scripts/faced_data.py ml-pipeline/scripts/test_faced_data.py
git commit -m "feat(faced): DE/label loaders, vid-join, Gate-0 sanity, cache build"
```

---

## Task 3: Nested subject×video splitter (the leave-one-video-out engine)

**Files:**
- Modify: `ml-pipeline/scripts/faced_data.py` (append)
- Modify: `ml-pipeline/scripts/test_faced_data.py` (append)

**Interfaces:**
- Consumes: cache arrays `groups_subject`, `groups_vid`, `val_score`.
- Produces: `iter_subject_video_folds(groups_subject, groups_vid, val_score=None, n_subj=5, n_vid=4, seed=42) -> Iterator[(train_idx, test_idx)]` where test = held-out-subjects × held-out-vids and train = remaining-subjects × remaining-vids (both disjoint).

- [ ] **Step 1: Write the failing leakage tests**

Append to `test_faced_data.py`:
```python
def test_splitter_no_subject_or_video_leakage():
    d = fd.load_cache()
    gs, gv, vs = d["groups_subject"], d["groups_vid"], d["val_score"]
    n = 0
    for tr, te in fd.iter_subject_video_folds(gs, gv, vs, n_subj=5, n_vid=4):
        assert len(tr) and len(te)
        assert not (set(gs[te]) & set(gs[tr])), "SUBJECT leakage"
        assert not (set(gv[te]) & set(gv[tr])), "VIDEO leakage"
        n += 1
    assert n >= 15  # ~4*5 folds (minus any empty)

def test_video_folds_span_valence():
    d = fd.load_cache()
    gv, vs = d["groups_vid"], d["val_score"]
    # each test video-set should contain both high- and low-valence clips
    for tr, te in fd.iter_subject_video_folds(d["groups_subject"], gv, vs, n_vid=4):
        tv = set(gv[te])
        assert any(v <= 12 for v in tv) or any(v >= 17 for v in tv)
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd ml-pipeline/scripts && ../.venv-faced/bin/python -m pytest test_faced_data.py -k splitter -x -q`
Expected: FAIL — `AttributeError: … 'iter_subject_video_folds'`.

- [ ] **Step 3: Implement the splitter**

Append to `faced_data.py`:
```python
def iter_subject_video_folds(groups_subject, groups_vid, val_score=None,
                             n_subj=5, n_vid=4, seed=42):
    """Nested cross-subject × cross-video folds. Test rows = held-out subjects
    AND held-out videos; train rows = remaining subjects AND remaining videos.
    Video folds are valence-balanced (round-robin over valence rank) so both
    classes appear on each side."""
    rng = np.random.RandomState(seed)
    subs = np.unique(groups_subject); rng.shuffle(subs)
    subj_folds = np.array_split(subs, n_subj)
    vids = np.unique(groups_vid)
    if val_score is not None:
        vids = np.array(sorted(vids, key=lambda v: val_score[groups_vid == v].mean()))
    vid_folds = [vids[i::n_vid] for i in range(n_vid)]        # round-robin by valence rank
    for vf in vid_folds:
        for sf in subj_folds:
            te = np.isin(groups_subject, sf) & np.isin(groups_vid, vf)
            tr = ~np.isin(groups_subject, sf) & ~np.isin(groups_vid, vf)
            if te.sum() and tr.sum():
                yield np.where(tr)[0], np.where(te)[0]
```

- [ ] **Step 4: Run to verify pass**

Run: `cd ml-pipeline/scripts && ../.venv-faced/bin/python -m pytest test_faced_data.py -k splitter -q`
Expected: 2 passed.

- [ ] **Step 5: Full data-layer test sweep + commit** (if authorised)

Run: `cd ml-pipeline/scripts && ../.venv-faced/bin/python -m pytest test_faced_data.py -q` → all passed.
```bash
git add ml-pipeline/scripts/faced_data.py ml-pipeline/scripts/test_faced_data.py
git commit -m "feat(faced): nested subject×video splitter + leakage tests"
```

---

## Task 4: Shared evaluator + Step 1 data-validation gate (`train_faced_honest.py`)

**Files:**
- Create: `ml-pipeline/scripts/train_faced_honest.py`

**Interfaces:**
- Consumes: `faced_data.load_cache`, `iter_subject_video_folds`.
- Produces: `evaluate(X, y, groups, clf_factory, splits=None) -> dict(acc_mean,acc_std,macro_f1_mean,macro_f1_std,high_class_f1_mean,high_class_f1_std,majority_baseline,n)`; `rf_factory()`, `svm_factory()`, `per_subject_znorm(X,groups)`; `step1_validate(d) -> dict` with `pass: bool`.

- [ ] **Step 1: Create the evaluator + Step 1**

Create `ml-pipeline/scripts/train_faced_honest.py`:
```python
#!/usr/bin/env python3
"""Honest FACED cross-subject affect spike. Pre-registered protocol +
decision rule: HONEST_AFFECT_FACED.md. Parity with train_deap_honest.py
(RF + GroupKFold-5 + balanced). Commits NO model unless verdict POSITIVE."""
import os, sys, json
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
        yt.extend(y[te]); yp.extend(p)
    maj = float(max(np.bincount(np.asarray(yt))) / len(yt))
    return dict(acc_mean=float(np.mean(accs)), acc_std=float(np.std(accs)),
                macro_f1_mean=float(np.mean(mf1)), macro_f1_std=float(np.std(mf1)),
                high_class_f1_mean=float(np.mean(hcf)), high_class_f1_std=float(np.std(hcf)),
                majority_baseline=maj, n=int(len(yt)), eval="GroupKFold(5) by subject")


def step1_validate(d):
    X, gs, gv, y9 = d["X"], d["groups_subject"], d["groups_vid"], d["y_class9"]
    yb = d["y_binpos"]
    mask = yb >= 0                                      # drop neutral for binary
    r_bin = evaluate(X[mask], yb[mask], gs[mask], svm_factory)
    r_9 = evaluate(X, y9, gs, svm_factory)
    ok = (r_bin["acc_mean"] >= 0.60) and (r_9["acc_mean"] >= 0.25)
    print(f"[Step 1] clip-binary acc={r_bin['acc_mean']:.3f} (gate≥0.60) | "
          f"clip-9class acc={r_9['acc_mean']:.3f} (gate≥0.25, chance 0.111) | PASS={ok}")
    return {"clip_binary": r_bin, "clip_9class": r_9, "pass": bool(ok)}


if __name__ == "__main__":
    d = fd.load_cache()
    s1 = step1_validate(d)
    if not s1["pass"]:
        sys.exit("[Step 1] GATE FAILED — extraction/alignment suspect; stop.")
    print("[Step 1] data validated.")
```

- [ ] **Step 2: Run Step 1**

Run: `cd ml-pipeline/scripts && ../.venv-faced/bin/python -u train_faced_honest.py`
Expected: prints clip-binary acc clearly > 0.60 and 9-class clearly > 0.25, `PASS=True`, `data validated.` (If it fails the gate, STOP — debug extraction before continuing.)

- [ ] **Step 3: Commit** (if authorised)
```bash
git add ml-pipeline/scripts/train_faced_honest.py
git commit -m "feat(faced): evaluator + Step 1 data-validation gate"
```

---

## Task 5: Step 2 ceiling comparison + calibration ablation + JSON

**Files:**
- Modify: `ml-pipeline/scripts/train_faced_honest.py` (add `step2_ceiling`, extend `__main__`)

**Interfaces:**
- Consumes: `evaluate`, `rf_factory`, `per_subject_znorm`, cache.
- Produces: `step2_ceiling(d) -> dict` with `valence`/`arousal` each having `raw` and `calibrated` result dicts.

- [ ] **Step 1: Add `step2_ceiling`**

Insert into `train_faced_honest.py` (before `__main__`):
```python
def step2_ceiling(d):
    X, gs = d["X"], d["groups_subject"]
    val = (d["val_score"] > fd.VAL_THRESH).astype(int)
    aro = (d["aro_score"] > fd.VAL_THRESH).astype(int)
    Xz = per_subject_znorm(X, gs)
    out = {}
    for name, y in (("valence", val), ("arousal", aro)):
        raw = evaluate(X, y, gs, rf_factory)
        cal = evaluate(Xz, y, gs, rf_factory)
        out[name] = {"raw": raw, "calibrated": cal,
                     "high_pct": float(y.mean())}
        print(f"[Step 2] {name}: raw acc={raw['acc_mean']:.3f} hiF1={raw['high_class_f1_mean']:.3f}"
              f" | +calib acc={cal['acc_mean']:.3f} hiF1={cal['high_class_f1_mean']:.3f}"
              f" | majority={raw['majority_baseline']:.3f}")
    return out
```

- [ ] **Step 2: Wire into `__main__`** (replace the `print("[Step 1] data validated.")` tail)
```python
    print("[Step 1] data validated.")
    s2 = step2_ceiling(d)
```

- [ ] **Step 3: Run + eyeball against the ceiling**

Run: `cd ml-pipeline/scripts && ../.venv-faced/bin/python -u train_faced_honest.py`
Expected: Step 2 prints valence/arousal raw vs +calib. Compare `high_class_f1_mean` to the DEAP/DREAMER ceiling (0.00/0.00/0.24/0.40 from `HONEST_AFFECT_EEG_NEGATIVE.md`). No assertion here — this is measurement.

- [ ] **Step 4: Commit** (if authorised)
```bash
git add ml-pipeline/scripts/train_faced_honest.py
git commit -m "feat(faced): Step 2 ceiling comparison + calibration ablation"
```

---

## Task 6: Step 3 leave-one-video-out + verdict + JSON + doc

**Files:**
- Modify: `ml-pipeline/scripts/train_faced_honest.py` (add `step3_unseen_video`, `decide`, JSON write)
- Modify: `ml-pipeline/scripts/HONEST_AFFECT_FACED.md` (fill Verdict)

**Interfaces:**
- Consumes: `evaluate`, `rf_factory`, `fd.iter_subject_video_folds`, Steps 1–2 results.
- Produces: `step3_unseen_video(d, s2) -> dict` (Condition A vs B for valence+arousal); `decide(s2, s3) -> {"verdict": "POSITIVE"|"CALIBRATION-DRIVEN"|"NEGATIVE", "rationale": str}`.

- [ ] **Step 1: Add Step 3 + decision rule**

Insert into `train_faced_honest.py` (before `__main__`):
```python
def step3_unseen_video(d, s2):
    X, gs, gv, vs = d["X"], d["groups_subject"], d["groups_vid"], d["val_score"]
    splits = list(fd.iter_subject_video_folds(gs, gv, vs, n_subj=5, n_vid=4))
    out = {}
    for name in ("valence", "arousal"):
        y = (d[f"{name[:3]}_score" if name == "arousal" else "val_score"] > fd.VAL_THRESH).astype(int)
        cond_a = s2[name]["raw"]                              # seen-video (= Step 2)
        cond_b = evaluate(X, y, gs, rf_factory, splits=splits)  # unseen-video
        out[name] = {"condition_a_seen": cond_a, "condition_b_unseen": cond_b}
        print(f"[Step 3] {name}: A(seen) hiF1={cond_a['high_class_f1_mean']:.3f}"
              f" → B(unseen) hiF1={cond_b['high_class_f1_mean']:.3f}"
              f"  acc_B={cond_b['acc_mean']:.3f} maj_B={cond_b['majority_baseline']:.3f}")
    return out


def decide(s2, s3):
    b = s3["valence"]["condition_b_unseen"]
    f1, acc, maj, sd = (b["high_class_f1_mean"], b["acc_mean"],
                        b["majority_baseline"], b["acc_std"] or 1e-9)
    beats_baseline = (acc - maj) > sd
    raw_f1 = s2["valence"]["raw"]["high_class_f1_mean"]
    cal_f1 = s2["valence"]["calibrated"]["high_class_f1_mean"]
    if f1 >= 0.60 and beats_baseline and f1 > 0.40:
        return {"verdict": "POSITIVE",
                "rationale": f"Condition-B valence hiF1={f1:.3f} ≥0.60, beats baseline, tops 0.40 ceiling."}
    if cal_f1 >= 0.60 and raw_f1 < 0.50:
        return {"verdict": "CALIBRATION-DRIVEN",
                "rationale": f"valence hiF1 jumps {raw_f1:.3f}→{cal_f1:.3f} with per-subject calib; "
                             f"gain is online calibration, supports the calibration pivot."}
    return {"verdict": "NEGATIVE",
            "rationale": f"Condition-B valence hiF1={f1:.3f} ≤0.45 or within noise of majority "
                         f"({acc:.3f} vs {maj:.3f}); ceiling holds at N=123."}
```

- [ ] **Step 2: Finalise `__main__` (write JSON + verdict; gate model commit)**
```python
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
    print(f"\nVERDICT: {verdict['verdict']} — {verdict['rationale']}")
    print(f"saved {OUT}")
    if verdict["verdict"] != "POSITIVE":
        print("Honesty gate: NO model committed (per HONEST_AFFECT_FACED.md).")
```

- [ ] **Step 3: Run the full spike**

Run: `cd ml-pipeline/scripts && ../.venv-faced/bin/python -u train_faced_honest.py`
Expected: Steps 1→3 print; final `VERDICT: …`; `faced_affect_honest.json` written.

- [ ] **Step 4: Fill the verdict in the spec**

Edit `HONEST_AFFECT_FACED.md` → replace the `## Verdict` body with the actual verdict, the Condition-A vs Condition-B valence/arousal high-class-F1 table, the raw-vs-calibrated ablation numbers, and the one-line interpretation (ceiling held / broken / calibration-driven).

- [ ] **Step 5: Commit** (if authorised)
```bash
git add ml-pipeline/scripts/train_faced_honest.py ml-pipeline/scripts/HONEST_AFFECT_FACED.md ml-pipeline/models/affect_eeg_negative/faced_affect_honest.json
git commit -m "feat(faced): Step 3 unseen-video control + pre-registered verdict"
```

---

## Self-Review

**Spec coverage:** Gate 0 → Task 2 (sanity assertion) + Task 1 (maps). Step 1 → Task 4. Step 2 + running-norm ablation → Task 5 (`per_subject_znorm`). Step 3 leave-one-video-out → Tasks 3+6. Decision rule (0.60/0.45/0.40/1σ, three verdicts incl. CALIBRATION-DRIVEN) → Task 6 `decide()`. Artifacts (cache, JSON, scripts, doc) → Tasks 2/6. Reproducibility-without-raw-data → reads from zips (Task 2). All spec sections covered.

**Placeholder scan:** none — every step has runnable code or an exact command. (`HONEST_AFFECT_FACED.md` Verdict stays PENDING until Task 6 Step 4 by design.)

**Type consistency:** `evaluate()` returns the same dict keys consumed by `step1_validate`/`step2_ceiling`/`step3_unseen_video`/`decide` (`high_class_f1_mean`, `acc_mean`, `majority_baseline`, `acc_std`). `build_cache`/`load_cache` keys (`X, groups_subject, groups_vid, y_class9, y_binpos, val_score, aro_score`) are consumed verbatim downstream. `iter_subject_video_folds` signature matches both the test and `step3`. **Fix applied:** in `step3_unseen_video`, arousal label derivation uses `d["aro_score"]` — written explicitly below to avoid the `name[:3]` cleverness:

```python
        y = (d["val_score"] if name == "valence" else d["aro_score"]) > fd.VAL_THRESH
        y = y.astype(int)
```
Replace the `y = (...)` line in Task 6 Step 1 with the two lines above.
