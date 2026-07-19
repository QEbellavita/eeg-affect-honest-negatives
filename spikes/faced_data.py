#!/usr/bin/env python3
"""FACED honest-spike data layer: DE features + self-report labels -> cached
per-(subject,video) matrix, joined by vid, with a Gate-0 sanity assertion.
Reads directly from EEG_Features.zip / Code.zip -- no full extraction, no raw
Data/ download needed. See HONEST_AFFECT_FACED.md for the protocol."""
import os
import io
import zipfile
import pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DL = os.path.expanduser(os.environ.get("FACED_DL", "~/Downloads"))
FEATURES_ZIP = os.environ.get("FACED_FEATURES_ZIP", os.path.join(DL, "EEG_Features.zip"))
CODE_ZIP = os.environ.get("FACED_CODE_ZIP", os.path.join(DL, "Code.zip"))
STIMULI_XLSX = os.environ.get("FACED_STIMULI", os.path.join(DL, "Stimuli_info.xlsx"))
# Derived from licensed data, so never committed — see .gitignore and the README.
CACHE = os.environ.get("FACED_CACHE",
                       os.path.join(HERE, "..", "cache", "faced_features_cache.npz"))

N_VIDS, N_ELEC, N_SEC, N_BAND = 28, 32, 30, 5
N_SUBJECTS = 123
ITEMS = ["Joy", "Tenderness", "Inspiration", "Amusement", "Anger", "Disgust",
         "Fear", "Sadness", "Arousal", "Valence", "Familiarity", "Liking"]
VAL_IDX, ARO_IDX, VAL_THRESH = 9, 8, 3.5
CLASS9 = ["Anger", "Disgust", "Fear", "Sadness", "Neutral",
          "Amusement", "Inspiration", "Joy", "Tenderness"]
_CLASS9_IDX = {c: i for i, c in enumerate(CLASS9)}


# ----------------------------------------------------------------- stimuli map
def stimuli_map(path=STIMULI_XLSX):
    """vid(1-28) -> (Targeted Emotion or 'Neutral', valence sign)."""
    import pandas as pd
    df = pd.read_excel(path)
    # Sheet has trailing free-text "Notes:" rows; keep only numeric Video index.
    df = df[pd.to_numeric(df["Video index"], errors="coerce").notna()]
    emo, sign = {}, {}
    for _, r in df.iterrows():
        vid = int(r["Video index"])
        s = str(r["Valence"]).strip()                       # Negative/Neutral/Positive
        emo[vid] = "Neutral" if s == "Neutral" else str(r["Targeted Emotion"]).strip()
        sign[vid] = s
    return emo, sign


def vid_to_class9(emo):
    return {vid: _CLASS9_IDX[e] for vid, e in emo.items()}


def vid_to_binpos(sign):
    return {vid: (1 if s == "Positive" else 0 if s == "Negative" else None)
            for vid, s in sign.items()}


# --------------------------------------------------------------------- loaders
def _find_member(z, suffix):
    for n in z.namelist():
        if n.endswith(suffix):
            return n
    raise FileNotFoundError(suffix)


def load_subject_de(sub, features_zip=FEATURES_ZIP):
    """Return DE tensor (28,32,30,5) for sub from the features zip."""
    with zipfile.ZipFile(features_zip) as z:
        name = _find_member(z, f"DE/{sub}.pkl.pkl")
        # Trusted source: official FACED EEG_Features.zip from Synapse (the only
        # format it ships in; manually downloaded by the user). Not untrusted input.
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


# ----------------------------------------------------------------------- cache
def build_cache(subjects=None, save=True):
    """Build the per-(subject,video) cache. DE position p <-> vid p+1; labels joined
    by vid. Runs the Gate-0 sanity assertion before saving/returning."""
    emo, sign = stimuli_map()
    c9, cbin = vid_to_class9(emo), vid_to_binpos(sign)
    subjects = subjects or [f"sub{i:03d}" for i in range(N_SUBJECTS)]
    X, gs, gv, y9, yb, vsc, asc = [], [], [], [], [], [], []
    for si, sub in enumerate(subjects):
        de = load_subject_de(sub)                                 # (28,32,30,5)
        feat = de.mean(axis=2).reshape(N_VIDS, N_ELEC * N_BAND)   # (28,160) mean over seconds
        labs = load_subject_labels_by_vid(sub)                    # vid -> score(12,)
        for p in range(N_VIDS):
            vid = p + 1                                           # alignment: DE[p] <-> vid p+1
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


# --------------------------------------------------- nested subject×video folds
def iter_subject_video_folds(groups_subject, groups_vid, val_score=None,
                             n_subj=5, n_vid=4, seed=42):
    """Nested cross-subject x cross-video folds. Test rows = held-out subjects
    AND held-out videos; train rows = remaining subjects AND remaining videos.
    Video folds are valence-balanced (round-robin over valence rank) so both
    classes appear on each side."""
    rng = np.random.RandomState(seed)
    subs = np.unique(groups_subject); rng.shuffle(subs)
    subj_folds = np.array_split(subs, n_subj)
    vids = np.unique(groups_vid)
    if val_score is not None:
        vids = np.array(sorted(vids, key=lambda v: val_score[groups_vid == v].mean()))
    vid_folds = [vids[i::n_vid] for i in range(n_vid)]            # round-robin by valence rank
    for vf in vid_folds:
        for sf in subj_folds:
            te = np.isin(groups_subject, sf) & np.isin(groups_vid, vf)
            tr = ~np.isin(groups_subject, sf) & ~np.isin(groups_vid, vf)
            if te.sum() and tr.sum():
                yield np.where(tr)[0], np.where(te)[0]


if __name__ == "__main__":
    build_cache()
