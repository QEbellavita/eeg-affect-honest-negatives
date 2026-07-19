import os

import numpy as np
import pytest

import faced_data as fd

# FACED is obtained separately under its own terms (see README), so these tests
# skip rather than fail on a clean clone. Set FACED_DL (or the per-file env vars
# in faced_data) to point at your copy.
needs_stimuli = pytest.mark.skipif(
    not os.path.exists(fd.STIMULI_XLSX),
    reason=f"FACED stimuli sheet not present at {fd.STIMULI_XLSX}",
)
needs_features = pytest.mark.skipif(
    not os.path.exists(fd.FEATURES_ZIP),
    reason=f"FACED EEG_Features.zip not present at {fd.FEATURES_ZIP}",
)
needs_cache = pytest.mark.skipif(
    not os.path.exists(fd.CACHE),
    reason=f"feature cache not built at {fd.CACHE} — run build_cache() first",
)


# ----- Task 1: stimuli map -----
@needs_stimuli
def test_stimuli_map_partitions_vids():
    emo, sign = fd.stimuli_map()
    assert len(emo) == 28
    assert all(sign[v] == "Negative" for v in range(1, 13))
    assert all(sign[v] == "Neutral" for v in range(13, 17))
    assert all(sign[v] == "Positive" for v in range(17, 29))
    assert emo[1] == "Anger" and emo[7] == "Fear" and emo[23] == "Joy" and emo[26] == "Tenderness"


@needs_stimuli
def test_class9_and_binpos():
    emo, sign = fd.stimuli_map()
    c9 = fd.vid_to_class9(emo)
    assert c9[1] == 0 and c9[13] == 4 and c9[23] == 7      # Anger=0, Neutral=4, Joy=7
    cb = fd.vid_to_binpos(sign)
    assert cb[1] == 0 and cb[28] == 1 and cb[14] is None    # neg / pos / neutral dropped


# ----- Task 2: Gate-0 alignment + shapes (real data, first few subjects) -----
@needs_stimuli
@needs_features
def test_alignment_sanity_positive_gt_negative():
    """Pre-registered Gate-0 check: positive-clip self-report valence must exceed
    negative-clip valence. Verifies VAL_IDX=9 AND the vid-join simultaneously."""
    d = fd.build_cache(subjects=[f"sub{i:03d}" for i in range(5)], save=False)
    gv, vs = d["groups_vid"], d["val_score"]
    pos = vs[np.isin(gv, range(17, 29))].mean()
    neg = vs[np.isin(gv, range(1, 13))].mean()
    assert pos > neg + 1.0, f"alignment FAILED: pos={pos:.2f} neg={neg:.2f}"


@needs_stimuli
@needs_features
def test_feature_shape_and_labels():
    d = fd.build_cache(subjects=[f"sub{i:03d}" for i in range(3)], save=False)
    assert d["X"].shape == (3 * 28, 160)
    assert set(np.unique(d["y_class9"])) <= set(range(9))
    assert set(np.unique(d["y_binpos"])) <= {-1, 0, 1}


# ----- Task 3: nested splitter leakage -----
@needs_cache
def test_splitter_no_subject_or_video_leakage():
    d = fd.load_cache()
    gs, gv, vs = d["groups_subject"], d["groups_vid"], d["val_score"]
    n = 0
    for tr, te in fd.iter_subject_video_folds(gs, gv, vs, n_subj=5, n_vid=4):
        assert len(tr) and len(te)
        assert not (set(gs[te]) & set(gs[tr])), "SUBJECT leakage"
        assert not (set(gv[te]) & set(gv[tr])), "VIDEO leakage"
        n += 1
    assert n >= 15  # ~4*5 folds


@needs_cache
def test_video_folds_span_valence():
    d = fd.load_cache()
    gv, vs = d["groups_vid"], d["val_score"]
    for tr, te in fd.iter_subject_video_folds(d["groups_subject"], gv, vs, n_vid=4):
        tv = set(int(v) for v in gv[te])
        assert any(v <= 12 for v in tv) or any(v >= 17 for v in tv)
