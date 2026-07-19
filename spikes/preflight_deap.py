#!/usr/bin/env python3
"""Preflight for the DEAP affect spikes: validate DEAP is correctly
placed AND canonical before the (slow) extraction run.

Catches the two costly mistakes:
  1. Misplaced / wrong-format download (would waste the long extraction).
  2. The non-canonical Kaggle DEAP mirror -- valence mean ~3.47 vs the real SAM
     ~5.4 -- whose labels silently corrupt a 5.0-threshold binarisation. This is
     exactly the "doubly unreliable" trap documented in
     HONEST_AFFECT_EEG_NEGATIVE.md.

Fast: shape check on s01, label stats on 4 subjects. Run with the spike's venv.
"""
import os
import sys
import pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DDIR = os.path.join(HERE, "..", "..", "datasets", "deap", "data_preprocessed_python")


def fail(msg):
    print("\nPREFLIGHT FAIL:", msg)
    sys.exit(1)


if not os.path.isdir(DDIR):
    fail(f"DEAP not found at:\n  {os.path.abspath(DDIR)}\n\n"
         "Get the OFFICIAL EULA download (NOT the Kaggle mirror):\n"
         "  https://www.eecs.qmul.ac.uk/mmv/datasets/deap/\n"
         "Request access, sign the EULA, download data_preprocessed_python.zip,\n"
         "then unzip so the 32 s01.dat..s32.dat live in the path above.")

dats = sorted(f for f in os.listdir(DDIR) if f.endswith(".dat"))
print(f"found {len(dats)} .dat files in {os.path.abspath(DDIR)}")
if not dats:
    fail("no .dat files. If you see .mat files you grabbed data_preprocessed_MATLAB "
         "-- the trainer needs the PYTHON pickles (data_preprocessed_python.zip).")
if len(dats) != 32:
    print(f"  WARN: expected 32 subjects, found {len(dats)}")

# shape check on the first subject (trusted official DEAP pickle)
with open(os.path.join(DDIR, dats[0]), "rb") as fh:
    d = pickle.load(fh, encoding="latin1")
if "data" not in d or "labels" not in d:
    fail(f"{dats[0]} has no 'data'/'labels' keys -- not official preprocessed_python.")
data, labels = d["data"], d["labels"]
print(f"  {dats[0]}: data{data.shape} labels{labels.shape}")
if data.shape != (40, 40, 8064):
    fail(f"data shape {data.shape} != (40, 40, 8064) -- wrong or old format.")
if labels.shape != (40, 4):
    fail(f"labels shape {labels.shape} != (40, 4).")

# label canonicality over up to 4 subjects (decisive vs the Kaggle mirror)
vs = labels[:, 0].tolist()                       # reuse the already-loaded s01
for fn in dats[1:4]:
    with open(os.path.join(DDIR, fn), "rb") as fh:
        vs.extend(pickle.load(fh, encoding="latin1")["labels"][:, 0].tolist())
vmean = float(np.mean(vs))
print(f"  valence mean over {len(vs)} trials = {vmean:.2f}  (canonical DEAP SAM ~5.4)")
if not (4.3 <= vmean <= 6.5):
    fail(f"valence mean {vmean:.2f} is OUTSIDE the canonical band -- this looks like the\n"
         "non-canonical Kaggle mirror (mean ~3.47). Binarising at 5.0 would corrupt the\n"
         "verdict. Use the official EULA download.")

print("\nPREFLIGHT OK -- dataset placed and canonical. Safe to run:")
print(f"  {sys.executable} -u {os.path.join(HERE, 'train_deap_honest.py')}")
