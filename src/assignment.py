"""Uniqueness-constrained label assignment, ported from the CO2 GNN pipeline
(GNN_CO2_QN_assignment, src/train.py: evaluate_physical_assignment).

A per-level classifier happily gives two levels the same label. Within each block
of mutually exclusive labels (for water: same J and symmetry block) every label may
be used at most once, so we solve a linear assignment problem on cost 1 - p.

`assigned_margin` is the logit of the assigned label minus the best competing
logit, computed *after* the solver. It goes negative when the solver overrules the
classifier's top choice, so it penalises contested levels automatically. Threshold
on it, not on the raw argmax margin.

With factorised heads (e.g. vibrational label + Ka), build a joint logit matrix
first (sum of log-probabilities) so uniqueness applies to the full label.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import softmax


def assign(blocks, logits, locked=None, allowed=None):
    """
    blocks:  (n,) block key per level (str or int), e.g. f"{J}_{sym}".
    logits:  (n, C) classifier logits.
    locked:  (n,) known label per level (training / MARVEL-matched), -1 if unknown.
             Locked labels are removed from their block's pool before solving.
    allowed: optional callable(block_key) -> label ids valid in that block
             (e.g. only labels with Ka <= J). Default: all C labels.

    Returns (labels, assigned_margin). Label is -1 where a block ran out of labels.
    """
    logits = np.asarray(logits, dtype=float)
    n, n_classes = logits.shape
    probs = softmax(logits, axis=1)
    blocks = np.asarray(blocks)
    locked = np.full(n, -1) if locked is None else np.asarray(locked, dtype=int)
    labels = locked.copy()

    for key in np.unique(blocks):
        idx = np.flatnonzero(blocks == key)
        pool = np.arange(n_classes) if allowed is None else np.asarray(allowed(key))
        pool = np.setdiff1d(pool, locked[idx])
        free = idx[locked[idx] < 0]
        if len(free) == 0 or len(pool) == 0:
            continue
        # More levels than labels: scipy returns the optimal subset, rest stay -1.
        rows, cols = linear_sum_assignment(1.0 - probs[np.ix_(free, pool)])
        labels[free[rows]] = pool[cols]

    margin = np.zeros(n)
    done = np.flatnonzero(labels >= 0)
    competitors = logits[done].copy()
    competitors[np.arange(len(done)), labels[done]] = -np.inf
    margin[done] = logits[done, labels[done]] - competitors.max(axis=1)
    return labels, margin


if __name__ == "__main__":
    # Two levels in one block both prefer label 0. Level 1 wants it more, so level 0
    # is pushed to label 1 and gets a negative margin.
    labels, margin = assign(["a", "a"], [[3.0, 2.5], [2.0, 0.0]])
    assert labels.tolist() == [1, 0], labels
    assert np.allclose(margin, [-0.5, 2.0]), margin

    # A locked level takes label 0 out of the pool; the free level gets label 1.
    # A level in another block is unaffected.
    labels, _ = assign(["a", "a", "b"], [[0, 0], [5.0, 0.0], [5.0, 0.0]], locked=[0, -1, -1])
    assert labels.tolist() == [0, 1, 0], labels

    # More levels than allowed labels: one level stays unassigned.
    labels, _ = assign(["a", "a"], [[1.0, 0.0], [0.5, 0.0]], allowed=lambda k: [0])
    assert labels.tolist() == [0, -1], labels
    print("assignment self-check passed")
