import numpy as np

from rezonate.eval import protocol
from rezonate.explore import compose_batch
from rezonate.taste.knn import KNNTaste


def test_batch_reserves_exploration_share_and_picks_unfamiliar(two_mode_taste):
    X, c, g = two_mode_taste
    known = np.where(c == 0)[0][:60]          # listener has only heard cluster 0
    mask = np.ones(len(X), bool)
    mask[known] = False
    mean = KNNTaste().fit(X[known], g[known]).score(X)
    rng = np.random.default_rng(0)
    picks = compose_batch(X, mean, mean + rng.normal(0, 0.05, len(X)), known, mask,
                          n=20, eps=0.25, n_clusters=6)
    assert len(picks) == 20
    ex = [p for p in picks if p.context == "explore"]
    assert len(ex) == 5
    assert all(mask[p.row] for p in picks)            # never re-recommend known tracks
    assert len({p.row for p in picks}) == 20          # no duplicates
    # exploration slots land outside the familiar cluster
    assert np.mean([c[p.row] != 0 for p in ex]) >= 0.8


def test_protocol_runs_and_summarises(two_mode_taste):
    X, c, g = two_mode_taste
    lab = np.arange(0, len(X), 2)
    setup = protocol.Setup(ids=[str(i) for i in range(len(X))], views={"v": X}, C=None,
                           C_slices={}, popularity=np.random.default_rng(0).random(len(X)),
                           lab_rows=lab, grades=g[lab], weights=np.ones(len(lab)))
    scorers = {
        "random": lambda s, tr, te, rng: rng.random(len(te)),
        "knn": lambda s, tr, te, rng: KNNTaste().fit(X[lab[tr]], s.grades[tr]).score(X[lab[te]]),
    }
    df = protocol.run(setup, scorers, n_splits=3, n_repeats=1, log=None)
    summ = protocol.summarise(df)
    assert summ.iloc[0]["model"] == "knn"
    assert summ.set_index("model").loc["knn", "ndcg@10"] > summ.set_index("model").loc["random", "ndcg@10"]
