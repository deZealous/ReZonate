"""Generate notebooks/01_explore.ipynb (kept as code so the notebook diff stays reviewable).

Run:  uv run python scripts/build_notebook.py
Then: uv run jupyter lab notebooks/01_explore.ipynb
"""

from pathlib import Path

import nbformat as nbf

CELLS = [
    ("md", """# ReZonate: exploration notebook

Use this to explore embeddings, check your taste modes, listen to a batch and rate it.
The fast rating loop is the CLI (`rezonate rate`); ratings made here are stored the same way
(`source='notebook'`).

**Demo mode:** with `DEMO = True` and no ratings of your own yet, the notebook uses the
*simulated* FMA listener from `rezonate.eval.simulate`. It never writes those labels to
your database."""),
    ("code", """import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import Audio, display, Markdown
from sklearn.decomposition import PCA
from rezonate import db, service
from rezonate.config import load_config, REPO_ROOT
from rezonate.embed.views import load_view

DEMO = True
cfg = load_config()
con = db.connect(cfg.db_path)
cat = service.load_catalog(cfg, con, "clap")
labels = service.current_labels(con)
labels = labels[labels.track_id.isin(cat.row)]
if DEMO and len(labels) < 20:
    from rezonate.ingest import fma
    from rezonate.eval.simulate import utility, grades_from_utility
    meta = fma.load_tracks(cfg.data_dir / "fma_meta").loc[[int(t.split(':')[1]) for t in cat.ids]]
    g = grades_from_utility(utility(meta, seed=0))
    rows = np.random.default_rng(0).choice(len(cat.ids), 150, replace=False)
    labels = pd.DataFrame({"track_id": [cat.ids[r] for r in rows], "grade": g.values[rows],
                           "weight": 1.0, "label_source": "simulated"})
    print("DEMO: using 150 simulated ratings (not saved)")
print(f"{len(cat.ids)} tracks with embeddings, {len(labels)} labels")"""),
    ("md", "## 1. Embedding maps (PCA, coloured by FMA top-level genre where known)"),
    ("code", """fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, view in zip(axes, ["clap", "mert:mean"]):
    try:
        ids, X = load_view(cfg.embeddings_dir, view, cat.ids)
    except Exception as e:
        ax.set_title(f"{view}: {e}"); continue
    Z = PCA(2, random_state=0).fit_transform(X)
    genre = cat.tracks.loc[ids, "genre_hint"].fillna("unknown")
    for j, gname in enumerate(genre.value_counts().index[:16]):
        m = (genre == gname).values
        ax.scatter(Z[m, 0], Z[m, 1], s=6, alpha=0.6, label=gname, color=plt.cm.tab20(j))
    ax.set_title(view); ax.set_xticks([]); ax.set_yticks([])
axes[0].legend(markerscale=3, fontsize=7, ncol=2)
plt.tight_layout()"""),
    ("md", "## 2. Embedding comparison (from `scripts/compare_embeddings.py`)"),
    ("code", """p = REPO_ROOT / "reports" / "embedding_comparison.md"
display(Markdown(p.read_text() if p.exists() else "Run scripts/compare_embeddings.py first."))"""),
    ("md", """## 3. Your taste modes
Liked tracks are clustered, and each mode is listed with the tracks closest to its centre.
If two modes look like the same thing to you, that's fine: the model doesn't need them to
be "genres"."""),
    ("code", """from rezonate.taste.knn import KNNTaste
rows = labels.track_id.map(cat.row).values
knn = KNNTaste().fit(cat.X[rows], labels.grade.values)
modes = knn.modes()
for i, c in enumerate(modes):
    near = np.argsort(-(cat.X @ c))[:5]
    t = cat.tracks.iloc[near][["artist", "title", "genre_hint"]]
    display(Markdown(f"**Mode {i + 1}**")); display(t)"""),
    ("md", "## 4. A recommendation batch, with audio"),
    ("code", """model = service.fit_bayes(cat, labels)
picks = service.recommend(cat, model, labels, n=10, eps=cfg.explore_fraction, seed=1)
batch = pd.DataFrame([{**cat.tracks.iloc[p.row][["artist", "title", "source", "license"]].to_dict(),
                       "context": p.context, "why": p.reason, "track_id": cat.ids[p.row]}
                      for p in picks])
display(batch)
for p in picks[:3]:
    r = cat.tracks.iloc[p.row]
    display(Markdown(f"{r.artist} — {r.title} *({p.context})*"))
    display(Audio(r.audio_path))"""),
    ("md", """## 5. Rate from the notebook
`rate(i, grade)` uses the batch index above and grades 0 = dislike, 1 = meh, 2 = like,
3 = love. In demo mode nothing is written."""),
    ("code", """def rate(i, grade):
    p = picks[i]
    if DEMO and (labels.label_source == "simulated").all():
        print("demo mode: not saved"); return
    db.add_feedback(con, [(cat.ids[p.row], grade, "notebook", p.context)])
    model.update(cat.X[p.row], grade)
    print("saved; P(like) now", float(model.score(cat.X[[p.row]])[0]))

# rate(0, 3)"""),
    ("md", "## 6. Offline evaluation reports"),
    ("code", """for name in ["eval_simulated.md", "exploration_simulated.md"]:
    p = REPO_ROOT / "reports" / name
    if p.exists():
        display(Markdown(p.read_text()))"""),
]


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(s) if k == "md" else nbf.v4.new_code_cell(s)
                for k, s in CELLS]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
    out = Path(__file__).resolve().parents[1] / "notebooks" / "01_explore.ipynb"
    nbf.write(nb, out)
    print(out)


if __name__ == "__main__":
    main()
