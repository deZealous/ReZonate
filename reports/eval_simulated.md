# Offline evaluation on a SIMULATED listener (FMA)

> These numbers validate the pipeline. They are **not** evidence about your taste: the
> listener is simulated from FMA genre labels (see `src/rezonate/eval/simulate.py`).
> Re-run `rezonate evaluate` on your own ratings before trusting any conclusion.

* catalogue: 1240 FMA tracks; each persona rates 300 random tracks
* persona taste modes: post-punk / garage, indie / lo-fi / shoegaze, alt / synth pop, soul / r&b, reggae / dub, jazz-punk / free jazz, heavy alt rock
* protocol: 5-fold x 2 repeats x 3 personas = 30 folds per model; NDCG@10 on held-out rated tracks (gain 2^grade - 1); 95 % bootstrap CI
* `top10_pop_pct`: mean popularity percentile of each model's top 10 (lower = more niche picks)
* context coverage on this sample: tags 47%, era 100%, lyrics 1.7% (LRCLIB rarely knows FMA artists)

## Taste models vs baselines (audio only)

| model                 |   ndcg@10 | ndcg@10_ci     |   ndcg@5 |   p@10(like+) |   top10_pop_pct |
|:----------------------|----------:|:---------------|---------:|--------------:|----------------:|
| knn | clap+mert:L8    |     0.491 | [0.455, 0.528] |    0.489 |         0.603 |           0.473 |
| knn | clap            |     0.488 | [0.447, 0.532] |    0.478 |         0.617 |           0.483 |
| bayes | clap          |     0.481 | [0.443, 0.518] |    0.498 |         0.570 |           0.503 |
| ranker | clap         |     0.473 | [0.427, 0.518] |    0.456 |         0.587 |           0.511 |
| ranker | clap+mert:L8 |     0.466 | [0.430, 0.503] |    0.453 |         0.580 |           0.507 |
| bayes | clap+mert:L8  |     0.449 | [0.407, 0.491] |    0.455 |         0.547 |           0.484 |
| bayes | mert:L8       |     0.438 | [0.407, 0.468] |    0.446 |         0.517 |           0.468 |
| ranker | mert:L8      |     0.419 | [0.385, 0.450] |    0.420 |         0.507 |           0.476 |
| popularity            |     0.417 | [0.359, 0.470] |    0.442 |         0.450 |           0.920 |
| knn | mert:L8         |     0.406 | [0.367, 0.446] |    0.431 |         0.490 |           0.454 |
| random                |     0.265 | [0.226, 0.304] |    0.234 |         0.340 |           0.497 |

## Context-feature ablation (CLAP audio + context)

| model                         |   ndcg@10 | ndcg@10_ci     |   ndcg@5 |   p@10(like+) |   top10_pop_pct |
|:------------------------------|----------:|:---------------|---------:|--------------:|----------------:|
| ranker | clap+tags            |     0.554 | [0.513, 0.591] |    0.615 |         0.633 |           0.560 |
| bayes | clap+tags+era+lyrics  |     0.546 | [0.508, 0.580] |    0.581 |         0.630 |           0.553 |
| ranker | clap+tags+era+lyrics |     0.546 | [0.505, 0.585] |    0.591 |         0.633 |           0.543 |
| bayes | clap+tags             |     0.545 | [0.508, 0.578] |    0.591 |         0.613 |           0.561 |
| bayes | clap+pop              |     0.538 | [0.493, 0.578] |    0.574 |         0.617 |           0.673 |
| ranker | clap+pop             |     0.513 | [0.467, 0.558] |    0.528 |         0.610 |           0.603 |
| bayes | clap                  |     0.481 | [0.443, 0.518] |    0.498 |         0.570 |           0.503 |
| ranker | clap+era             |     0.475 | [0.432, 0.517] |    0.463 |         0.600 |           0.503 |
| bayes | clap+lyrics           |     0.475 | [0.435, 0.513] |    0.497 |         0.553 |           0.500 |
| ranker | clap                 |     0.473 | [0.427, 0.518] |    0.456 |         0.587 |           0.511 |
| ranker | clap+lyrics          |     0.473 | [0.424, 0.519] |    0.456 |         0.590 |           0.511 |
| bayes | clap+era              |     0.471 | [0.429, 0.511] |    0.484 |         0.573 |           0.505 |

## Paired differences (same folds; more sensitive than comparing the CIs above)

| comparison                                            |   mean Δ | 95% CI           | folds won   |
|:------------------------------------------------------|---------:|:-----------------|:------------|
| knn | clap  vs  popularity                            |   +0.072 | [-0.006, +0.154] | 18/30       |
| bayes | clap  vs  popularity                          |   +0.065 | [-0.008, +0.141] | 17/30       |
| ranker | clap  vs  popularity                         |   +0.057 | [-0.025, +0.140] | 18/30       |
| knn | clap  vs  knn | mert:L8                         |   +0.082 | [+0.045, +0.123] | 23/30       |
| bayes | clap  vs  knn | clap                          |   -0.007 | [-0.046, +0.035] | 15/30       |
| ranker | clap  vs  knn | clap                         |   -0.015 | [-0.045, +0.018] | 12/30       |
| ranker | clap+tags  vs  ranker | clap                 |   +0.081 | [+0.044, +0.118] | 25/30       |
| bayes | clap+tags  vs  bayes | clap                   |   +0.064 | [+0.025, +0.099] | 22/30       |
| ranker | clap+era  vs  ranker | clap                  |   +0.002 | [-0.014, +0.018] | 12/30       |
| ranker | clap+lyrics  vs  ranker | clap               |   -0.000 | [-0.010, +0.009] | 11/30       |
| bayes | clap+pop  vs  bayes | clap                    |   +0.057 | [+0.026, +0.089] | 23/30       |
| ranker | clap+tags+era+lyrics  vs  ranker | clap+tags |   -0.008 | [-0.025, +0.008] | 15/30       |
