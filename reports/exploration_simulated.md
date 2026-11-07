# Exploration: online simulation (SIMULATED listener, CLAP view)

Start: 30 ratings drawn only from 2 of 7 taste modes. 10 rounds x 20 recommendations.
Means over 8 seeds (per-seed rows in the CSV).
*Full pool*: the 1,240-track sample, in which liked music is common (the sample oversamples taste sub-genres). *Rare*: every non-starting mode is cut to 12 tracks, the niche case this project is for.

| scenario       |   eps |   hit_rate |   explore_hit_rate |   modes_found |   new_modes_by_round3 |   rounds_to_new_mode |   catalogue_ndcg@50 |
|:---------------|------:|-----------:|-------------------:|--------------:|----------------------:|---------------------:|--------------------:|
| full pool      |  0.00 |       0.52 |             nan    |          6.50 |                  0.70 |                 3.45 |                0.43 |
| full pool      |  0.25 |       0.49 |               0.34 |          7.00 |                  0.88 |                 2.00 |                0.46 |
| full pool      |  0.50 |       0.45 |               0.33 |          7.00 |                  0.93 |                 1.78 |                0.42 |
| rare (12/mode) |  0.00 |       0.67 |             nan    |          5.75 |                  0.28 |                 6.72 |                0.39 |
| rare (12/mode) |  0.25 |       0.59 |               0.27 |          6.00 |                  0.43 |                 5.38 |                0.52 |
| rare (12/mode) |  0.50 |       0.48 |               0.27 |          6.50 |                  0.50 |                 4.88 |                0.53 |

* `hit_rate`: share of recommended tracks the listener likes. Exploration costs some of this in the short run.
* `modes_found`: taste modes (of 7) with at least one recommended like. Includes the 2 starting modes.
* `explore_hit_rate`: like rate of the exploration slots alone.
* `new_modes_by_round3`: share of the 5 non-starting modes liked within 3 batches.
* `rounds_to_new_mode`: mean batch index at which a non-starting mode is first liked (11 = not found within 10 batches). Lower = faster discovery.
* `catalogue_ndcg@50`: final model's ranking of every still-unrated track.
