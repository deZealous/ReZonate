# Exploration: online simulation (SIMULATED listener, CLAP view)

Start: 30 ratings drawn only from 2 of 7 taste modes. 10 rounds x 20 recommendations.
Mean (std) over 5 seeds.

|   eps |   hit_rate (mean) |   hit_rate (std) |   modes_found (mean) |   modes_found (std) |   catalogue_ndcg@50 (mean) |   catalogue_ndcg@50 (std) |
|------:|------------------:|-----------------:|---------------------:|--------------------:|---------------------------:|--------------------------:|
| 0.000 |             0.525 |            0.030 |                6.600 |               0.548 |                      0.438 |                     0.105 |
| 0.250 |             0.495 |            0.055 |                7.000 |               0.000 |                      0.461 |                     0.060 |
| 0.500 |             0.434 |            0.029 |                7.000 |               0.000 |                      0.433 |                     0.065 |

* `hit_rate`: share of recommended tracks the listener likes. Exploration costs some of this in the short run.
* `modes_found`: taste modes (of 7) with at least one recommended like. Includes the 2 starting modes.
* `catalogue_ndcg@50`: final model's ranking of every still-unrated track.
