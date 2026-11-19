# Results (v2 pipeline)

Test set. Thresholds tuned for macro F1 on validation. 95% intervals resample users.

## anxiety

| Features | Seed | Tweet ROC-AUC | Tweet macro F1 | User ROC-AUC | User macro F1 |
|---|---|---|---|---|---|
| length,sentiment,language | 42 | 0.599 [0.560, 0.638] | 0.544 [0.522, 0.568] | 0.674 [0.620, 0.721] | 0.603 [0.560, 0.645] |
| none | 42 | 0.593 [0.556, 0.634] | 0.535 [0.513, 0.562] | 0.643 [0.592, 0.693] | 0.555 [0.513, 0.597] |

## bipolar

| Features | Seed | Tweet ROC-AUC | Tweet macro F1 | User ROC-AUC | User macro F1 |
|---|---|---|---|---|---|
| length,sentiment,language | 42 | 0.591 [0.551, 0.631] | 0.546 [0.519, 0.571] | 0.637 [0.582, 0.688] | 0.597 [0.552, 0.636] |
| none | 42 | 0.595 [0.557, 0.634] | 0.544 [0.518, 0.567] | 0.631 [0.578, 0.682] | 0.595 [0.550, 0.638] |

## depression

| Features | Seed | Tweet ROC-AUC | Tweet macro F1 | User ROC-AUC | User macro F1 |
|---|---|---|---|---|---|
| length,sentiment,language | 42 | 0.558 [0.522, 0.595] | 0.523 [0.507, 0.540] | 0.599 [0.547, 0.649] | 0.522 [0.480, 0.561] |
| none | 42 | 0.559 [0.522, 0.595] | 0.525 [0.508, 0.541] | 0.605 [0.552, 0.654] | 0.541 [0.498, 0.579] |

## mental_health

| Features | Seed | Tweet ROC-AUC | Tweet macro F1 | User ROC-AUC | User macro F1 |
|---|---|---|---|---|---|
| length,sentiment,language | 42 | 0.673 [0.641, 0.707] | 0.620 [0.594, 0.646] | 0.760 [0.720, 0.799] | 0.689 [0.652, 0.727] |
| none | 42 | 0.666 [0.635, 0.699] | 0.613 [0.587, 0.638] | 0.756 [0.715, 0.793] | 0.682 [0.644, 0.719] |
