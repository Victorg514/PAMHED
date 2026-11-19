# Paired bootstrap: text + features vs text only (seed 42)

1000 bootstrap samples of test users, both models scored on the same sample. Difference = features minus text only. p = two-sided bootstrap p-value. Covers test-set sampling noise, not training noise (seeds).

## Twitter test set

| Task | Metric | Text only | + features | Difference | 95% CI | p | Features better in |
|---|---|---|---|---|---|---|---|
| mental_health | tweet ROC-AUC | 0.749 | 0.757 | +0.008 | [-0.001, +0.017] | 0.096 | 95% |
| mental_health | user ROC-AUC | 0.841 | 0.837 | -0.004 | [-0.018, +0.010] | 0.626 | 31% |
| mental_health | user macro F1 | 0.762 | 0.744 | -0.018 | [-0.039, +0.005] | 0.136 | 7% |
| depression | tweet ROC-AUC | 0.618 | 0.620 | +0.002 | [-0.008, +0.013] | 0.650 | 68% |
| depression | user ROC-AUC | 0.666 | 0.665 | -0.002 | [-0.020, +0.017] | 0.848 | 42% |
| depression | user macro F1 | 0.570 | 0.531 | -0.038 | [-0.076, -0.003] | 0.034 | 2% |
| anxiety | tweet ROC-AUC | 0.656 | 0.642 | -0.014 | [-0.026, -0.003] | 0.020 | 1% |
| anxiety | user ROC-AUC | 0.700 | 0.700 | +0.000 | [-0.018, +0.023] | 0.996 | 50% |
| anxiety | user macro F1 | 0.612 | 0.588 | -0.024 | [-0.068, +0.016] | 0.290 | 14% |
| bipolar | tweet ROC-AUC | 0.654 | 0.653 | -0.001 | [-0.014, +0.011] | 0.906 | 45% |
| bipolar | user ROC-AUC | 0.706 | 0.693 | -0.013 | [-0.030, +0.003] | 0.130 | 6% |
| bipolar | user macro F1 | 0.600 | 0.596 | -0.004 | [-0.030, +0.021] | 0.784 | 39% |

## eRisk Reddit (depressed vs control users)

| Task | Metric | Text only | + features | Difference | 95% CI | p | Features better in |
|---|---|---|---|---|---|---|---|
| mental_health | tweet ROC-AUC | 0.692 | 0.715 | +0.023 | [+0.016, +0.030] | 0.000 | 100% |
| mental_health | user ROC-AUC | 0.842 | 0.851 | +0.009 | [-0.011, +0.031] | 0.390 | 80% |
| mental_health | user macro F1 | 0.458 | 0.444 | -0.013 | [-0.029, +0.001] | 0.094 | 5% |
| depression | tweet ROC-AUC | 0.684 | 0.711 | +0.027 | [+0.018, +0.035] | 0.000 | 100% |
| depression | user ROC-AUC | 0.846 | 0.843 | -0.004 | [-0.021, +0.013] | 0.726 | 36% |
| depression | user macro F1 | 0.535 | 0.551 | +0.016 | [-0.004, +0.034] | 0.112 | 94% |
