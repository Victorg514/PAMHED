# Paired bootstrap: text + features vs text only (seed 42)

1000 bootstrap samples of test users, both models scored on the same sample. Difference = features minus text only. p = two-sided bootstrap p-value. Covers test-set sampling noise, not training noise (seeds).

## Twitter test set

| Task | Metric | Text only | + features | Difference | 95% CI | p | Features better in |
|---|---|---|---|---|---|---|---|
| mental_health | tweet ROC-AUC | 0.666 | 0.673 | +0.007 | [-0.002, +0.018] | 0.128 | 94% |
| mental_health | user ROC-AUC | 0.756 | 0.760 | +0.004 | [-0.008, +0.017] | 0.536 | 73% |
| mental_health | user macro F1 | 0.682 | 0.689 | +0.007 | [-0.013, +0.027] | 0.476 | 76% |
| depression | tweet ROC-AUC | 0.559 | 0.558 | -0.001 | [-0.011, +0.008] | 0.846 | 42% |
| depression | user ROC-AUC | 0.605 | 0.599 | -0.006 | [-0.026, +0.013] | 0.624 | 31% |
| depression | user macro F1 | 0.541 | 0.522 | -0.019 | [-0.058, +0.016] | 0.306 | 15% |
| anxiety | tweet ROC-AUC | 0.593 | 0.599 | +0.006 | [-0.007, +0.020] | 0.400 | 80% |
| anxiety | user ROC-AUC | 0.643 | 0.674 | +0.030 | [+0.008, +0.052] | 0.008 | 100% |
| anxiety | user macro F1 | 0.555 | 0.603 | +0.049 | [+0.017, +0.082] | 0.002 | 100% |
| bipolar | tweet ROC-AUC | 0.595 | 0.591 | -0.004 | [-0.012, +0.004] | 0.366 | 18% |
| bipolar | user ROC-AUC | 0.631 | 0.637 | +0.006 | [-0.006, +0.018] | 0.326 | 84% |
| bipolar | user macro F1 | 0.595 | 0.597 | +0.001 | [-0.016, +0.020] | 0.954 | 52% |

## eRisk Reddit (depressed vs control users)

| Task | Metric | Text only | + features | Difference | 95% CI | p | Features better in |
|---|---|---|---|---|---|---|---|
| mental_health | tweet ROC-AUC | 0.673 | 0.672 | -0.001 | [-0.006, +0.004] | 0.710 | 36% |
| mental_health | user ROC-AUC | 0.845 | 0.848 | +0.004 | [-0.011, +0.022] | 0.682 | 66% |
| mental_health | user macro F1 | 0.469 | 0.459 | -0.011 | [-0.026, +0.003] | 0.158 | 7% |
| depression | tweet ROC-AUC | 0.628 | 0.625 | -0.003 | [-0.010, +0.004] | 0.416 | 21% |
| depression | user ROC-AUC | 0.807 | 0.793 | -0.014 | [-0.041, +0.020] | 0.292 | 15% |
| depression | user macro F1 | 0.448 | 0.583 | +0.135 | [+0.097, +0.170] | 0.000 | 100% |
