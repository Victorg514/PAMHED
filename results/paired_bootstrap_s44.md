# Paired bootstrap: text + features vs text only (seed 44)

1000 bootstrap samples of test users, both models scored on the same sample. Difference = features minus text only. p = two-sided bootstrap p-value. Covers test-set sampling noise, not training noise (seeds).

## Twitter test set

| Task | Metric | Text only | + features | Difference | 95% CI | p | Features better in |
|---|---|---|---|---|---|---|---|
| anxiety | tweet ROC-AUC | 0.648 | 0.647 | -0.001 | [-0.009, +0.007] | 0.756 | 38% |
| anxiety | user ROC-AUC | 0.706 | 0.707 | +0.001 | [-0.010, +0.013] | 0.830 | 58% |
| anxiety | user macro F1 | 0.627 | 0.602 | -0.025 | [-0.048, -0.002] | 0.034 | 2% |
