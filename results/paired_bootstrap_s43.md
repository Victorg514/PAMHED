# Paired bootstrap: text + features vs text only (seed 43)

1000 bootstrap samples of test users, both models scored on the same sample. Difference = features minus text only. p = two-sided bootstrap p-value. Covers test-set sampling noise, not training noise (seeds).

## Twitter test set

| Task | Metric | Text only | + features | Difference | 95% CI | p | Features better in |
|---|---|---|---|---|---|---|---|
| anxiety | tweet ROC-AUC | 0.651 | 0.662 | +0.011 | [-0.001, +0.023] | 0.076 | 96% |
| anxiety | user ROC-AUC | 0.703 | 0.712 | +0.009 | [-0.007, +0.025] | 0.324 | 84% |
| anxiety | user macro F1 | 0.598 | 0.602 | +0.004 | [-0.037, +0.043] | 0.830 | 58% |
