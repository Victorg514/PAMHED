# Results (v2 pipeline)

Test set. Thresholds tuned for macro F1 on validation. 95% intervals resample users.

## 4class

| Features | Seed | Tweet macro F1 | User macro F1 |
|---|---|---|---|
| length,sentiment,language | 42 | 0.353 [0.331, 0.372] | 0.372 [0.335, 0.413] |
| none | 42 | 0.349 [0.330, 0.365] | 0.399 [0.355, 0.439] |

## anxiety

| Features | Seed | Tweet ROC-AUC | Tweet macro F1 | User ROC-AUC | User macro F1 |
|---|---|---|---|---|---|
| length,sentiment,language | 42 | 0.642 [0.596, 0.683] | 0.561 [0.535, 0.585] | 0.700 [0.639, 0.757] | 0.588 [0.538, 0.639] |
| length,sentiment,language | 43 | 0.662 [0.615, 0.705] | 0.575 [0.547, 0.601] | 0.712 [0.654, 0.767] | 0.602 [0.554, 0.651] |
| length,sentiment,language | 44 | 0.647 [0.601, 0.689] | 0.565 [0.539, 0.591] | 0.707 [0.650, 0.763] | 0.602 [0.552, 0.650] |
| none | 42 | 0.656 [0.608, 0.699] | 0.566 [0.539, 0.592] | 0.700 [0.641, 0.754] | 0.612 [0.561, 0.664] |
| none | 43 | 0.651 [0.603, 0.692] | 0.565 [0.539, 0.588] | 0.703 [0.641, 0.759] | 0.598 [0.551, 0.645] |
| none | 44 | 0.648 [0.603, 0.686] | 0.560 [0.536, 0.583] | 0.706 [0.647, 0.762] | 0.627 [0.575, 0.677] |

Mean ± std over seeds:

| Features | n | tweet_roc_auc | tweet_macro_f1 | user_roc_auc | user_macro_f1 |
|---|---|---|---|---|---|
| length,sentiment,language | 3 | 0.650 ± 0.011 | 0.567 ± 0.007 | 0.706 ± 0.006 | 0.598 ± 0.008 |
| none | 3 | 0.652 ± 0.004 | 0.564 ± 0.004 | 0.703 ± 0.003 | 0.613 ± 0.014 |

## bipolar

| Features | Seed | Tweet ROC-AUC | Tweet macro F1 | User ROC-AUC | User macro F1 |
|---|---|---|---|---|---|
| length,sentiment,language | 42 | 0.653 [0.615, 0.690] | 0.572 [0.545, 0.599] | 0.693 [0.637, 0.745] | 0.596 [0.549, 0.643] |
| none | 42 | 0.654 [0.616, 0.690] | 0.568 [0.544, 0.593] | 0.706 [0.651, 0.759] | 0.600 [0.552, 0.648] |

## depression

| Features | Seed | Tweet ROC-AUC | Tweet macro F1 | User ROC-AUC | User macro F1 |
|---|---|---|---|---|---|
| length,sentiment,language | 42 | 0.620 [0.577, 0.658] | 0.542 [0.522, 0.563] | 0.665 [0.609, 0.717] | 0.531 [0.487, 0.575] |
| none | 42 | 0.618 [0.578, 0.655] | 0.543 [0.522, 0.563] | 0.666 [0.614, 0.720] | 0.570 [0.521, 0.620] |

## mental_health

| Features | Seed | Tweet ROC-AUC | Tweet macro F1 | User ROC-AUC | User macro F1 |
|---|---|---|---|---|---|
| length,sentiment,language | 42 | 0.757 [0.726, 0.784] | 0.683 [0.657, 0.704] | 0.837 [0.800, 0.871] | 0.744 [0.705, 0.781] |
| none | 42 | 0.749 [0.720, 0.774] | 0.677 [0.654, 0.697] | 0.841 [0.805, 0.873] | 0.762 [0.722, 0.797] |
