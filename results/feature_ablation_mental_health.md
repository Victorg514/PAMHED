# Feature ablation: mental_health

Behavioural features only (no text), gradient-boosted trees, validation split. 0.5 = chance.

| feature set | n | val tweet ROC-AUC | val user ROC-AUC |
|---|---|---|---|
| all (incl. redundant) | 32 | 0.7523 | 0.7663 |
| default | 20 | 0.7152 | 0.7830 |
| length+sentiment+language | 10 | 0.6871 | 0.7987 |
| only length | 1 | 0.5352 | 0.5925 |
| only sentiment | 5 | 0.5864 | 0.6480 |
| only timing | 4 | 0.5066 | 0.4875 |
| only activity | 4 | 0.5895 | 0.5914 |
| only language | 4 | 0.6449 | 0.7989 |
| only indicators | 2 | 0.5047 | 0.6008 |
| single: len_z | 1 | 0.5352 | 0.5925 |
| single: sent_compound_z | 1 | 0.5213 | 0.5582 |
| single: neg | 1 | 0.5243 | 0.6247 |
| single: pos | 1 | 0.5070 | 0.5066 |
| single: emotion_volatility_10tw | 1 | 0.5660 | 0.5945 |
| single: polarity_shift | 1 | 0.5158 | 0.5735 |
| single: circadian_deviation | 1 | 0.5210 | 0.5393 |
| single: hour_sin | 1 | 0.5083 | 0.5083 |
| single: hour_cos | 1 | 0.5058 | 0.5017 |
| single: late_night | 1 | 0.5088 | 0.5475 |
| single: tpd_7d_dev | 1 | 0.5525 | 0.5534 |
| single: tpd_30d_dev | 1 | 0.5742 | 0.5582 |
| single: burst_posting | 1 | 0.5529 | 0.6146 |
| single: silence_period | 1 | 0.5196 | 0.6109 |
| single: self_focus | 1 | 0.6349 | 0.8051 |
| single: exclamation_rate | 1 | 0.5050 | 0.5437 |
| single: isolation_rate | 1 | 0.5002 | 0.5107 |
| single: mention_count | 1 | 0.5138 | 0.5591 |
| single: manic_indicator | 1 | 0.5001 | 0.4901 |
| single: depressive_indicator | 1 | 0.5048 | 0.6432 |
| default minus length | 19 | 0.7072 | 0.7763 |
| default minus sentiment | 15 | 0.6925 | 0.7668 |
| default minus timing | 16 | 0.7128 | 0.7818 |
| default minus activity | 16 | 0.6861 | 0.8002 |
| default minus language | 16 | 0.6691 | 0.7097 |
| default minus indicators | 18 | 0.7158 | 0.7863 |
