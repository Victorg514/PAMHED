# Feature ablation: anxiety

Behavioural features only (no text), gradient-boosted trees, validation split. 0.5 = chance.

| feature set | n | val tweet ROC-AUC | val user ROC-AUC |
|---|---|---|---|
| all (incl. redundant) | 32 | 0.6449 | 0.6349 |
| default | 20 | 0.6098 | 0.6143 |
| length+sentiment+language | 10 | 0.5998 | 0.6092 |
| only length | 1 | 0.5187 | 0.5443 |
| only sentiment | 5 | 0.5514 | 0.5498 |
| only timing | 4 | 0.4797 | 0.4360 |
| only activity | 4 | 0.5250 | 0.5283 |
| only language | 4 | 0.5795 | 0.6431 |
| only indicators | 2 | 0.5049 | 0.5933 |
| single: len_z | 1 | 0.5187 | 0.5443 |
| single: sent_compound_z | 1 | 0.5342 | 0.5005 |
| single: neg | 1 | 0.5069 | 0.6059 |
| single: pos | 1 | 0.5096 | 0.5336 |
| single: emotion_volatility_10tw | 1 | 0.4893 | 0.4356 |
| single: polarity_shift | 1 | 0.5068 | 0.5137 |
| single: circadian_deviation | 1 | 0.5140 | 0.5142 |
| single: hour_sin | 1 | 0.5017 | 0.5108 |
| single: hour_cos | 1 | 0.5021 | 0.4980 |
| single: late_night | 1 | 0.5083 | 0.5578 |
| single: tpd_7d_dev | 1 | 0.5585 | 0.5659 |
| single: tpd_30d_dev | 1 | 0.5145 | 0.4942 |
| single: burst_posting | 1 | 0.5510 | 0.5930 |
| single: silence_period | 1 | 0.5304 | 0.5511 |
| single: self_focus | 1 | 0.5817 | 0.6682 |
| single: exclamation_rate | 1 | 0.4937 | 0.4982 |
| single: isolation_rate | 1 | 0.5010 | 0.5604 |
| single: mention_count | 1 | 0.5031 | 0.5808 |
| single: manic_indicator | 1 | 0.4986 | 0.4861 |
| single: depressive_indicator | 1 | 0.5065 | 0.6109 |
| default minus length | 19 | 0.6075 | 0.6248 |
| default minus sentiment | 15 | 0.5934 | 0.6108 |
| default minus timing | 16 | 0.6051 | 0.6185 |
| default minus activity | 16 | 0.5852 | 0.5966 |
| default minus language | 16 | 0.5712 | 0.5668 |
| default minus indicators | 18 | 0.6086 | 0.6240 |
