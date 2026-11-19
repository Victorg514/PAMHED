# Feature ablation: mental_health

Behavioural features only (no text), gradient-boosted trees, validation split. 0.5 = chance.

| feature set | n | val tweet ROC-AUC | val user ROC-AUC |
|---|---|---|---|
| all (incl. redundant) | 32 | 0.6636 | 0.7054 |
| default | 20 | 0.6365 | 0.7062 |
| length+sentiment+language | 10 | 0.6454 | 0.7269 |
| only length | 1 | 0.5190 | 0.5514 |
| only sentiment | 5 | 0.5625 | 0.6235 |
| only timing | 4 | 0.5053 | 0.5389 |
| only activity | 4 | 0.5418 | 0.5498 |
| only language | 4 | 0.6230 | 0.7100 |
| only indicators | 2 | 0.5008 | 0.5531 |
| single: len_z | 1 | 0.5190 | 0.5514 |
| single: sent_compound_z | 1 | 0.5175 | 0.5719 |
| single: neg | 1 | 0.5204 | 0.5692 |
| single: pos | 1 | 0.5063 | 0.5253 |
| single: emotion_volatility_10tw | 1 | 0.5458 | 0.5807 |
| single: polarity_shift | 1 | 0.5073 | 0.5411 |
| single: circadian_deviation | 1 | 0.5079 | 0.5111 |
| single: hour_sin | 1 | 0.5162 | 0.5492 |
| single: hour_cos | 1 | 0.5151 | 0.5370 |
| single: late_night | 1 | 0.4963 | 0.5186 |
| single: tpd_7d_dev | 1 | 0.5064 | 0.5027 |
| single: tpd_30d_dev | 1 | 0.5166 | 0.4968 |
| single: burst_posting | 1 | 0.5047 | 0.5375 |
| single: silence_period | 1 | 0.5063 | 0.5459 |
| single: self_focus | 1 | 0.6157 | 0.7116 |
| single: exclamation_rate | 1 | 0.5101 | 0.5312 |
| single: isolation_rate | 1 | 0.4998 | 0.4987 |
| single: mention_count | 1 | 0.5093 | 0.5653 |
| single: manic_indicator | 1 | 0.4988 | 0.4960 |
| single: depressive_indicator | 1 | 0.5021 | 0.5680 |
| default minus length | 19 | 0.6273 | 0.7045 |
| default minus sentiment | 15 | 0.6188 | 0.6863 |
| default minus timing | 16 | 0.6393 | 0.7086 |
| default minus activity | 16 | 0.6372 | 0.7138 |
| default minus language | 16 | 0.5855 | 0.6385 |
| default minus indicators | 18 | 0.6352 | 0.7040 |
