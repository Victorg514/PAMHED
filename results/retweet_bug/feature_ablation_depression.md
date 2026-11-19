# Feature ablation: depression

Behavioural features only (no text), gradient-boosted trees, validation split. 0.5 = chance.

| feature set | n | val tweet ROC-AUC | val user ROC-AUC |
|---|---|---|---|
| all (incl. redundant) | 32 | 0.5497 | 0.5396 |
| default | 20 | 0.5556 | 0.5543 |
| length+sentiment+language | 10 | 0.5820 | 0.5775 |
| only length | 1 | 0.4997 | 0.4784 |
| only sentiment | 5 | 0.5575 | 0.5521 |
| only timing | 4 | 0.5080 | 0.4775 |
| only activity | 4 | 0.5618 | 0.5572 |
| only language | 4 | 0.5562 | 0.6038 |
| only indicators | 2 | 0.5007 | 0.4642 |
| single: len_z | 1 | 0.4997 | 0.4784 |
| single: sent_compound_z | 1 | 0.5372 | 0.5118 |
| single: neg | 1 | 0.5022 | 0.4737 |
| single: pos | 1 | 0.5091 | 0.4973 |
| single: emotion_volatility_10tw | 1 | 0.5064 | 0.4735 |
| single: polarity_shift | 1 | 0.4978 | 0.5111 |
| single: circadian_deviation | 1 | 0.4776 | 0.4679 |
| single: hour_sin | 1 | 0.4979 | 0.4923 |
| single: hour_cos | 1 | 0.4979 | 0.4774 |
| single: late_night | 1 | 0.5167 | 0.5848 |
| single: tpd_7d_dev | 1 | 0.5216 | 0.5149 |
| single: tpd_30d_dev | 1 | 0.5122 | 0.5169 |
| single: burst_posting | 1 | 0.4936 | 0.4982 |
| single: silence_period | 1 | 0.4967 | 0.4833 |
| single: self_focus | 1 | 0.5556 | 0.6090 |
| single: exclamation_rate | 1 | 0.5092 | 0.5351 |
| single: isolation_rate | 1 | 0.4998 | 0.4922 |
| single: mention_count | 1 | 0.4949 | 0.4505 |
| single: manic_indicator | 1 | 0.5014 | 0.5042 |
| single: depressive_indicator | 1 | 0.4993 | 0.4520 |
| default minus length | 19 | 0.5563 | 0.5452 |
| default minus sentiment | 15 | 0.5507 | 0.5788 |
| default minus timing | 16 | 0.5644 | 0.5527 |
| default minus activity | 16 | 0.5686 | 0.5705 |
| default minus language | 16 | 0.5543 | 0.5391 |
| default minus indicators | 18 | 0.5551 | 0.5467 |
