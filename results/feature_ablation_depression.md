# Feature ablation: depression

Behavioural features only (no text), gradient-boosted trees, validation split. 0.5 = chance.

| feature set | n | val tweet ROC-AUC | val user ROC-AUC |
|---|---|---|---|
| all (incl. redundant) | 32 | 0.5833 | 0.5806 |
| default | 20 | 0.5650 | 0.5874 |
| length+sentiment+language | 10 | 0.5819 | 0.6223 |
| only length | 1 | 0.5082 | 0.5608 |
| only sentiment | 5 | 0.5306 | 0.5676 |
| only timing | 4 | 0.5314 | 0.5612 |
| only activity | 4 | 0.5470 | 0.5746 |
| only language | 4 | 0.5861 | 0.6652 |
| only indicators | 2 | 0.5009 | 0.5187 |
| single: len_z | 1 | 0.5082 | 0.5608 |
| single: sent_compound_z | 1 | 0.4776 | 0.5007 |
| single: neg | 1 | 0.5091 | 0.5360 |
| single: pos | 1 | 0.5060 | 0.5596 |
| single: emotion_volatility_10tw | 1 | 0.5509 | 0.6134 |
| single: polarity_shift | 1 | 0.5107 | 0.5588 |
| single: circadian_deviation | 1 | 0.4979 | 0.4391 |
| single: hour_sin | 1 | 0.4982 | 0.4766 |
| single: hour_cos | 1 | 0.4911 | 0.4553 |
| single: late_night | 1 | 0.4922 | 0.4968 |
| single: tpd_7d_dev | 1 | 0.5252 | 0.5521 |
| single: tpd_30d_dev | 1 | 0.5091 | 0.5372 |
| single: burst_posting | 1 | 0.5335 | 0.5722 |
| single: silence_period | 1 | 0.5083 | 0.5583 |
| single: self_focus | 1 | 0.5820 | 0.6679 |
| single: exclamation_rate | 1 | 0.5143 | 0.5423 |
| single: isolation_rate | 1 | 0.5005 | 0.5502 |
| single: mention_count | 1 | 0.5056 | 0.4665 |
| single: manic_indicator | 1 | 0.4991 | 0.4784 |
| single: depressive_indicator | 1 | 0.5018 | 0.5212 |
| default minus length | 19 | 0.5721 | 0.5947 |
| default minus sentiment | 15 | 0.5793 | 0.6054 |
| default minus timing | 16 | 0.5610 | 0.5946 |
| default minus activity | 16 | 0.5731 | 0.6114 |
| default minus language | 16 | 0.5529 | 0.5873 |
| default minus indicators | 18 | 0.5711 | 0.5916 |
