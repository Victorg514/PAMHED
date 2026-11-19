# Feature ablation: bipolar

Behavioural features only (no text), gradient-boosted trees, validation split. 0.5 = chance.

| feature set | n | val tweet ROC-AUC | val user ROC-AUC |
|---|---|---|---|
| all (incl. redundant) | 32 | 0.6635 | 0.6790 |
| default | 20 | 0.6342 | 0.6710 |
| length+sentiment+language | 10 | 0.6221 | 0.6943 |
| only length | 1 | 0.5191 | 0.5663 |
| only sentiment | 5 | 0.5649 | 0.5705 |
| only timing | 4 | 0.5295 | 0.5635 |
| only activity | 4 | 0.5580 | 0.5824 |
| only language | 4 | 0.6134 | 0.7368 |
| only indicators | 2 | 0.4996 | 0.5965 |
| single: len_z | 1 | 0.5191 | 0.5663 |
| single: sent_compound_z | 1 | 0.5333 | 0.5696 |
| single: neg | 1 | 0.5271 | 0.6192 |
| single: pos | 1 | 0.4928 | 0.4787 |
| single: emotion_volatility_10tw | 1 | 0.5659 | 0.5736 |
| single: polarity_shift | 1 | 0.5130 | 0.5604 |
| single: circadian_deviation | 1 | 0.4857 | 0.5125 |
| single: hour_sin | 1 | 0.5009 | 0.4988 |
| single: hour_cos | 1 | 0.5057 | 0.5140 |
| single: late_night | 1 | 0.5017 | 0.5263 |
| single: tpd_7d_dev | 1 | 0.5140 | 0.5535 |
| single: tpd_30d_dev | 1 | 0.5421 | 0.5176 |
| single: burst_posting | 1 | 0.5228 | 0.5471 |
| single: silence_period | 1 | 0.5034 | 0.5984 |
| single: self_focus | 1 | 0.6006 | 0.7275 |
| single: exclamation_rate | 1 | 0.5160 | 0.5273 |
| single: isolation_rate | 1 | 0.5008 | 0.5551 |
| single: mention_count | 1 | 0.5130 | 0.5530 |
| single: manic_indicator | 1 | 0.4978 | 0.4935 |
| single: depressive_indicator | 1 | 0.5017 | 0.5966 |
| default minus length | 19 | 0.6394 | 0.6756 |
| default minus sentiment | 15 | 0.6059 | 0.6502 |
| default minus timing | 16 | 0.6419 | 0.6768 |
| default minus activity | 16 | 0.6260 | 0.7009 |
| default minus language | 16 | 0.5975 | 0.6099 |
| default minus indicators | 18 | 0.6331 | 0.6649 |
