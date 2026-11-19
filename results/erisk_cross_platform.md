# Cross-platform test: Twitter-trained models on eRisk (Reddit)

eRisk depression collection, users with >= 50 posts: 64 depressed, 622 control (324684 post-baseline posts). Positive = depressed user. No retraining; thresholds tuned on the Twitter validation split; 95% intervals resample users.

| Model trained for | Features | Twitter test user AUC | eRisk post AUC | eRisk user AUC | eRisk user macro F1 | User recall | Users flagged |
|---|---|---|---|---|---|---|---|
| depression | text only | 0.666 | 0.684 [0.658, 0.710] | 0.846 [0.806, 0.886] | 0.535 [0.494, 0.574] | 0.92 | 0.44 |
| depression | length, sentiment, language | 0.665 | 0.711 [0.683, 0.737] | 0.843 [0.805, 0.882] | 0.551 [0.510, 0.591] | 0.89 | 0.41 |
| mental_health | text only | 0.841 | 0.692 [0.662, 0.722] | 0.842 [0.795, 0.887] | 0.458 [0.421, 0.494] | 0.95 | 0.56 |
| mental_health | length, sentiment, language | 0.837 | 0.715 [0.685, 0.746] | 0.851 [0.813, 0.890] | 0.444 [0.409, 0.483] | 0.98 | 0.59 |
