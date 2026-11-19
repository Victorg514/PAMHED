# Cross-platform test: Twitter-trained models on eRisk (Reddit)

eRisk depression collection, users with >= 50 posts: 64 depressed, 622 control (324684 post-baseline posts). Positive = depressed user. No retraining; thresholds tuned on the Twitter validation split; 95% intervals resample users.

| Model trained for | Features | Twitter test user AUC | eRisk post AUC | eRisk user AUC | eRisk user macro F1 | User recall | Users flagged |
|---|---|---|---|---|---|---|---|
| depression | text only | 0.605 | 0.628 [0.606, 0.651] | 0.807 [0.753, 0.857] | 0.448 [0.413, 0.484] | 0.91 | 0.56 |
| depression | length, sentiment, language | 0.599 | 0.625 [0.602, 0.649] | 0.793 [0.744, 0.841] | 0.583 [0.541, 0.629] | 0.61 | 0.27 |
| mental_health | text only | 0.756 | 0.673 [0.643, 0.703] | 0.845 [0.800, 0.888] | 0.469 [0.429, 0.509] | 0.97 | 0.55 |
| mental_health | length, sentiment, language | 0.760 | 0.672 [0.640, 0.703] | 0.848 [0.805, 0.891] | 0.459 [0.420, 0.500] | 0.95 | 0.56 |
