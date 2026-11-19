# SHAP: `mental_health__length+sentiment+language__s42`

Task mental_health, 10 features, test threshold 0.500. 40 test tweets sampled (at most one per user): FN 10, FP 10, TN 10, TP 10.

## 1. Behavioural features (text held fixed)

Beeswarm: `figures/shap_features_mental_health.png`. Mean |SHAP| is the average absolute change in P(positive) attributed to the feature. 'Direction' is the correlation between the scaled feature value and its SHAP value (+ = higher value pushes towards positive).

| Feature | Mean \|SHAP\| | Mean SHAP | Direction |
|---|---|---|---|
| emotion_volatility_10tw | 0.0267 | -0.0094 | +0.91 |
| len_z | 0.0110 | +0.0026 | -0.89 |
| self_focus | 0.0052 | +0.0002 | +0.89 |
| polarity_shift | 0.0046 | +0.0011 | +0.87 |
| sent_compound_z | 0.0023 | -0.0001 | -0.43 |
| pos | 0.0011 | -0.0003 | -0.21 |
| neg | 0.0010 | -0.0002 | -0.36 |
| mention_count | 0.0009 | -0.0001 | -0.90 |
| exclamation_rate | 0.0007 | -0.0007 | +0.42 |
| isolation_rate | 0.0001 | -0.0000 | +nan |

Total |SHAP| over all features per tweet: median 0.049, 90th percentile 0.097 (on a 0-1 probability scale).

## 2. Text (features held fixed)

Word-level explanations for 12 tweets: `shap_text_mental_health.html` (open in a browser).

## 3. Gate values (how much of the feature path is let through, 0 = ignored)

| Outcome | n | mean | std | min | median | max |
|---|---|---|---|---|---|---|
| FN | 10 | 0.603 | 0.010 | 0.585 | 0.605 | 0.620 |
| FP | 10 | 0.622 | 0.005 | 0.612 | 0.622 | 0.627 |
| TN | 10 | 0.598 | 0.017 | 0.558 | 0.601 | 0.614 |
| TP | 10 | 0.621 | 0.005 | 0.613 | 0.623 | 0.628 |

Overall mean gate 0.611 (std 0.015).

## Sampled tweets

| Outcome | p | Mean gate | Text |
|---|---|---|---|
| FN | 0.02 | 0.58 | We're kicking off Day 1 at   in San Diego! Stop by booth #2309 to meet our CEO, Rich Gliklich, and learn how OM1's robust, patented #AI technology and services  |
| FN | 0.06 | 0.61 | lando taking a photo with his whole fam😭 [IMG_CAP] a group of people standing around a man in a yellow shirt |
| FN | 0.07 | 0.59 | Amazon.com: D.V. eBook : Vreeland, Diana: Kindle Store On Sale Today for #Fashionistas [IMG_CAP]  |
| FN | 0.12 | 0.60 | J’ai fini Arcane un ruisseau sur mes joues [IMG_CAP]  |
| FN | 0.24 | 0.60 | It is time. 🙏 [IMG_CAP]  |
| FN | 0.34 | 0.60 |  [IMG_CAP]  |
| FN | 0.40 | 0.61 | good. [IMG_CAP]  |
| FN | 0.44 | 0.61 | the trees are obstructing the view but i think they add to it, personally [IMG_CAP] a sunset view from the back porch of a house |
| FN | 0.48 | 0.62 | me    +    my bias wrecker ❤️‍🔥 [IMG_CAP] a woman with long black hair and a red dress a woman with black hair and sunglasses on |
| FN | 0.48 | 0.61 | 😭😂 [IMG_CAP]  |
| FP | 0.56 | 0.61 | It’s so nice to know someone in Hollywood can still write a good article. . [IMG_CAP]  |
| FP | 0.63 | 0.62 | "Words won't change anything — let's silence them with our actions. The kings are nominated for many awards, so let's vote for them and show everyone what kind  |
| FP | 0.63 | 0.62 | Thanks for your support ✨  #ReleaseBYCLeaders #ReleaseMahrangBaloch [IMG_CAP]  |
| FP | 0.65 | 0.62 | 4. [IMG_CAP] a screenshot of a woman ' s profile |
| FP | 0.66 | 0.62 | she could sell me a 50 year warranty on a pencil [IMG_CAP]  |
| FP | 0.79 | 0.62 | #KPOPDEMONHUNTERS #jinumi  I would have never recovered if I saw this in the movie, however in my opinion the fact that Rumi cannot touch Jinu at the end of the |
| FP | 0.82 | 0.62 | I swear  hires smug flight attendants.   My flight is grounded and I asked for a ginger ale. He offered water and juice. I politely declined. He then tries to s |
| FP | 0.85 | 0.63 | It take me so long to get over stuff, I’m still mad bout shit from 3 years ago [IMG_CAP]  |
| FP | 0.95 | 0.63 | Apparently one of my Co-workers was bitching about me to my friend. I feel great :\| [IMG_CAP]  |
| FP | 0.98 | 0.63 | Park Minsu (player 125) is the real definition of looks like a cinnamon roll but could actually kill you #SquidGame   #SquidGameS3                #SquidGameSeas |
| TN | 0.01 | 0.56 | 🟡 LIVE at 1:30 PM ET — It’s Drop Lounge time!  We’re cracking open the From Gotham to Metropolis drop on our YouTube channel 🎥🦇☀️  Slabs, giveaways, chaos… you  |
| TN | 0.01 | 0.59 | Pink Venom rises to #230(+19) on stats fm with 4.6K streams   Let's also push Pink Venom 💪🏻 we've to reach 1B on it! [IMG_CAP] the app for the iphone shows the  |
| TN | 0.04 | 0.59 | "The eyes of the world see no further than this life, but the eyes of a Christian see deep into Eternity."  -St. John Vianney [IMG_CAP] the holy heart of jesus  |
| TN | 0.05 | 0.60 | All heroes stand with Palestine. [IMG_CAP] a cartoon superman and a child with the words go to superman |
| TN | 0.09 | 0.61 | "character": "[main character – in the images attached]",   "scene": {     "description": "The character stands facing forward in a bold, cinematic pose, wearin |
| TN | 0.13 | 0.60 | Top offensive performances from the ABC Open event by :   : 18 ABs, .500 AVG, 7 RBI, 11 Runs  : 16 ABs, .500 AVG, 5 Doubles  : .429 AVG, 1 Sac, .667 OBP [IMG_CA |
| TN | 0.22 | 0.61 | while we wait for vee formation on  who says a little hex hunt will hurt? i finished with peoples score on the leaderboard.😂 and that was because i got tired an |
| TN | 0.30 | 0.61 | Facebook has become that old friend who lost their way, yet you can't seem to give up on them [IMG_CAP] a cartoon character sitting on a bench with a capt that  |
| TN | 0.46 | 0.61 | due for one soon 🥺… [IMG_CAP]  |
| TN | 0.49 | 0.60 | KANGDANIEL 😍😍😍 [IMG_CAP]  |
| TP | 0.51 | 0.62 | Menyedihkan [IMG_CAP]  |
| TP | 0.52 | 0.61 | Because, ________ [IMG_CAP]  |
| TP | 0.63 | 0.62 | Yk what i fw cross universe tcest that shit so cute ong [IMG_CAP]  |
| TP | 0.64 | 0.62 | HAHAHAHA [IMG_CAP]  |
| TP | 0.77 | 0.62 | Me rn [IMG_CAP]  |
| TP | 0.77 | 0.62 | She has doxxed, harassed, wished dead, mocked trauma's, made homophobic and racist remarks, misandrist remarks, sexualized little children, talked bad about lit |
| TP | 0.77 | 0.62 | im sorry it’s basic but im literally in love with josh allen [IMG_CAP]  |
| TP | 0.79 | 0.62 | my gf found out abt my twt acc and made me delete it from my phone so i wont be active here much, however, i have the app still on my ipad so ill come here and  |
| TP | 0.88 | 0.63 | The way I’m sitting is hurting my hip but if it means he’s calm, I’ll suffer. [IMG_CAP] a black cat laying on a red blanket |
| TP | 0.89 | 0.63 | trust once school starts i WILL be meeting my step goal [IMG_CAP] the clock app on the samsung smart watch |
