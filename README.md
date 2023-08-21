# NBA DuoFit
## Link To WebApp: http://192.168.1.6:8511
Python WebApp to Predict NBA AllStar Duo Success

# Will 2 NBA Players Fit Together?
If there's one thing we've learnt over the years of watching the modern NBA with all its "superteams", it's that simply putting any all stars together is not guarenteed success. There's plenty examples of this: the 2021 Nets, 2022 Lakers, 2023 Mavs, 2017 Thunder among countless others. 

**WHY? IT'S BECAUSE NOT ALL PLAYERS _FIT_ TOGETHER**. \
Player fit is hard to quantify. Sure, there's the eye test. However, beyond a certain point, it's mostly speculative. How will Bradley Beal fit with the Suns? One could say he provides added scoring in plenty, but that scoring also requires the ball to be away from Devin Booker who was so potent this playoffs (KD has always been able to adjust to an offball role). Moreover, Beal doesn't add much defensively. How does that impact the Suns?

**Here's my proposed solution to the problem of predicting all star fit.**
### Solution  - Data Science Version:
First, I will use a an unsupervised clustering algorithm to examine the different types of all stars. I will compute the 20 closest players to each of the selected players using euclidean distance, and find the actual net rating (if available) for all combinations of those players and aggregate them.

### Solution - English Version: 
Think about any recommendation algorithm: Netflix, Spotify, TikTok. They all operate on the same assumption: if you like object X on their platform, you will also like the closest possible thing to X. Same principle here: find the actual net rating for combinations of players CLOSE to the chosen player and aggregate them

## Algorithm & Logic:
1. Condense NBA players into 3 dimensional spaces using Principal Component Analysis. Think of it as condensing their 30 statistical measures into 3 to be able to plot them in an xyz plane
2. Find the 20 most similar players to each selected player, and find the actual average net rating for any existng combinations of those players (using 2 man NBA lineup data)",
3. Check how this net rating compares to the average 2 man NBA all star net rating (as a ratio to the total mean)

### Data Credits: Basketball Reference and NBA.com
