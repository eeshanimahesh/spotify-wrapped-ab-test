# 🎵 Spotify Wrapped A/B Test: Network Interference & Cluster Randomization

> **The problem most A/B tests ignore:** When your feature goes viral, your control group gets contaminated — and your causal estimate is wrong.

---

## 📌 Business Question

Does showing users their personalized **Spotify Wrapped** year-in-review increase 30-day retention compared to users who don't see it?

**Why this is harder than it sounds:** Wrapped is designed to be shared. When treated users post their Wrapped on Instagram and Twitter, their friends in the control group see it too — violating a core assumption of standard A/B testing.

---

## 🧠 The Core Problem: SUTVA Violation

Standard A/B testing relies on **SUTVA** — the Stable Unit Treatment Value Assumption:

> *One user's treatment should not affect another user's outcome.*

Spotify Wrapped **breaks this assumption** because:
- Treated users share their Wrapped on social media
- Control users who see their friends' Wrapped get indirectly influenced
- This **inflates control group retention**, making the treatment look less effective than it really is

---

## 🔬 What This Project Covers

| Section | Concept |
|---|---|
| 1 | Build a realistic social network (5,000 users, 250 clusters) |
| 2 | Naive user-level randomization — demonstrate bias |
| 3 | Cluster randomization — eliminate interference |
| 4 | Quantify spillover effects by exposure level |
| 5 | Network spillover model (direct + indirect effect decomposition) |
| 6 | Visualizations: network graph, bias comparison, cluster design |
| 7 | Decision framework + business impact estimate |

---

## 📊 Key Results

```
                    ESTIMATE    BIAS        RELIABLE?
Naive (user-level)  8.02%       Contaminated   ❌ NO
Cluster             8.83%       Isolated       ✅ YES
Ground truth        8.00%       —              ✅ YES

Spillover decomposition:
  Direct effect:    8.09%   (Wrapped itself)
  Indirect effect:  8.83%   (social sharing virality)
  Total effect:     16.92%

Business impact @ 50M MAU: $1.44B incremental annual revenue
```

**Decision: 🚀 SHIP — and the free virality is worth as much as the feature itself.**

---

## 💡 Key Insight

The naive analysis missed that **52% of the total value comes from spillover** — users sharing their Wrapped with friends. This means:

1. Standard A/B testing *underestimates* the true value of Wrapped
2. Cluster randomization isolates this effect correctly
3. The social virality is itself a business asset worth quantifying

---

## 🗂️ Repository Structure

```
spotify_wrapped/
├── spotify_wrapped_analysis.py      # Full analysis (7 sections)
├── 01_network_interference.png      # Network viz + bias comparison + spillover analysis
├── 02_cluster_design.png            # Cluster vs user-level randomization design
└── README.md
```

---

## 🚀 How to Run

```bash
pip install numpy pandas scipy matplotlib seaborn statsmodels networkx
python spotify_wrapped_analysis.py
```

---

## 🛠️ Skills Demonstrated

`Network Interference` · `SUTVA Violation` · `Cluster Randomization` · `Spillover Effect Quantification` · `Social Network Analysis` · `Linear Probability Model` · `Causal Inference` · `A/B Experiment Design` · `Python` · `NetworkX`

---

## 📈 Visualizations

![Network Interference Analysis](01_network_interference.png)
![Cluster Randomization Design](02_cluster_design.png)

---

*Part of a portfolio of applied experimentation projects for product data science roles.*
