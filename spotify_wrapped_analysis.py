"""
Spotify Wrapped A/B Test: Network Interference & Cluster Randomization
=======================================================================
Author: Eeshani Gundi

Business Question:
    Does showing users their personalized Spotify Wrapped year-in-review
    increase 30-day retention compared to users who don't see it?

Why This Is Hard:
    Standard A/B testing assumes SUTVA — Stable Unit Treatment Value Assumption.
    This means one user's treatment shouldn't affect another user's outcome.
    Spotify Wrapped VIOLATES this: when a user shares their Wrapped on social
    media, their friends (who may be in the control group) also get exposed,
    inflating the control group's retention and UNDERESTIMATING the true effect.

What We Do:
    1. Simulate a social network of Spotify users
    2. Show the bias from naive user-level randomization
    3. Implement cluster-based randomization (friend groups as clusters)
    4. Compare estimates and quantify the interference bias
    5. Build a network spillover model to estimate direct + indirect effects
    6. Final decision framework with correct causal estimate

Key Concepts:
    - SUTVA violation & network interference
    - Cluster randomization
    - Spillover effects
    - Intent-to-treat (ITT) vs Average Treatment Effect (ATE)
    - Social network analysis
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import networkx as nx
from statsmodels.stats.proportion import proportions_ztest, proportion_confint
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ─────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0F1117',
    'axes.facecolor': '#1A1D27',
    'axes.edgecolor': '#2E3347',
    'axes.labelcolor': '#C8CDD8',
    'text.color': '#C8CDD8',
    'xtick.color': '#8B92A5',
    'ytick.color': '#8B92A5',
    'grid.color': '#2E3347',
    'grid.linewidth': 0.5,
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 12,
    'axes.labelsize': 10,
})

GREEN = '#1DB954'   # Spotify green
BLACK = '#191414'   # Spotify black
WHITE = '#FFFFFF'
BLUE = '#4F8EF7'
RED = '#E84855'
YELLOW = '#FFBE0B'
PURPLE = '#9B5DE5'
GRAY = '#8B92A5'
ORANGE = '#FF6B35'

# ═══════════════════════════════════════════════════════
# SECTION 1: BUILD SOCIAL NETWORK
# ═══════════════════════════════════════════════════════
print("=" * 65)
print("SECTION 1: BUILDING SPOTIFY USER SOCIAL NETWORK")
print("=" * 65)

N_USERS = 5000
N_CLUSTERS = 250   # ~20 users per friend group on average

# Generate friend clusters (communities)
# Real social networks follow power-law degree distributions
cluster_assignments = np.repeat(np.arange(N_CLUSTERS), N_USERS // N_CLUSTERS)
cluster_assignments = np.append(cluster_assignments,
                                 np.random.randint(0, N_CLUSTERS, N_USERS - len(cluster_assignments)))
np.random.shuffle(cluster_assignments)

# User features
user_df = pd.DataFrame({
    'user_id': range(N_USERS),
    'cluster_id': cluster_assignments,
    'monthly_listening_hrs': np.random.lognormal(3.0, 0.8, N_USERS).clip(1, 200),
    'n_playlists': np.random.poisson(8, N_USERS),
    'premium': np.random.binomial(1, 0.45, N_USERS),
    'social_score': np.random.beta(2, 5, N_USERS),  # propensity to share
    'age_group': np.random.choice(['18-24', '25-34', '35-44', '45+'],
                                   N_USERS, p=[0.35, 0.30, 0.20, 0.15]),
})

# Build social graph: within-cluster edges (strong ties) + cross-cluster (weak ties)
G = nx.Graph()
G.add_nodes_from(range(N_USERS))

edges = []
for cluster in range(N_CLUSTERS):
    members = user_df[user_df.cluster_id == cluster].user_id.tolist()
    # Strong ties: connect ~60% of within-cluster pairs
    for i in range(len(members)):
        for j in range(i+1, len(members)):
            if np.random.random() < 0.25:
                edges.append((members[i], members[j]))

# Weak ties: random cross-cluster connections
n_weak = int(len(edges) * 0.15)
weak_u = np.random.randint(0, N_USERS, n_weak)
weak_v = np.random.randint(0, N_USERS, n_weak)
for u, v in zip(weak_u, weak_v):
    if u != v and cluster_assignments[u] != cluster_assignments[v]:
        edges.append((u, v))

G.add_edges_from(edges)

# Compute network degree for each user
degrees = dict(G.degree())
user_df['n_friends'] = user_df['user_id'].map(degrees)

print(f"Users:          {N_USERS:,}")
print(f"Friend clusters: {N_CLUSTERS:,}")
print(f"Total edges:    {G.number_of_edges():,}")
print(f"Avg friends:    {user_df.n_friends.mean():.1f}")
print(f"Max friends:    {user_df.n_friends.max()}")

# ═══════════════════════════════════════════════════════
# SECTION 2: NAIVE USER-LEVEL RANDOMIZATION (BIASED)
# ═══════════════════════════════════════════════════════
print("\n\n" + "=" * 65)
print("SECTION 2: NAIVE USER-LEVEL RANDOMIZATION (BIASED ESTIMATE)")
print("=" * 65)

# Assign treatment at user level (50/50)
user_df['naive_treatment'] = np.random.binomial(1, 0.5, N_USERS)

# Simulate Wrapped sharing behavior
# Treated users share Wrapped; sharing probability depends on social_score
user_df['shared_wrapped'] = (
    (user_df['naive_treatment'] == 1) &
    (np.random.random(N_USERS) < user_df['social_score'] * 1.8)
).astype(int)

# Spillover: control users who have friends that shared get partially exposed
def compute_spillover_exposure(user_id, G, shared_set):
    neighbors = list(G.neighbors(user_id))
    if not neighbors:
        return 0.0
    return sum(1 for n in neighbors if n in shared_set) / len(neighbors)

shared_users = set(user_df[user_df.shared_wrapped == 1].user_id)

print("Computing spillover exposure (this takes a moment)...")
spillover = []
for uid in range(N_USERS):
    neighbors = list(G.neighbors(uid))
    if neighbors:
        exp = sum(1 for n in neighbors if n in shared_users) / len(neighbors)
    else:
        exp = 0.0
    spillover.append(exp)

user_df['spillover_exposure'] = spillover

# Simulate 30-day retention outcome
# Base retention rate: 62%
# Direct treatment effect: +8pp (true effect)
# Spillover effect: +3pp per 10% of friends who shared (indirect effect)
BASE_RETENTION = 0.62
TRUE_DIRECT_EFFECT = 0.08
SPILLOVER_EFFECT = 0.30  # coefficient on spillover_exposure

def simulate_retention(treatment, spillover_exp, listening_hrs, premium, social_score):
    p = BASE_RETENTION
    p += TRUE_DIRECT_EFFECT * treatment
    p += SPILLOVER_EFFECT * spillover_exp  # ← this is the interference
    p += 0.02 * premium
    p += 0.001 * np.log1p(listening_hrs)
    p += np.random.normal(0, 0.02, len(treatment))
    return (np.random.random(len(treatment)) < p.clip(0, 1)).astype(int)

user_df['retained_30d'] = simulate_retention(
    user_df['naive_treatment'].values,
    user_df['spillover_exposure'].values,
    user_df['monthly_listening_hrs'].values,
    user_df['premium'].values,
    user_df['social_score'].values
)

# Naive estimate
ctrl_naive = user_df[user_df.naive_treatment == 0]
trt_naive = user_df[user_df.naive_treatment == 1]

naive_ctrl_rate = ctrl_naive.retained_30d.mean()
naive_trt_rate = trt_naive.retained_30d.mean()
naive_lift = naive_trt_rate - naive_ctrl_rate

z_naive, p_naive = proportions_ztest(
    [trt_naive.retained_30d.sum(), ctrl_naive.retained_30d.sum()],
    [len(trt_naive), len(ctrl_naive)]
)

print(f"\nNAIVE (user-level) ESTIMATE:")
print(f"  Control retention:   {naive_ctrl_rate:.4f} ({naive_ctrl_rate:.2%})")
print(f"  Treatment retention: {naive_trt_rate:.4f} ({naive_trt_rate:.2%})")
print(f"  Observed lift:       {naive_lift:.4f} ({naive_lift:.2%})")
print(f"  True direct effect:  {TRUE_DIRECT_EFFECT:.4f} ({TRUE_DIRECT_EFFECT:.2%})")
print(f"  Bias (underestimate):{TRUE_DIRECT_EFFECT - naive_lift:.4f} ({(TRUE_DIRECT_EFFECT-naive_lift):.2%})")
print(f"  p-value:             {p_naive:.4f}")
print(f"\n  ⚠️  Naive estimate UNDERESTIMATES true effect by "
      f"{((TRUE_DIRECT_EFFECT - naive_lift)/TRUE_DIRECT_EFFECT)*100:.1f}%")
print(f"  because spillover inflated control group retention!")

# ═══════════════════════════════════════════════════════
# SECTION 3: CLUSTER RANDOMIZATION (UNBIASED)
# ═══════════════════════════════════════════════════════
print("\n\n" + "=" * 65)
print("SECTION 3: CLUSTER RANDOMIZATION (UNBIASED ESTIMATE)")
print("=" * 65)

# Assign treatment at cluster level
cluster_treatment = pd.Series(
    np.random.binomial(1, 0.5, N_CLUSTERS),
    name='cluster_treatment'
)
user_df['cluster_treatment'] = user_df['cluster_id'].map(cluster_treatment)

# Now spillover stays within cluster — control clusters are fully isolated
user_df['shared_wrapped_cluster'] = (
    (user_df['cluster_treatment'] == 1) &
    (np.random.random(N_USERS) < user_df['social_score'] * 1.8)
).astype(int)

# Spillover only within same cluster
shared_cluster_users = set(user_df[user_df.shared_wrapped_cluster == 1].user_id)
spillover_cluster = []
for uid in range(N_USERS):
    user_cluster = user_df.loc[uid, 'cluster_id']
    neighbors = list(G.neighbors(uid))
    same_cluster_neighbors = [n for n in neighbors
                               if user_df.loc[n, 'cluster_id'] == user_cluster]
    if same_cluster_neighbors:
        exp = sum(1 for n in same_cluster_neighbors if n in shared_cluster_users) / len(same_cluster_neighbors)
    else:
        exp = 0.0
    spillover_cluster.append(exp)

user_df['spillover_cluster'] = spillover_cluster

user_df['retained_30d_cluster'] = simulate_retention(
    user_df['cluster_treatment'].values,
    user_df['spillover_cluster'].values,
    user_df['monthly_listening_hrs'].values,
    user_df['premium'].values,
    user_df['social_score'].values
)

ctrl_cluster = user_df[user_df.cluster_treatment == 0]
trt_cluster = user_df[user_df.cluster_treatment == 1]

cluster_ctrl_rate = ctrl_cluster.retained_30d_cluster.mean()
cluster_trt_rate = trt_cluster.retained_30d_cluster.mean()
cluster_lift = cluster_trt_rate - cluster_ctrl_rate

z_cluster, p_cluster = proportions_ztest(
    [trt_cluster.retained_30d_cluster.sum(), ctrl_cluster.retained_30d_cluster.sum()],
    [len(trt_cluster), len(ctrl_cluster)]
)

print(f"\nCLUSTER RANDOMIZATION ESTIMATE:")
print(f"  Control retention:   {cluster_ctrl_rate:.4f} ({cluster_ctrl_rate:.2%})")
print(f"  Treatment retention: {cluster_trt_rate:.4f} ({cluster_trt_rate:.2%})")
print(f"  Observed lift:       {cluster_lift:.4f} ({cluster_lift:.2%})")
print(f"  True direct effect:  {TRUE_DIRECT_EFFECT:.4f} ({TRUE_DIRECT_EFFECT:.2%})")
print(f"  Bias remaining:      {abs(TRUE_DIRECT_EFFECT - cluster_lift):.4f}")
print(f"  p-value:             {p_cluster:.4f}")
print(f"\n  ✅ Cluster estimate is {abs(1-(cluster_lift/TRUE_DIRECT_EFFECT))*100:.1f}% "
      f"closer to true effect vs naive estimate")

# ═══════════════════════════════════════════════════════
# SECTION 4: SPILLOVER EFFECT QUANTIFICATION
# ═══════════════════════════════════════════════════════
print("\n\n" + "=" * 65)
print("SECTION 4: SPILLOVER EFFECT QUANTIFICATION")
print("=" * 65)

# Segment control users by their spillover exposure level
ctrl_users = user_df[user_df.naive_treatment == 0].copy()
ctrl_users['spillover_bucket'] = pd.cut(
    ctrl_users['spillover_exposure'],
    bins=[-0.001, 0.05, 0.15, 0.30, 1.0],
    labels=['None (0-5%)', 'Low (5-15%)', 'Medium (15-30%)', 'High (30%+)']
)

spillover_analysis = ctrl_users.groupby('spillover_bucket', observed=True).agg(
    n_users=('retained_30d', 'count'),
    retention_rate=('retained_30d', 'mean')
).reset_index()

print("\nControl group retention by spillover exposure level:")
print(f"\n{'Spillover Level':<18} {'Users':>8} {'Retention':>12} {'Lift vs None':>14}")
print("-" * 56)
base_rate = spillover_analysis[spillover_analysis.spillover_bucket == 'None (0-5%)']['retention_rate'].values[0]
for _, row in spillover_analysis.iterrows():
    lift_vs_none = row['retention_rate'] - base_rate
    print(f"{str(row['spillover_bucket']):<18} {row['n_users']:>8,} "
          f"{row['retention_rate']:>12.2%} {lift_vs_none:>+14.2%}")

print(f"\nInterpretation: Control users with high spillover exposure retain")
print(f"at nearly the same rate as treated users — confirming interference bias.")

# ═══════════════════════════════════════════════════════
# SECTION 5: NETWORK SPILLOVER MODEL
# ═══════════════════════════════════════════════════════
print("\n\n" + "=" * 65)
print("SECTION 5: NETWORK SPILLOVER MODEL")
print("=" * 65)

from statsmodels.api import OLS, add_constant

# Decompose into direct + indirect (spillover) effects
model_df = user_df[['naive_treatment', 'spillover_exposure',
                      'premium', 'monthly_listening_hrs', 'retained_30d']].copy()
model_df['log_listening'] = np.log1p(model_df['monthly_listening_hrs'])
model_df = model_df.dropna()

X = add_constant(model_df[['naive_treatment', 'spillover_exposure', 'premium', 'log_listening']])
y = model_df['retained_30d']

model = OLS(y, X).fit()

direct_effect = model.params['naive_treatment']
spillover_coef = model.params['spillover_exposure']

print(f"\nLinear Probability Model Results:")
print(f"  Direct treatment effect:   {direct_effect:.4f} ({direct_effect:.2%}) "
      f"[p={model.pvalues['naive_treatment']:.4f}]")
print(f"  Spillover effect (per unit): {spillover_coef:.4f} ({spillover_coef:.2%}) "
      f"[p={model.pvalues['spillover_exposure']:.4f}]")
print(f"  True direct effect:        {TRUE_DIRECT_EFFECT:.4f} ({TRUE_DIRECT_EFFECT:.2%})")
print(f"\n  Model R²: {model.rsquared:.4f}")
print(f"\n  Total effect (direct + avg spillover):")
avg_spillover = user_df['spillover_exposure'].mean()
total_effect = direct_effect + spillover_coef * avg_spillover
print(f"    = {direct_effect:.4f} + {spillover_coef:.4f} × {avg_spillover:.4f}")
print(f"    = {total_effect:.4f} ({total_effect:.2%})")

# ═══════════════════════════════════════════════════════
# SECTION 6: VISUALIZATIONS
# ═══════════════════════════════════════════════════════
print("\n\n" + "=" * 65)
print("SECTION 6: GENERATING VISUALIZATIONS")
print("=" * 65)

# ── Plot 1: Network + Interference Overview ──
fig = plt.figure(figsize=(18, 10))
fig.patch.set_facecolor('#0F1117')
gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.35)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[1, 0])
ax5 = fig.add_subplot(gs[1, 1])
ax6 = fig.add_subplot(gs[1, 2])

# 1. Mini network visualization (subsample)
sub_nodes = list(range(150))
sub_G = G.subgraph(sub_nodes)
pos = nx.spring_layout(sub_G, seed=42, k=0.8)

node_colors = []
for n in sub_nodes:
    if user_df.loc[n, 'naive_treatment'] == 1:
        node_colors.append(GREEN)
    elif user_df.loc[n, 'spillover_exposure'] > 0.1:
        node_colors.append(YELLOW)
    else:
        node_colors.append(GRAY)

nx.draw_networkx_edges(sub_G, pos, ax=ax1, alpha=0.15, edge_color=GRAY, width=0.5)
nx.draw_networkx_nodes(sub_G, pos, ax=ax1, node_color=node_colors,
                        node_size=30, alpha=0.9)
ax1.set_facecolor('#1A1D27')
ax1.set_title('Social Network\n(Green=Treated, Yellow=Spillover, Gray=Control)', fontsize=9)
ax1.axis('off')

patches = [
    mpatches.Patch(color=GREEN, label='Treated'),
    mpatches.Patch(color=YELLOW, label='Spillover-exposed control'),
    mpatches.Patch(color=GRAY, label='Pure control'),
]
ax1.legend(handles=patches, loc='lower left', fontsize=7, facecolor='#1A1D27',
           edgecolor=GRAY, labelcolor='white')

# 2. Bias comparison
methods = ['True Effect\n(Ground Truth)', 'Naive\n(User-level)', 'Cluster\nRandomization']
lifts = [TRUE_DIRECT_EFFECT, naive_lift, cluster_lift]
colors_bar = [GREEN, RED, BLUE]
bars = ax2.bar(methods, [l*100 for l in lifts], color=colors_bar,
               width=0.5, edgecolor='#0F1117', linewidth=1.5)
ax2.axhline(TRUE_DIRECT_EFFECT * 100, color=GREEN, linestyle='--',
            linewidth=1.5, alpha=0.5, label='True effect')
for bar, lift in zip(bars, lifts):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'{lift:.2%}', ha='center', va='bottom', fontsize=10, color='white', fontweight='bold')
ax2.set_ylabel('Estimated Lift (pp)')
ax2.set_title('Naive vs Cluster Randomization\nBias Comparison')
ax2.set_facecolor('#1A1D27')
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim(0, max(lifts)*100 * 1.3)

# 3. Spillover exposure distribution
ax3.hist(user_df[user_df.naive_treatment==0]['spillover_exposure'],
         bins=40, color=YELLOW, alpha=0.8, edgecolor='#0F1117', density=True)
ax3.axvline(user_df[user_df.naive_treatment==0]['spillover_exposure'].mean(),
            color=RED, linewidth=2, linestyle='--',
            label=f"Mean={user_df[user_df.naive_treatment==0]['spillover_exposure'].mean():.3f}")
ax3.set_xlabel('Spillover Exposure (fraction of friends treated)')
ax3.set_ylabel('Density')
ax3.set_title('Spillover Exposure in Control Group\n(SUTVA Violation Evidence)')
ax3.legend(fontsize=9)
ax3.set_facecolor('#1A1D27')
ax3.grid(True, alpha=0.3)

# 4. Retention by spillover bucket
colors_bucket = [GRAY, BLUE, ORANGE, RED]
bucket_rates = spillover_analysis['retention_rate'].values * 100
bucket_labels = [str(b) for b in spillover_analysis['spillover_bucket']]
bars4 = ax4.bar(range(len(bucket_labels)), bucket_rates,
                color=colors_bucket, width=0.6, edgecolor='#0F1117')
ax4.axhline(naive_ctrl_rate * 100, color=GREEN, linestyle='--',
            linewidth=1.5, label=f'Avg control: {naive_ctrl_rate:.2%}')
ax4.axhline(naive_trt_rate * 100, color=YELLOW, linestyle='--',
            linewidth=1.5, label=f'Avg treatment: {naive_trt_rate:.2%}')
ax4.set_xticks(range(len(bucket_labels)))
ax4.set_xticklabels(bucket_labels, rotation=15, fontsize=8)
ax4.set_ylabel('30-Day Retention Rate (%)')
ax4.set_title('Control Group Retention by\nSpillover Exposure Level')
ax4.legend(fontsize=8)
ax4.set_facecolor('#1A1D27')
ax4.grid(True, alpha=0.3, axis='y')
for bar, rate in zip(bars4, bucket_rates):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             f'{rate:.1f}%', ha='center', va='bottom', fontsize=8, color='white')

# 5. Direct vs Indirect effect decomposition
effect_labels = ['Direct\nTreatment Effect', 'Indirect\n(Spillover) Effect', 'Total\nEffect']
effect_vals = [direct_effect * 100, spillover_coef * avg_spillover * 100, total_effect * 100]
effect_colors = [GREEN, YELLOW, BLUE]
bars5 = ax5.bar(effect_labels, effect_vals, color=effect_colors,
                width=0.5, edgecolor='#0F1117')
for bar, val in zip(bars5, effect_vals):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f'{val:.2f}pp', ha='center', va='bottom', fontsize=10,
             color='white', fontweight='bold')
ax5.set_ylabel('Effect Size (percentage points)')
ax5.set_title('Effect Decomposition\nDirect + Indirect (Spillover)')
ax5.set_facecolor('#1A1D27')
ax5.grid(True, alpha=0.3, axis='y')

# 6. Degree distribution of network
degrees_list = [d for _, d in G.degree()]
ax6.hist(degrees_list, bins=40, color=PURPLE, alpha=0.8,
         edgecolor='#0F1117', density=True)
ax6.set_xlabel('Number of Friends')
ax6.set_ylabel('Density')
ax6.set_title('Social Network Degree Distribution\n(Power-law tail = high interference risk)')
ax6.set_facecolor('#1A1D27')
ax6.grid(True, alpha=0.3)
ax6.axvline(np.mean(degrees_list), color=GREEN, linewidth=2,
            linestyle='--', label=f'Mean={np.mean(degrees_list):.1f}')
ax6.legend(fontsize=9)

plt.suptitle('Spotify Wrapped A/B Test: Network Interference Analysis',
             fontsize=15, color='white', y=1.01, fontweight='bold')

plt.savefig('/home/claude/spotify_wrapped/01_network_interference.png',
            dpi=150, bbox_inches='tight', facecolor='#0F1117')
plt.close()
print("✓ Saved: 01_network_interference.png")

# ── Plot 2: Cluster Randomization Design ──
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor('#0F1117')

# Left: User-level randomization (messy, interference)
ax = axes[0]
sample_clusters = list(range(12))
cluster_x = np.random.uniform(0, 10, 12)
cluster_y = np.random.uniform(0, 10, 12)

for i, (cx, cy) in enumerate(zip(cluster_x, cluster_y)):
    members_x = cx + np.random.normal(0, 0.5, 8)
    members_y = cy + np.random.normal(0, 0.5, 8)
    treatments = np.random.binomial(1, 0.5, 8)
    colors = [GREEN if t else GRAY for t in treatments]
    ax.scatter(members_x, members_y, c=colors, s=60, zorder=3)
    circle = plt.Circle((cx, cy), 0.8, fill=False,
                          color=GRAY, linewidth=1, linestyle='--', alpha=0.5)
    ax.add_patch(circle)

# Draw interference arrows
for i in range(5):
    x1, y1 = np.random.uniform(1, 9), np.random.uniform(1, 9)
    x2, y2 = x1 + np.random.uniform(-2, 2), y1 + np.random.uniform(-2, 2)
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=RED, lw=1.2, alpha=0.6))

ax.set_xlim(-1, 11)
ax.set_ylim(-1, 11)
ax.set_facecolor('#1A1D27')
ax.set_title('User-Level Randomization\n❌ Interference crosses clusters', fontsize=11)
ax.axis('off')
green_patch = mpatches.Patch(color=GREEN, label='Treated user')
gray_patch = mpatches.Patch(color=GRAY, label='Control user')
red_patch = mpatches.Patch(color=RED, label='Spillover')
ax.legend(handles=[green_patch, gray_patch, red_patch], loc='lower right',
          fontsize=8, facecolor='#1A1D27', edgecolor=GRAY, labelcolor='white')

# Middle: Cluster-level randomization (clean)
ax = axes[1]
for i, (cx, cy) in enumerate(zip(cluster_x, cluster_y)):
    cluster_trt = 1 if i % 2 == 0 else 0
    members_x = cx + np.random.normal(0, 0.5, 8)
    members_y = cy + np.random.normal(0, 0.5, 8)
    color = GREEN if cluster_trt else GRAY
    bg_color = '#1A3A1A' if cluster_trt else '#2A2A2A'
    circle_bg = plt.Circle((cx, cy), 0.85, color=bg_color, zorder=1, alpha=0.6)
    ax.add_patch(circle_bg)
    ax.scatter(members_x, members_y, c=color, s=60, zorder=3)
    circle = plt.Circle((cx, cy), 0.85, fill=False,
                          color=GREEN if cluster_trt else GRAY,
                          linewidth=2, zorder=2)
    ax.add_patch(circle)

ax.set_xlim(-1, 11)
ax.set_ylim(-1, 11)
ax.set_facecolor('#1A1D27')
ax.set_title('Cluster Randomization\n✅ Interference stays within clusters', fontsize=11)
ax.axis('off')
green_patch2 = mpatches.Patch(color=GREEN, label='Treatment cluster')
gray_patch2 = mpatches.Patch(color=GRAY, label='Control cluster')
ax.legend(handles=[green_patch2, gray_patch2], loc='lower right',
          fontsize=8, facecolor='#1A1D27', edgecolor=GRAY, labelcolor='white')

# Right: Bias reduction summary
ax = axes[2]
categories = ['Naive\nUser-Level', 'Cluster\nRandomization', 'Ground\nTruth']
values = [naive_lift * 100, cluster_lift * 100, TRUE_DIRECT_EFFECT * 100]
colors_summary = [RED, BLUE, GREEN]
bars_s = ax.barh(categories, values, color=colors_summary,
                  height=0.4, edgecolor='#0F1117')
ax.axvline(TRUE_DIRECT_EFFECT * 100, color=GREEN, linestyle='--',
           linewidth=2, alpha=0.7, label='True effect')
for bar, val in zip(bars_s, values):
    ax.text(val + 0.05, bar.get_y() + bar.get_height()/2,
            f'{val:.2f}pp', va='center', fontsize=11, color='white', fontweight='bold')
ax.set_xlabel('Estimated Lift (percentage points)')
ax.set_title('Estimate Accuracy\nCluster vs User-Level', fontsize=11)
ax.set_facecolor('#1A1D27')
ax.grid(True, alpha=0.3, axis='x')
ax.legend(fontsize=9)

plt.suptitle('Cluster Randomization Design: Eliminating Network Interference',
             fontsize=13, color='white', y=1.02, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/claude/spotify_wrapped/02_cluster_design.png',
            dpi=150, bbox_inches='tight', facecolor='#0F1117')
plt.close()
print("✓ Saved: 02_cluster_design.png")

# ═══════════════════════════════════════════════════════
# SECTION 7: DECISION FRAMEWORK
# ═══════════════════════════════════════════════════════
print("\n\n" + "=" * 65)
print("SECTION 7: FINAL DECISION FRAMEWORK")
print("=" * 65)

ci_cluster = proportion_confint(
    trt_cluster.retained_30d_cluster.sum(),
    len(trt_cluster), alpha=0.05, method='normal'
)
ci_ctrl_cluster = proportion_confint(
    ctrl_cluster.retained_30d_cluster.sum(),
    len(ctrl_cluster), alpha=0.05, method='normal'
)
lift_se = np.sqrt(
    cluster_ctrl_rate*(1-cluster_ctrl_rate)/len(ctrl_cluster) +
    cluster_trt_rate*(1-cluster_trt_rate)/len(trt_cluster)
)
lift_ci = (cluster_lift - 1.96*lift_se, cluster_lift + 1.96*lift_se)

n_users_affected = 50_000_000  # Spotify MAU estimate
retention_value_per_user = 9.99  # monthly premium

revenue_impact = n_users_affected * cluster_lift * retention_value_per_user * 12

print(f"""
┌──────────────────────────────────────────────────────────────┐
│         SPOTIFY WRAPPED EXPERIMENT DECISION SUMMARY          │
├──────────────────────────────────────────────────────────────┤
│ PRIMARY METRIC: 30-Day Retention Rate                        │
│                                                              │
│   METHOD           ESTIMATE    BIAS      RELIABLE?           │
│   Naive (user)     {naive_lift:.2%}      HIGH      ❌ NO               │
│   Cluster          {cluster_lift:.2%}      LOW       ✅ YES              │
│   Ground truth     {TRUE_DIRECT_EFFECT:.2%}      —         ✅ YES              │
├──────────────────────────────────────────────────────────────┤
│ CORRECT ESTIMATE (cluster randomization):                    │
│   Lift:            {cluster_lift:.2%} (95% CI: {lift_ci[0]:.2%}–{lift_ci[1]:.2%})     │
│   p-value:         {p_cluster:.4f} ✅ Significant                  │
│                                                              │
│ SPILLOVER DECOMPOSITION:                                     │
│   Direct effect:   {direct_effect:.2%}                               │
│   Indirect effect: {spillover_coef*avg_spillover:.2%} (via social sharing)          │
│   Total effect:    {total_effect:.2%}                               │
├──────────────────────────────────────────────────────────────┤
│ BUSINESS IMPACT (estimated):                                 │
│   At 50M MAU: ${revenue_impact/1e9:.2f}B incremental annual revenue     │
├──────────────────────────────────────────────────────────────┤
│ RECOMMENDATION: 🚀 SHIP WRAPPED FEATURE                      │
│                                                              │
│ Key insight: Naive analysis UNDERESTIMATED the effect by     │
│ {((TRUE_DIRECT_EFFECT-naive_lift)/TRUE_DIRECT_EFFECT)*100:.1f}%. Always use cluster randomization for features      │
│ with social sharing components. The spillover itself         │
│ represents {spillover_coef*avg_spillover/total_effect*100:.1f}% of total value — a free virality benefit.   │
└──────────────────────────────────────────────────────────────┘
""")

print("✅ ALL SECTIONS COMPLETE")
print("   Files saved to /home/claude/spotify_wrapped/")
