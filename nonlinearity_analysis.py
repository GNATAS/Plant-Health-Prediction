"""
Non-Linearity Analysis for Plant Health Prediction Dataset
==========================================================
รัน script นี้เพื่อสร้าง visualization แสดง trend ของ data
และวิเคราะห์ว่าความสัมพันธ์เป็นแบบ Non-linear หรือไม่

วิธีรัน:
    python nonlinearity_analysis.py
    หรือ copy แต่ละ section ไปรันใน Jupyter Notebook ที่มีอยู่แล้ว
"""

import matplotlib
matplotlib.use('Agg')  # ใช้ backend ที่ไม่ต้องการ display (ถ้ารันบน terminal)

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import f_oneway
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA

# ── Config ──────────────────────────────────────────────────────────────────
DARK_BG     = '#1e1e2e'
PANEL_BG    = '#2a2a3e'
TEXT_COLOR  = 'white'
GRID_COLOR  = '#444'
CLASS_ORDER  = ['Healthy', 'Moderate Stress', 'High Stress']
CLASS_COLORS = {'Healthy': '#4CAF50', 'Moderate Stress': '#FFA726', 'High Stress': '#EF5350'}
PALETTE      = [CLASS_COLORS[c] for c in CLASS_ORDER]
KEY_FEATURES = [
    'Soil_Moisture', 'Chlorophyll_Content', 'Nitrogen_Level',
    'Electrochemical_Signal', 'Light_Intensity', 'Humidity'
]
ALL_FEATURES = [
    'Soil_Moisture','Ambient_Temperature','Soil_Temperature',
    'Humidity','Light_Intensity','Soil_pH',
    'Nitrogen_Level','Phosphorus_Level','Potassium_Level',
    'Chlorophyll_Content','Electrochemical_Signal'
]

# ── Load data ────────────────────────────────────────────────────────────────
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path   = os.path.join(script_dir, 'data', 'plant_health_data.csv')
df = pd.read_csv(csv_path)
print(f"Data loaded: {df.shape}")
print(df['Plant_Health_Status'].value_counts())


# ════════════════════════════════════════════════════════════════════════════
# PLOT 1: Violin Plot — Feature distribution per class
# ════════════════════════════════════════════════════════════════════════════
print("\n[1/6] Violin Plots ...")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle(
    'Feature Distribution per Plant Health Class  (Violin Plot)\n'
    'ดูการกระจายและ overlap ของแต่ละ class — ถ้า distribution ต่างกันชัด = feature discriminative',
    fontsize=13, color=TEXT_COLOR, fontweight='bold', y=1.01
)

for ax, feat in zip(axes.ravel(), KEY_FEATURES):
    ax.set_facecolor(PANEL_BG)
    sns.violinplot(
        data=df, x='Plant_Health_Status', y=feat,
        order=CLASS_ORDER, palette=PALETTE,
        inner='box', cut=0, ax=ax
    )
    ax.set_title(feat, color=TEXT_COLOR, fontsize=11, fontweight='bold')
    ax.set_xlabel('Health Status', color=TEXT_COLOR, fontsize=9)
    ax.set_ylabel(feat, color=TEXT_COLOR, fontsize=9)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax.tick_params(axis='x', labelrotation=15)
    ax.set_xticklabels(ax.get_xticklabels(), color=TEXT_COLOR)
    ax.spines[['top','right','left','bottom']].set_color(GRID_COLOR)
    ax.grid(axis='y', color=GRID_COLOR, linestyle='--', alpha=0.4)

plt.tight_layout()
out1 = os.path.join(script_dir, 'nl_violin.png')
plt.savefig(out1, dpi=120, bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print(f"   → saved: {out1}")


# ════════════════════════════════════════════════════════════════════════════
# PLOT 2: Scatter + LOWESS trend line
# ════════════════════════════════════════════════════════════════════════════
print("[2/6] Scatter + LOWESS ...")
try:
    from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("   ⚠  statsmodels not found — LOWESS curves skipped (pip install statsmodels)")

SCATTER_PAIRS = [
    ('Soil_Moisture',       'Chlorophyll_Content'),
    ('Nitrogen_Level',      'Electrochemical_Signal'),
    ('Light_Intensity',     'Chlorophyll_Content'),
    ('Humidity',            'Soil_Moisture'),
    ('Soil_pH',             'Nitrogen_Level'),
    ('Ambient_Temperature', 'Electrochemical_Signal'),
]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle(
    'Scatter Plot + LOWESS Trend (per class)\n'
    'เส้นโค้ง LOWESS ที่ไม่ตรง = ความสัมพันธ์แบบ Non-linear',
    fontsize=13, color=TEXT_COLOR, fontweight='bold', y=1.01
)

legend_patches = [mpatches.Patch(color=CLASS_COLORS[c], label=c) for c in CLASS_ORDER]

for ax, (xf, yf) in zip(axes.ravel(), SCATTER_PAIRS):
    ax.set_facecolor(PANEL_BG)
    for cls, col in CLASS_COLORS.items():
        sub = df[df['Plant_Health_Status'] == cls]
        ax.scatter(sub[xf], sub[yf], c=col, alpha=0.25, s=12)
        if HAS_STATSMODELS:
            xs = sub[xf].values; ys = sub[yf].values
            idx = np.argsort(xs)
            try:
                smooth = sm_lowess(ys[idx], xs[idx], frac=0.4, return_sorted=True)
                ax.plot(smooth[:, 0], smooth[:, 1], c=col, linewidth=2.5, alpha=0.9)
            except Exception:
                pass
    ax.set_xlabel(xf, color=TEXT_COLOR, fontsize=9)
    ax.set_ylabel(yf, color=TEXT_COLOR, fontsize=9)
    ax.set_title(f'{xf}  vs  {yf}', color=TEXT_COLOR, fontsize=10, fontweight='bold')
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax.spines[['top','right','left','bottom']].set_color(GRID_COLOR)
    ax.grid(color=GRID_COLOR, linestyle='--', alpha=0.3)

fig.legend(handles=legend_patches, loc='lower center', ncol=3,
           frameon=False, fontsize=10, labelcolor=TEXT_COLOR, bbox_to_anchor=(0.5, -0.04))
plt.tight_layout()
out2 = os.path.join(script_dir, 'nl_scatter_lowess.png')
plt.savefig(out2, dpi=120, bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print(f"   → saved: {out2}")


# ════════════════════════════════════════════════════════════════════════════
# PLOT 3: Normalized Feature Means per class (grouped bar)
# ════════════════════════════════════════════════════════════════════════════
print("[3/6] Class-mean bar chart ...")
scaler = MinMaxScaler()
df_norm = df.copy()
df_norm[ALL_FEATURES] = scaler.fit_transform(df[ALL_FEATURES])

class_means = df_norm.groupby('Plant_Health_Status')[ALL_FEATURES].mean().loc[CLASS_ORDER]

fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor(DARK_BG)
ax.set_facecolor(PANEL_BG)

x = np.arange(len(ALL_FEATURES))
width = 0.25
for i, (cls, col) in enumerate(CLASS_COLORS.items()):
    vals = class_means.loc[cls].values
    ax.bar(x + i*width, vals, width, label=cls, color=col, alpha=0.85, edgecolor='none')

ax.set_xticks(x + width)
ax.set_xticklabels(ALL_FEATURES, rotation=35, ha='right', color=TEXT_COLOR, fontsize=9)
ax.set_ylabel('Normalized Mean (0–1)', color=TEXT_COLOR, fontsize=10)
ax.set_title(
    'Normalized Feature Means per Health Class\n'
    'Pattern ที่ไม่ monotone ระหว่าง class บ่งชี้ Non-linear interaction',
    color=TEXT_COLOR, fontsize=12, fontweight='bold'
)
ax.tick_params(colors=TEXT_COLOR)
ax.spines[['top','right','left','bottom']].set_color(GRID_COLOR)
ax.grid(axis='y', color=GRID_COLOR, linestyle='--', alpha=0.4)
ax.legend(frameon=False, labelcolor=TEXT_COLOR, fontsize=10)
plt.tight_layout()
out3 = os.path.join(script_dir, 'nl_class_means.png')
plt.savefig(out3, dpi=120, bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print(f"   → saved: {out3}")


# ════════════════════════════════════════════════════════════════════════════
# PLOT 4: Binned class-proportion heatmap
# ════════════════════════════════════════════════════════════════════════════
print("[4/6] Binned heatmap ...")
N_BINS = 8
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle(
    'Class Proportion per Feature Bin (Heatmap)\n'
    'สีที่สลับกัน (non-monotone) = class proportion เปลี่ยนแบบ Non-linear',
    fontsize=13, color=TEXT_COLOR, fontweight='bold', y=1.01
)

for ax, feat in zip(axes.ravel(), KEY_FEATURES):
    ax.set_facecolor(PANEL_BG)
    df_tmp = df.copy()
    df_tmp['_bin'] = pd.cut(df_tmp[feat], bins=N_BINS, labels=False)
    pivot = df_tmp.groupby(['_bin','Plant_Health_Status']).size().unstack(fill_value=0)
    pivot = pivot.div(pivot.sum(axis=1), axis=0)
    pivot = pivot.reindex(columns=CLASS_ORDER, fill_value=0)
    pivot_T = pivot.T

    im = ax.imshow(pivot_T.values, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=1)
    ax.set_yticks(range(len(CLASS_ORDER)))
    ax.set_yticklabels(CLASS_ORDER, color=TEXT_COLOR, fontsize=8)
    ax.set_xticks(range(N_BINS))
    bin_edges = pd.cut(df_tmp[feat], bins=N_BINS, retbins=True)[1]
    ax.set_xticklabels([f'{v:.1f}' for v in bin_edges[:-1]], rotation=30,
                       ha='right', color=TEXT_COLOR, fontsize=7)
    ax.set_xlabel(f'{feat} (bin start)', color=TEXT_COLOR, fontsize=9)
    ax.set_title(feat, color=TEXT_COLOR, fontsize=10, fontweight='bold')
    ax.tick_params(colors=TEXT_COLOR)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=TEXT_COLOR)

plt.tight_layout()
out4 = os.path.join(script_dir, 'nl_bin_heatmap.png')
plt.savefig(out4, dpi=120, bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print(f"   → saved: {out4}")


# ════════════════════════════════════════════════════════════════════════════
# PLOT 5: ANOVA F-score + PCA 2D scatter
# ════════════════════════════════════════════════════════════════════════════
print("[5/6] ANOVA F-score + PCA ...")
groups = [df[df['Plant_Health_Status']==c][ALL_FEATURES].values for c in CLASS_ORDER]
f_scores = []
for i, feat in enumerate(ALL_FEATURES):
    g = [grp[:, i] for grp in groups]
    f_val, _ = f_oneway(*g)
    f_scores.append(f_val)

f_df = pd.DataFrame({'Feature': ALL_FEATURES, 'F-score': f_scores}).sort_values('F-score', ascending=True)

sc2 = StandardScaler()
X_scaled = sc2.fit_transform(df[ALL_FEATURES])
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor(DARK_BG)

ax = axes[0]
ax.set_facecolor(PANEL_BG)
bar_colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(f_df)))
bars_h = ax.barh(f_df['Feature'], f_df['F-score'], color=bar_colors, edgecolor='none')
ax.set_xlabel('ANOVA F-Score', color=TEXT_COLOR, fontsize=10)
ax.set_title('Feature Discriminability (ANOVA)\nHigher F → separates classes better',
             color=TEXT_COLOR, fontsize=11, fontweight='bold')
ax.tick_params(colors=TEXT_COLOR)
ax.spines[['top','right','left','bottom']].set_color(GRID_COLOR)
ax.grid(axis='x', color=GRID_COLOR, linestyle='--', alpha=0.4)
for val, bar in zip(f_df['F-score'], bars_h):
    ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}', va='center', ha='left', color=TEXT_COLOR, fontsize=8)

ax2 = axes[1]
ax2.set_facecolor(PANEL_BG)
for cls, col in CLASS_COLORS.items():
    mask = df['Plant_Health_Status'] == cls
    ax2.scatter(X_pca[mask, 0], X_pca[mask, 1],
                c=col, label=cls, alpha=0.45, s=18, edgecolors='none')
ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)', color=TEXT_COLOR, fontsize=10)
ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)', color=TEXT_COLOR, fontsize=10)
ax2.set_title(
    'PCA 2D Projection\n(class ปะปนกัน = ต้องการ Non-linear boundary)',
    color=TEXT_COLOR, fontsize=11, fontweight='bold'
)
ax2.tick_params(colors=TEXT_COLOR)
ax2.spines[['top','right','left','bottom']].set_color(GRID_COLOR)
ax2.grid(color=GRID_COLOR, linestyle='--', alpha=0.3)
ax2.legend(frameon=False, labelcolor=TEXT_COLOR, fontsize=10)

plt.tight_layout()
out5 = os.path.join(script_dir, 'nl_anova_pca.png')
plt.savefig(out5, dpi=120, bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print(f"   → saved: {out5}")


# ════════════════════════════════════════════════════════════════════════════
# PLOT 6: Linear vs Non-linear model comparison (5-fold CV)
# ════════════════════════════════════════════════════════════════════════════
print("[6/6] Linear vs Non-linear model comparison ...")
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

le2 = LabelEncoder()
y_all = le2.fit_transform(df['Plant_Health_Status'])
X_all = StandardScaler().fit_transform(df[ALL_FEATURES])

models_compare = {
    'Logistic\nRegression\n(Linear)':  LogisticRegression(max_iter=1000, random_state=42),
    'Linear SVM\n(Linear kernel)':     SVC(kernel='linear', random_state=42),
    'Decision Tree\n(Non-linear)':     DecisionTreeClassifier(max_depth=5, random_state=42),
    'RBF SVM\n(Non-linear)':           SVC(kernel='rbf', random_state=42),
}

cv_results = {}
for name, model in models_compare.items():
    scores = cross_val_score(model, X_all, y_all, cv=5, scoring='f1_weighted')
    cv_results[name] = scores
    print(f"   {name.replace(chr(10),' '):35s}: F1={scores.mean():.4f} ± {scores.std():.4f}")

fig, ax = plt.subplots(figsize=(12, 5))
fig.patch.set_facecolor(DARK_BG)
ax.set_facecolor(PANEL_BG)

names  = list(cv_results.keys())
means  = [cv_results[n].mean() for n in names]
stds   = [cv_results[n].std()  for n in names]
b_cols = ['#5C6BC0','#42A5F5','#4CAF50','#FF7043']

bars = ax.bar(names, means, color=b_cols, alpha=0.85, edgecolor='none', width=0.55,
              yerr=stds, capsize=6, error_kw=dict(ecolor='white', elinewidth=1.5))
ax.set_ylim(0, 1.1)
ax.set_ylabel('5-Fold CV F1-Weighted', color=TEXT_COLOR, fontsize=11)
ax.set_title(
    'Linear vs Non-linear Model Comparison  (5-Fold CV)\n'
    'ถ้า Non-linear models ชนะ → Data มี Non-linear structure',
    color=TEXT_COLOR, fontsize=12, fontweight='bold'
)
ax.tick_params(colors=TEXT_COLOR)
ax.set_xticklabels(names, color=TEXT_COLOR, fontsize=10)
ax.spines[['top','right','left','bottom']].set_color(GRID_COLOR)
ax.grid(axis='y', color=GRID_COLOR, linestyle='--', alpha=0.4)

best_idx = int(np.argmax(means))
for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
    ax.text(bar.get_x()+bar.get_width()/2, mean+std+0.02,
            f'{mean:.4f}', ha='center', va='bottom', color=TEXT_COLOR,
            fontsize=10, fontweight='bold')
bars[best_idx].set_edgecolor('gold')
bars[best_idx].set_linewidth(2.5)
ax.axhline(y=max(means), color='gold', linestyle=':', alpha=0.6, linewidth=1.5)

plt.tight_layout()
out6 = os.path.join(script_dir, 'nl_model_compare.png')
plt.savefig(out6, dpi=120, bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print(f"   → saved: {out6}")

# ── Final summary ────────────────────────────────────────────────────────────
lr_best  = max(cv_results['Logistic\nRegression\n(Linear)'].mean(),
               cv_results['Linear SVM\n(Linear kernel)'].mean())
nl_best  = max(cv_results['Decision Tree\n(Non-linear)'].mean(),
               cv_results['RBF SVM\n(Non-linear)'].mean())

print("\n" + "="*60)
print("📌  SUMMARY — Non-linearity Evidence")
print("="*60)
print(f"   Best Linear model F1  : {lr_best:.4f}")
print(f"   Best Non-linear F1    : {nl_best:.4f}")
if nl_best > lr_best:
    print(f"   ✅ Non-linear models ชนะ → Data เป็น NON-LINEAR")
else:
    print(f"   ⚠️  Linear models แค่พอกัน → Data อาจ linearly separable")
print(f"\nGenerated plots:")
for p in [out1, out2, out3, out4, out5, out6]:
    print(f"   {p}")
