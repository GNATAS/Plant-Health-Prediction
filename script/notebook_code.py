import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier
import lightgbm as lgb



df = pd.read_csv("data\plant_health_data.csv")
print(f"Shape: {df.shape}")
df.head()


print("=== Data Types & Non-null Counts ===")
df.info()


print("=== Missing Values ===")
print(df.isna().sum())
print(f"\nTotal missing: {df.isna().sum().sum()}")


print("=== Descriptive Statistics ===")
df.describe().T.style.format("{:.2f}")


print("=== Target Class Distribution ===")
cnt = df['Plant_Health_Status'].value_counts()
pct = df['Plant_Health_Status'].value_counts(normalize=True) * 100

dist_df = pd.DataFrame({'Count': cnt, 'Percent (%)': pct.round(2)})
display(dist_df)

# Pie chart
fig, ax = plt.subplots(figsize=(6, 4))
fig.patch.set_facecolor('#1e1e2e')
ax.set_facecolor('#1e1e2e')
colors = ['#66BB6A', '#FFA726', '#EF5350']
wedges, texts, autotexts = ax.pie(
    cnt, labels=cnt.index, colors=colors,
    autopct='%1.1f%%', startangle=140,
    textprops={'color': 'white', 'fontsize': 11}
)
for at in autotexts:
    at.set_fontweight('bold')
ax.set_title('Plant Health Status Distribution', color='white', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()


# ── กำหนดค่าคงที่ ──────────────────────────────────────────────────────────
FEATURES    = ['Soil_Moisture','Ambient_Temperature','Soil_Temperature',
               'Humidity','Light_Intensity','Soil_pH',
               'Nitrogen_Level','Phosphorus_Level','Potassium_Level',
               'Chlorophyll_Content','Electrochemical_Signal']
PLANT_IDS   = sorted(df['Plant_ID'].unique())
N_PLANTS    = len(PLANT_IDS)
DARK_BG     = '#1e1e2e'
PANEL_BG    = 'white'
TEXT_COLOR  = 'black'
GRID_COLOR  = '#444'
CMAP_PLANT  = plt.cm.get_cmap('tab10', N_PLANTS)
COLORS      = [CMAP_PLANT(i) for i in range(N_PLANTS)]
print(f"Plant IDs: {PLANT_IDS}  |  N={N_PLANTS}")


ncols = 3
nrows = -(-len(FEATURES) // ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 3.8))
fig.suptitle('Feature Distribution per Plant ID', fontsize=16,
             color=TEXT_COLOR, fontweight='bold', y=1.01)

for ax, feat in zip(axes.flat, FEATURES):
    ax.set_facecolor(PANEL_BG)
    data = [df[df['Plant_ID'] == pid][feat].values for pid in PLANT_IDS]
    bp = ax.boxplot(data, patch_artist=True,
                    medianprops={'color': 'white', 'linewidth': 1.8},
                    whiskerprops={'color': '#aaa'}, capprops={'color': '#aaa'},
                    flierprops={'marker': 'o', 'markersize': 2,
                                'markerfacecolor': '#aaa', 'alpha': 0.5})
    for patch, color in zip(bp['boxes'], COLORS):
        patch.set_facecolor(color); patch.set_alpha(0.75)
    ax.set_title(feat, fontsize=10, color=TEXT_COLOR, fontweight='bold', pad=6)
    ax.set_xticks(range(1, N_PLANTS + 1))
    ax.set_xticklabels([f'P{p}' for p in PLANT_IDS], fontsize=8, color=TEXT_COLOR)
    ax.tick_params(axis='y', colors=TEXT_COLOR, labelsize=8)
    ax.spines[['top','right','left','bottom']].set_color(GRID_COLOR)
    ax.grid(axis='y', linestyle='--', alpha=0.2, color='white')

for ax in axes.flat[len(FEATURES):]:
    ax.set_visible(False)

plt.tight_layout()
plt.show()


mean_df = df.groupby('Plant_ID')[FEATURES].mean()
norm_df = (mean_df - mean_df.min()) / (mean_df.max() - mean_df.min())

fig, ax = plt.subplots(figsize=(14, 5))
ax.set_facecolor(DARK_BG)

im = ax.imshow(norm_df.T, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
ax.set_xticks(range(N_PLANTS))
ax.set_xticklabels([f'Plant {p}' for p in PLANT_IDS], color=TEXT_COLOR, fontsize=10)
ax.set_yticks(range(len(FEATURES)))
ax.set_yticklabels(FEATURES, color=TEXT_COLOR, fontsize=9)
ax.tick_params(colors=TEXT_COLOR)
ax.spines[['top','right','left','bottom']].set_color(GRID_COLOR)

for i, feat in enumerate(FEATURES):
    for j, pid in enumerate(PLANT_IDS):
        raw = mean_df.loc[pid, feat]
        nrm = norm_df.loc[pid, feat]
        tc  = 'black' if 0.3 < nrm < 0.75 else 'white'
        ax.text(j, i, f'{raw:.1f}', ha='center', va='center', fontsize=7,
                color=tc, fontweight='bold')

cbar = fig.colorbar(im, ax=ax, fraction=0.015, pad=0.02)
cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR, labelsize=8)
cbar.set_label('Normalized Mean', color=TEXT_COLOR, fontsize=9)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COLOR)
ax.set_title('🌡️ Mean Feature Value per Plant ID (Normalized | Raw value shown)',
             fontsize=13, color=TEXT_COLOR, fontweight='bold', pad=12)
plt.tight_layout()
plt.show()


TOP_FEATS    = ['Soil_Moisture','Nitrogen_Level','Phosphorus_Level',
                'Chlorophyll_Content','Humidity']
LINE_COLORS  = ['#4FC3F7','#81C784','#FFB74D','#CE93D8','#F48FB1']

fig, axes = plt.subplots(1, len(TOP_FEATS), figsize=(18, 4))
fig.suptitle('📉 Mean ± Std of Key Features across Plant IDs',
             fontsize=14, color=TEXT_COLOR, fontweight='bold', y=1.02)

for ax, feat, color in zip(axes, TOP_FEATS, LINE_COLORS):
    ax.set_facecolor(PANEL_BG)
    means = df.groupby('Plant_ID')[feat].mean()
    stds  = df.groupby('Plant_ID')[feat].std()
    ax.plot(PLANT_IDS, means, marker='o', color=color,
            linewidth=1.8, markersize=6, zorder=3)
    ax.fill_between(PLANT_IDS, means - stds, means + stds, alpha=0.2, color=color)
    ax.set_title(feat, fontsize=9, color=TEXT_COLOR, fontweight='bold', pad=6)
    ax.set_xticks(PLANT_IDS)
    ax.set_xticklabels([f'P{p}' for p in PLANT_IDS], fontsize=8, color=TEXT_COLOR)
    ax.tick_params(axis='y', colors=TEXT_COLOR, labelsize=8)
    ax.spines[['top','right','left','bottom']].set_color(GRID_COLOR)
    ax.grid(linestyle='--', alpha=0.2, color='white')
    ax.set_xlabel('Plant ID', color=TEXT_COLOR, fontsize=8)

plt.tight_layout()
plt.show()


status_order  = ['Healthy', 'Moderate Stress', 'High Stress']
health_colors = {'Healthy': '#66BB6A', 'Moderate Stress': '#FFA726', 'High Stress': '#EF5350'}

status_cnts = (df.groupby(['Plant_ID','Plant_Health_Status'])
                 .size().unstack(fill_value=0)[status_order])
status_pct  = status_cnts.div(status_cnts.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(12, 5))
ax.set_facecolor(PANEL_BG)
bottom = np.zeros(N_PLANTS)

for status in status_order:
    vals = status_pct[status].values
    bars = ax.bar(PLANT_IDS, vals, bottom=bottom,
                  color=health_colors[status], label=status,
                  edgecolor='#1e1e2e', linewidth=0.6)
    for bar, val in zip(bars, vals):
        if val > 6:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_y() + bar.get_height()/2,
                    f'{val:.0f}%', ha='center', va='center',
                    fontsize=8, color='white', fontweight='bold')
    bottom += vals

ax.set_xticks(PLANT_IDS)
ax.set_xticklabels([f'Plant {p}' for p in PLANT_IDS], color=TEXT_COLOR, fontsize=10)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.0f}%'))
ax.tick_params(colors=TEXT_COLOR)
ax.spines[['top','right','left','bottom']].set_color(GRID_COLOR)
ax.set_ylim(0, 105)
ax.set_ylabel('Proportion (%)', color=TEXT_COLOR, fontsize=11)
ax.set_title('🌱 Health Status Distribution per Plant ID',
             fontsize=14, color=TEXT_COLOR, fontweight='bold', pad=12)
ax.legend(framealpha=0.3, labelcolor='white', facecolor='#2e2e3e', fontsize=10)
ax.grid(axis='y', linestyle='--', alpha=0.2, color='white')
plt.tight_layout()
plt.show()


# Custom ordinal mapping
custom_mapping = {'High Stress': 2, 'Moderate Stress': 1, 'Healthy': 0}
df['Plant_Health_Status_Encoded'] = df['Plant_Health_Status'].map(custom_mapping)

unique_value_counts = df['Plant_Health_Status_Encoded'].value_counts()
print(unique_value_counts)


# Drop Timestamp และ text label 
df = df.drop(['Timestamp', 'Plant_Health_Status','Plant_Name'], axis=1)
print("Columns after drop:", df.columns.tolist())
df.head()


numerical_features = df.select_dtypes(include=[np.number]).columns

z_scores = zscore(df[numerical_features])
outliers_zscore = (np.abs(z_scores) > 3).sum(axis=0)
print("Outliers Detected with Z-scores:")
display(pd.DataFrame({'Column': numerical_features,
                      'Outlier Count': outliers_zscore}))


correlations = df.corr()['Plant_Health_Status_Encoded'].sort_values(ascending=False)
correlation_table = correlations.to_frame(
    name='Correlation with Plant_Health_Status_Encoded').reset_index()
correlation_table.rename(columns={'index': 'Feature'}, inplace=True)

display(correlation_table.style
        .bar(subset=['Correlation with Plant_Health_Status_Encoded'],
             align='mid', color=['#EF5350', '#66BB6A'])
        .format({'Correlation with Plant_Health_Status_Encoded': '{:.4f}'}))


X = df.drop(columns=['Plant_Health_Status_Encoded'])
y = df['Plant_Health_Status_Encoded']
FEATURE_NAMES = list(X.columns)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Features : {FEATURE_NAMES}")
print(f"X_train  : {X_train.shape}")
print(f"X_test   : {X_test.shape}")
print(f"y_train  : {y_train.shape}")
print(f"y_test   : {y_test.shape}")


selector = SelectKBest(score_func=mutual_info_classif, k='all')
selector.fit(X_train, y_train)

feature_scores = pd.DataFrame({
    'Feature':  FEATURE_NAMES,
    'MI Score': selector.scores_
}).sort_values('MI Score', ascending=False).reset_index(drop=True)

print("📊 Mutual Information Scores (incl. Plant_ID):")
display(feature_scores.style
        .bar(subset=['MI Score'], color='#4C72B0')
        .format({'MI Score': '{:.4f}'})
        .set_caption('Feature Importance — Mutual Information'))

# Visualize
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_facecolor(PANEL_BG)
colors_bar = ['#4FC3F7' if s > 0.05 else '#555' for s in feature_scores['MI Score']]
bars = ax.barh(feature_scores['Feature'], feature_scores['MI Score'],
               color=colors_bar, edgecolor='black', linewidth=0.4)
ax.axvline(0.05, color='#FFA726', linestyle='--', linewidth=1.5, label='Threshold = 0.05')
for bar, val in zip(bars, feature_scores['MI Score']):
    ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=8, color='black')
ax.set_xlabel('MI Score', color=TEXT_COLOR)
ax.tick_params(colors=TEXT_COLOR)
ax.spines[['top','right','left','bottom']].set_color(GRID_COLOR)
ax.invert_yaxis()
ax.legend(framealpha=0.3, labelcolor='black', facecolor='#2e2e3e')
ax.set_title('Feature Importance (Mutual Information)', fontsize=13,
             color=TEXT_COLOR, fontweight='bold')
plt.tight_layout()
plt.savefig('feature_importance_mi.png', dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.show()


THRESHOLD = 0.05
selected_features = feature_scores[feature_scores['MI Score'] > THRESHOLD]['Feature'].tolist()
selected_idx = [FEATURE_NAMES.index(f) for f in selected_features]

X_train_fs = X_train[:, selected_idx]
X_test_fs  = X_test[:, selected_idx]

plant_id_in = 'Plant_ID' in selected_features
print(f"✅ Selected {len(selected_features)} features [MI > {THRESHOLD}]:")
for f in selected_features:
    score = feature_scores.loc[feature_scores['Feature']==f, 'MI Score'].values[0]
    print(f"   • {f:30s}  MI={score:.4f}")
print(f"\nPlant_ID ผ่าน threshold? → {'✅ YES — ยังอยู่ใน selected set' if plant_id_in else '❌ NO — ถูกตัดออกโดย selection'}")
print(f"Feature count: {X_train.shape[1]} → {X_train_fs.shape[1]}")


smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train_fs, y_train)

label_map = {0: 'Healthy', 1: 'Moderate Stress', 2: 'High Stress'}
before = y_train.value_counts().rename(label_map)
after  = pd.Series(y_train_sm).value_counts().rename(label_map)

smote_compare = pd.DataFrame({'Before SMOTE': before, 'After SMOTE': after})
smote_compare['Δ Added'] = smote_compare['After SMOTE'] - smote_compare['Before SMOTE']
display(smote_compare)
print(f"\nTotal samples: {len(y_train)} → {len(y_train_sm)} (+{len(y_train_sm)-len(y_train)})")

# Bar comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

STATUS_COLORS = ['#66BB6A', '#FFA726', '#EF5350']
for ax, (title, data) in zip(axes, [('Before SMOTE', before), ('After SMOTE', after)]):

    bars = ax.bar(data.index, data.values, color=STATUS_COLORS,
                  edgecolor='white', linewidth=0.5)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                str(int(bar.get_height())), ha='center', fontsize=11,
                color='white', fontweight='bold')
    ax.set_title(title, color=TEXT_COLOR, fontsize=12, fontweight='bold')
    ax.tick_params(colors=TEXT_COLOR); ax.set_ylim(0, max(after.values) * 1.15)
    ax.spines[['top','right','left','bottom']].set_color(GRID_COLOR)
    ax.grid(axis='y', linestyle='--', alpha=0.2, color='white')
plt.tight_layout()
plt.show()


CLASS_NAMES = ['Healthy', 'Moderate Stress', 'High Stress']

def get_models():
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree':       DecisionTreeClassifier(max_depth=10, random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
        'SVM':                 SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42),
        'Gradient Boosting':   GradientBoostingClassifier(n_estimators=200, learning_rate=0.1,
                                                           max_depth=5, random_state=42),
        'LightGBM':            lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1,
                                                   max_depth=5, random_state=42, verbose=-1),
        'CatBoost':            CatBoostClassifier(iterations=200, learning_rate=0.1,
                                                  depth=5, random_state=42, verbose=0),
    }

def run_experiment(X_tr, y_tr, X_te, y_te, exp_name):
    print(f"\n{'━'*60}")
    print(f"{exp_name}")
    print(f"{'━'*60}")
    models = get_models()
    rows   = []
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        rows.append({
            'Model':      name,
            'Accuracy':   round(accuracy_score(y_te, y_pred), 4),
            'Precision':  round(precision_score(y_te, y_pred, average='weighted', zero_division=0), 4),
            'Recall':     round(recall_score(y_te, y_pred, average='weighted', zero_division=0), 4),
            'F1-Score':   round(f1_score(y_te, y_pred, average='weighted', zero_division=0), 4),
            '_y_pred':    y_pred,
            '_model_obj': model,  
        })
        print(f"  [{name:22s}]  Acc={rows[-1]['Accuracy']:.4f}  F1={rows[-1]['F1-Score']:.4f}")
    df_res = pd.DataFrame(rows).sort_values('F1-Score', ascending=False).reset_index(drop=True)
    df_res.index += 1
    return df_res 

DISP_COLS = ['Model','Accuracy','Precision','Recall','F1-Score']

def show_table(df_res, caption):
    display(df_res[DISP_COLS].style
            .format({c: '{:.4f}' for c in ['Accuracy','Precision','Recall','F1-Score']})
            .set_caption(caption))



df_exp1 = run_experiment(X_train, y_train, X_test, y_test,
                         f'EXP 1 — Baseline ({X_train.shape[1]} features, No SMOTE)')
show_table(df_exp1, 'EXP 1 — Baseline (All Features)')


df_exp2 = run_experiment(X_train_fs, y_train, X_test_fs, y_test,
                         f'EXP 2 — Feature Selection ({X_train_fs.shape[1]} features, No SMOTE)')
show_table(df_exp2, f'EXP 2 — Feature Selection: {selected_features}')


df_exp3 = run_experiment(X_train_sm, y_train_sm, X_test_fs, y_test,
                         f'EXP 3 — Feature Selection + SMOTE ({X_train_fs.shape[1]} features)')
show_table(df_exp3, 'EXP 3 — Feature Selection + SMOTE')


import joblib
import os

os.makedirs('saved_models', exist_ok=True)

# ── Save best model จากแต่ละ experiment ──────────────────────
EXP_MAP = {
    'exp1_baseline': df_exp1,
    'exp2_feat_sel': df_exp2,
    'exp3_fs_smote': df_exp3,
}

for exp_name, df_e in EXP_MAP.items():
    best       = df_e.iloc[0]
    model_name = best['Model'].replace(' ', '_').lower()
    model_obj  = best['_model_obj']
    path       = f'saved_models/{exp_name}_{model_name}.pkl'

    joblib.dump(model_obj, path)
    print(f'Saved: {path}')

# ── Save scaler ด้วย (สำคัญมาก!) ─────────────────────────────
joblib.dump(scaler, 'saved_models/scaler.pkl')
print('Saved: saved_models/scaler.pkl')

print('\n✅ Done! Files in saved_models/:')
for f in os.listdir('saved_models'):
    size = os.path.getsize(f'saved_models/{f}') / 1024
    print(f'   {f:50s}  {size:.1f} KB')


EXP_MAP = {
    'EXP1 (Baseline)':   df_exp1,
    'EXP2 (Feat.Sel.)':  df_exp2,
    'EXP3 (FS+SMOTE)':   df_exp3,
}

model_order = df_exp1['Model'].tolist()
comparison  = pd.DataFrame({'Model': model_order})
for label, df_e in EXP_MAP.items():
    tmp = df_e[['Model','F1-Score']].rename(columns={'F1-Score': label})
    comparison = comparison.merge(tmp, on='Model', how='left')

comparison = comparison.set_index('Model')
comparison['Δ (EXP3−EXP1)'] = (comparison['EXP3 (FS+SMOTE)'] - comparison['EXP1 (Baseline)']).round(4)

display(comparison.style
        .map(lambda v: 'color:#66BB6A;font-weight:bold' if isinstance(v, float) and v > 0
                  else ('color:#EF5350;font-weight:bold' if isinstance(v, float) and v < 0 else ''),
                  subset=['Δ (EXP3−EXP1)'])
        .format({k: '{:.4f}' for k in list(EXP_MAP.keys()) + ['Δ (EXP3−EXP1)']})
        .set_caption('F1-Score (weighted) | Δ = EXP3 − EXP1'))


exp_cols   = list(EXP_MAP.keys())
exp_labels = ['Baseline\n(all feat.)', 'Feat. Selection\n(no SMOTE)', 'Feat. Sel.\n+ SMOTE']
palette    = ['#4C72B0', '#55A868', '#DD8452']
x          = np.arange(len(model_order))
bar_width  = 0.25

fig, ax = plt.subplots(figsize=(16, 6))

for i, (col, label, color) in enumerate(zip(exp_cols, exp_labels, palette)):
    vals = comparison[col].values
    bars = ax.bar(x + i * bar_width, vals, bar_width, label=label,
                  color=color, alpha=0.88, edgecolor='black', linewidth=0.4)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.003, f'{h:.3f}',
                ha='center', va='bottom', fontsize=7, color='black', fontweight='bold')

ax.set_xticks(x + bar_width)
ax.set_xticklabels(model_order, rotation=18, ha='right', fontsize=10, color='black')
ax.set_ylim(0, 1.12)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.0%}'))
ax.tick_params(colors='black')
ax.spines[['top','right','left','bottom']].set_color('#444')
ax.legend(framealpha=0.3, labelcolor='black', facecolor='#2e2e3e', fontsize=10)
ax.set_title('F1-Score Comparison: Baseline vs Feature Selection vs FS+SMOTE',
             fontsize=13, color='black', fontweight='bold', pad=14)
ax.set_ylabel('F1-Score (weighted)', color='black', fontsize=11)
ax.grid(axis='y', linestyle='--', alpha=0.25, color='black')
plt.tight_layout()
plt.savefig('experiment_comparison.png', dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.show()


fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle('Confusion Matrices — Best Model per Experiment',
             fontsize=13, color='white', fontweight='bold', y=1.02)

overall_best = {'f1': -1, 'model': '', 'exp': ''}

for ax, (exp_label, df_e), short in zip(
        axes, EXP_MAP.items(), ['Baseline','Feat.Sel.','FS+SMOTE']):
    best = df_e.iloc[0]
    cm   = confusion_matrix(y_test, best['_y_pred'])
    ax.set_facecolor(DARK_BG)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    if best['F1-Score'] > overall_best['f1']:
        overall_best = {'f1': best['F1-Score'], 'model': best['Model'], 'exp': short}
    ax.set_title(f'{short}\n{best["Model"]}\nF1={best["F1-Score"]:.4f}',
                 fontsize=10, color='white', fontweight='bold', pad=8)
    ax.xaxis.label.set_color('white'); ax.yaxis.label.set_color('white')
    ax.tick_params(colors='white', labelsize=8)
    for text in disp.text_.ravel(): text.set_color('black')
    ax.spines[['top','right','left','bottom']].set_color('#444')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.show()

print(f'\n🏆 Best Overall:')
print(f'   Experiment : {overall_best["exp"]}')
print(f'   Model      : {overall_best["model"]}')
print(f'   F1-Score   : {overall_best["f1"]:.4f}')


# หา experiment และ model ที่ดีที่สุด
best_exp_df = max(EXP_MAP.values(), key=lambda d: d.iloc[0]['F1-Score'])
best_row    = best_exp_df.iloc[0]

print(f"🏆 Best Model: {best_row['Model']}")
print(f"\n{classification_report(y_test, best_row['_y_pred'], target_names=CLASS_NAMES)}")


import joblib

model = joblib.load("saved_models/best_model_exp2_feat_sel_decision_tree.pkl")

tree = model.tree_
feature_names = ["Soil_Moisture", "Nitrogen_Level"]

for i in range(tree.node_count):
    if tree.feature[i] != -2:
        name = feature_names[tree.feature[i]]
        threshold = tree.threshold[i]
        print(f"{name} <= {threshold:.2f}")

from sklearn.tree import export_text

rules = export_text(model, feature_names=["Soil_Moisture", "Nitrogen_Level"])
print(rules)

import joblib

scaler = joblib.load("saved_models/scaler.pkl")

mean = scaler.mean_
std = scaler.scale_

print(mean)
print(std)

import numpy as np
import joblib

# =========================
# LOAD MODEL + SCALER
# =========================
model  = joblib.load("saved_models/best_model_exp2_feat_sel_decision_tree.pkl")
scaler = joblib.load("saved_models/scaler.pkl")

# =========================
# CONFIG
# =========================
ALL_FEATURE_NAMES = [
    'Plant_ID', 'Soil_Moisture', 'Ambient_Temperature', 'Soil_Temperature',
    'Humidity', 'Light_Intensity', 'Soil_pH', 'Nitrogen_Level',
    'Phosphorus_Level', 'Potassium_Level', 'Chlorophyll_Content',
    'Electrochemical_Signal',
]

SELECTED_FEATURES = ['Soil_Moisture', 'Nitrogen_Level']
SELECTED_INDICES  = [ALL_FEATURE_NAMES.index(f) for f in SELECTED_FEATURES]

label_map = {
    0: "Healthy",
    1: "Moderate Stress",
    2: "High Stress"
}

# =========================
# PREDICT FUNCTION
# =========================
def predict_one(soil_val, nitrogen_val):
    full_input = np.zeros((1, len(ALL_FEATURE_NAMES)))
    full_input[0, ALL_FEATURE_NAMES.index('Soil_Moisture')]  = soil_val
    full_input[0, ALL_FEATURE_NAMES.index('Nitrogen_Level')] = nitrogen_val
    
    scaled_full = scaler.transform(full_input)
    scaled_selected = scaled_full[:, SELECTED_INDICES]
    
    pred = int(model.predict(scaled_selected)[0])
    return label_map[pred]

# =========================
# 🔥 YOUR NEW THRESHOLDS
# =========================

def soil_level(x):
    if x <= 20:
        return "Dry"
    elif x <= 30:
        return "Moist"
    else:
        return "Wet"

def nitrogen_level(x):
    if x <= 19.5:
        return "Yellow"
    elif x <= 21:
        return "Green"
    else:
        return "Dark Green"

# =========================
# REPRESENTATIVE VALUES
# =========================

soil_rep = {
    "Dry": 15,      
    "Moist": 25,
    "Wet": 35
}

nitrogen_rep = {
    "Yellow": 18,   
    "Green": 20.5,
    "Dark Green": 30
}

# =========================
# 🔥 TEST GRID (3x3)
# =========================
print("\n=== TEST 3x3 GRID ===")

for s_label, s_val in soil_rep.items():
    for n_label, n_val in nitrogen_rep.items():
        pred = predict_one(s_val, n_val)
        print(f"{s_label:>6} x {n_label:>10}  ->  {pred}")



from sklearn.model_selection import train_test_split
custom_mapping = {
    'High Stress': 2,
    'Moderate Stress': 1,
    'Healthy': 0
}

df['label'] = df['Plant_Health_Status'].map(custom_mapping)
# เตรียม data
df_work = df.copy()
df_work = df_work.drop(['Timestamp', 'Plant_Health_Status', 'Plant_Name'], axis=1)

X = df_work.drop(columns=['label'])
y = df_work['label']

# scale
X_scaled = scaler.transform(X)

# split
_, X_test, _, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# inverse กลับมาเป็นค่าจริง
X_test_unscaled = scaler.inverse_transform(X_test)

# index ของ feature
sm_idx = ALL_FEATURE_NAMES.index('Soil_Moisture')
nl_idx = ALL_FEATURE_NAMES.index('Nitrogen_Level')

correct = 0
total = 0

for i in range(len(X_test)):
    raw_sm = X_test_unscaled[i, sm_idx]
    raw_nl = X_test_unscaled[i, nl_idx]

    s_level = soil_level(raw_sm)
    n_level = nitrogen_level(raw_nl)

    pred = predict_one(
        soil_rep[s_level],
        nitrogen_rep[n_level]
    )

    if pred == label_map[y_test.iloc[i]]:
        correct += 1
    total += 1

print("Accuracy (new rule):", correct / total)

from sklearn.metrics import classification_report, confusion_matrix

# เก็บ prediction
y_pred = []

for i in range(len(X_test)):
    raw_sm = X_test_unscaled[i, sm_idx]
    raw_nl = X_test_unscaled[i, nl_idx]

    s_level = soil_level(raw_sm)
    n_level = nitrogen_level(raw_nl)

    pred = predict_one(
        soil_rep[s_level],
        nitrogen_rep[n_level]
    )

    # แปลงชื่อกลับเป็น id (สำคัญ)
    pred_id = {v: k for k, v in label_map.items()}[pred]

    y_pred.append(pred_id)

y_pred = np.array(y_pred)

# =========================
# 🔥 METRICS
# =========================

print("\n=== LEVEL-BASED RESULTS ===")

print("\nAccuracy:")
print(correct / total)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Healthy", "Moderate Stress", "High Stress"]
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))