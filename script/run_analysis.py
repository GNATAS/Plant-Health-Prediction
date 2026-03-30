import numpy as np
import joblib
import pandas as pd
import os
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(os.path.join(BASE_DIR, 'saved_models', 'best_model_exp2_feat_sel_decision_tree.pkl'))
scaler = joblib.load(os.path.join(BASE_DIR, 'saved_models', 'scaler.pkl'))

AF = ['Plant_ID','Soil_Moisture','Ambient_Temperature','Soil_Temperature',
      'Humidity','Light_Intensity','Soil_pH','Nitrogen_Level',
      'Phosphorus_Level','Potassium_Level','Chlorophyll_Content','Electrochemical_Signal']
SI = [AF.index('Soil_Moisture'), AF.index('Nitrogen_Level')]
L = {0:'Healthy', 1:'Moderate Stress', 2:'High Stress'}

df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'plant_health_data.csv'))
df['label'] = df['Plant_Health_Status'].map({'High Stress':2,'Moderate Stress':1,'Healthy':0})
dw = df.drop(['Timestamp','Plant_Health_Status','Plant_Name'], axis=1)
X = dw.drop(columns=['label'])
y = dw['label']
Xs = scaler.transform(X)
_, Xt, _, yt = train_test_split(Xs, y, test_size=0.2, random_state=42, stratify=y)
Xu = scaler.inverse_transform(Xt)
yp_exact = model.predict(Xt[:, SI])
acc_exact = accuracy_score(yt, yp_exact)

def predict_one(sv, nv):
    inp = np.zeros((1, 12))
    inp[0, AF.index('Soil_Moisture')] = sv
    inp[0, AF.index('Nitrogen_Level')] = nv
    scaled = scaler.transform(inp)
    return int(model.predict(scaled[:, SI])[0])

def test_config(soil_thresholds, soil_reps, nit_thresholds, nit_reps, soil_names, nit_names, config_name):
    """Test a given level configuration"""
    def classify(val, thresholds):
        for i, t in enumerate(thresholds):
            if val <= t:
                return i
        return len(thresholds)
    
    yp = []
    for i in range(len(Xt)):
        raw_sm = Xu[i, AF.index('Soil_Moisture')]
        raw_nl = Xu[i, AF.index('Nitrogen_Level')]
        sm_lv = classify(raw_sm, soil_thresholds)
        nl_lv = classify(raw_nl, nit_thresholds)
        yp.append(predict_one(soil_reps[sm_lv], nit_reps[nl_lv]))
    
    acc = accuracy_score(yt, yp)
    report = classification_report(yt, yp, target_names=list(L.values()))
    
    # 3x3 table
    table_lines = []
    for si, (sn, sv) in enumerate(zip(soil_names, soil_reps)):
        for ni, (nn, nv) in enumerate(zip(nit_names, nit_reps)):
            pred = predict_one(sv, nv)
            table_lines.append(f"    {sn:>25s} x {nn:>20s}  (SM={sv:.2f}, NL={nv:.2f})  ->  {L[pred]}")
    
    return acc, report, table_lines

# ============================================
# Config A: 3x3 (percentile-based)
# ============================================
sm_p = [df.Soil_Moisture.quantile(q) for q in [0.33, 0.67]]
nl_p = [df.Nitrogen_Level.quantile(q) for q in [0.33, 0.67]]

sm_reps_3 = [
    df[df.Soil_Moisture <= sm_p[0]].Soil_Moisture.median(),
    df[(df.Soil_Moisture > sm_p[0]) & (df.Soil_Moisture <= sm_p[1])].Soil_Moisture.median(),
    df[df.Soil_Moisture > sm_p[1]].Soil_Moisture.median()
]
nl_reps_3 = [
    df[df.Nitrogen_Level <= nl_p[0]].Nitrogen_Level.median(),
    df[(df.Nitrogen_Level > nl_p[0]) & (df.Nitrogen_Level <= nl_p[1])].Nitrogen_Level.median(),
    df[df.Nitrogen_Level > nl_p[1]].Nitrogen_Level.median()
]

# ============================================
# Config B: 5 soil x 3 nitrogen
# ============================================
sm_p5 = [df.Soil_Moisture.quantile(q) for q in [0.20, 0.40, 0.60, 0.80]]
sm_reps_5 = []
bins = [(-999, sm_p5[0]), (sm_p5[0], sm_p5[1]), (sm_p5[1], sm_p5[2]), (sm_p5[2], sm_p5[3]), (sm_p5[3], 999)]
for lo, hi in bins:
    subset = df[(df.Soil_Moisture > lo) & (df.Soil_Moisture <= hi)].Soil_Moisture
    sm_reps_5.append(subset.median())

# ============================================
# Config C: Class-mean based (3x3)
# ============================================
sm_cls = [df[df.label==k].Soil_Moisture.mean() for k in [2,1,0]]
nl_cls = [df[df.label==k].Nitrogen_Level.mean() for k in [2,1,0]]

# class-boundary thresholds (midpoints between class means)
sm_cls_t = [(sm_cls[0]+sm_cls[1])/2, (sm_cls[1]+sm_cls[2])/2]
nl_cls_t = [(nl_cls[0]+nl_cls[1])/2, (nl_cls[1]+nl_cls[2])/2]

# ============================================
# Config D: 5 soil x 5 nitrogen (fine-grained)
# ============================================
nl_p5 = [df.Nitrogen_Level.quantile(q) for q in [0.20, 0.40, 0.60, 0.80]]
nl_reps_5 = []
bins_nl = [(-999, nl_p5[0]), (nl_p5[0], nl_p5[1]), (nl_p5[1], nl_p5[2]), (nl_p5[2], nl_p5[3]), (nl_p5[3], 999)]
for lo, hi in bins_nl:
    subset = df[(df.Nitrogen_Level > lo) & (df.Nitrogen_Level <= hi)].Nitrogen_Level
    nl_reps_5.append(subset.median())

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'analysis_output.txt')
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(f"Exact values accuracy: {acc_exact:.4f}\n\n")
    
    configs = [
        ("Config A: 3 Soil x 3 Nitrogen (Percentile P33/P67)",
         sm_p, sm_reps_3, nl_p, nl_reps_3,
         ['Dry', 'Moist', 'Wet'],
         ['Yellow/Pale', 'Normal Green', 'Dark Green']),
        ("Config B: 5 Soil x 3 Nitrogen", 
         sm_p5, sm_reps_5, nl_p, nl_reps_3,
         ['Very Dry', 'Dry', 'Moist', 'Slightly Wet', 'Wet'],
         ['Yellow/Pale', 'Normal Green', 'Dark Green']),
        ("Config C: 3 Soil x 3 Nitrogen (Class-mean based)",
         sm_cls_t, sm_cls, nl_cls_t, nl_cls,
         ['Dry', 'Moist', 'Wet'],
         ['Yellow/Pale', 'Normal Green', 'Dark Green']),
        ("Config D: 5 Soil x 5 Nitrogen (Fine-grained)",
         sm_p5, sm_reps_5, nl_p5, nl_reps_5,
         ['Very Dry', 'Dry', 'Moist', 'Slightly Wet', 'Wet'],
         ['Very Yellow', 'Light Yellow', 'Normal Green', 'Green', 'Dark Green']),
    ]
    
    for name, st, sr, nt, nr, snames, nnames in configs:
        acc, report, table = test_config(st, sr, nt, nr, snames, nnames, name)
        f.write(f"{'='*70}\n")
        f.write(f"  {name}\n")
        f.write(f"{'='*70}\n\n")
        f.write(f"  Soil thresholds:     {[f'{t:.2f}' for t in st]}\n")
        f.write(f"  Soil reps:           {[f'{r:.2f}' for r in sr]}\n")
        f.write(f"  Nitrogen thresholds: {[f'{t:.2f}' for t in nt]}\n")
        f.write(f"  Nitrogen reps:       {[f'{r:.2f}' for r in nr]}\n\n")
        f.write(f"  Accuracy: {acc:.4f}  (drop from exact: {acc_exact - acc:.4f})\n\n")
        f.write(f"  Prediction Table:\n")
        for line in table:
            f.write(line + "\n")
        f.write(f"\n  Classification Report:\n{report}\n\n")
    
print(f"Done! Results in {out_path}")
