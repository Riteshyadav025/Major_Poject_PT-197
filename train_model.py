python -m streamlit run app.pyfrom pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'cardio_train.csv'
FEATURES = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']

if not DATA.exists():
    raise SystemExit('Dataset missing: put Kaggle cardio_train.csv inside data/ and run again.')

df = pd.read_csv(DATA, sep=None, engine='python')
df.columns = df.columns.str.strip()
required = FEATURES + ['cardio']
missing = sorted(set(required) - set(df.columns))
if missing:
    raise SystemExit(f'Missing dataset columns: {missing}. Use the Kaggle Cardiovascular Disease dataset.')
df = df[required].copy()
for c in required:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df = df.dropna().drop_duplicates()
df = df[(df.age.between(365 * 18, 365 * 100)) & (df.height.between(100, 230)) & (df.weight.between(30, 250)) & (df.ap_hi.between(70, 250)) & (df.ap_lo.between(40, 150)) & (df.ap_hi > df.ap_lo)]
for c, valid in {'gender':[1,2], 'cholesterol':[1,2,3], 'gluc':[1,2,3], 'smoke':[0,1], 'alco':[0,1], 'active':[0,1], 'cardio':[0,1]}.items():
    df = df[df[c].isin(valid)]
if len(df) < 100 or df.cardio.nunique() != 2:
    raise SystemExit('Insufficient valid data after cleaning; check dataset.')
df['age'] = df['age'] / 365.25
X, y = df[FEATURES], df['cardio'].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = Pipeline([('scaler', StandardScaler()), ('ann', MLPClassifier(hidden_layer_sizes=(32,16), max_iter=250, early_stopping=True, random_state=42))])
model.fit(X_train, y_train)
pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]
metrics = {'records_after_cleaning':int(len(df)), 'test_records':int(len(y_test)), 'accuracy':float(accuracy_score(y_test,pred)), 'precision':float(precision_score(y_test,pred,zero_division=0)), 'recall':float(recall_score(y_test,pred,zero_division=0)), 'f1':float(f1_score(y_test,pred,zero_division=0)), 'roc_auc':float(roc_auc_score(y_test,prob)), 'confusion_matrix':confusion_matrix(y_test,pred).tolist()}
joblib.dump(model, ROOT / 'model.joblib')
(ROOT / 'metrics.json').write_text(json.dumps(metrics, indent=2))
print('Training complete!')
print(json.dumps(metrics, indent=2))
