from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
FEATURES = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
st.set_page_config(page_title='CardioPredict | Major Project', page_icon='❤️', layout='wide')
st.title('❤️ CardioPredict')
st.subheader('Predictive Analytic Health Monitoring Using Machine Learning')
st.caption('IILM University | B.Tech Major Project Phase-I | ANN Prototype')
st.warning('Academic demonstration only. This is not a clinical risk score, diagnosis, or medical advice.')
if not (ROOT / 'model.joblib').exists():
    st.error('Model not found. First download the dataset and run: python train_model.py')
    st.stop()
model = joblib.load(ROOT / 'model.joblib')
with st.form('patient'):
    a,b,c = st.columns(3)
    with a:
        age = st.number_input('Age (years)', 18, 100, 45)
        gender = st.selectbox('Gender (dataset encoding)', [1,2], format_func=lambda x: '1 — Female' if x == 1 else '2 — Male')
        height = st.number_input('Height (cm)', 100, 230, 170)
        weight = st.number_input('Weight (kg)', 30, 250, 70)
    with b:
        ap_hi = st.number_input('Systolic BP (mmHg)', 70, 250, 120)
        ap_lo = st.number_input('Diastolic BP (mmHg)', 40, 150, 80)
        cholesterol = st.selectbox('Cholesterol category', [1,2,3], format_func=lambda x: {1:'Normal',2:'Above normal',3:'Well above normal'}[x])
        gluc = st.selectbox('Glucose category', [1,2,3], format_func=lambda x: {1:'Normal',2:'Above normal',3:'Well above normal'}[x])
    with c:
        smoke = st.selectbox('Smoking', [0,1], format_func=lambda x: 'Yes' if x else 'No')
        alco = st.selectbox('Alcohol intake', [0,1], format_func=lambda x: 'Yes' if x else 'No')
        active = st.selectbox('Physically active', [1,0], format_func=lambda x: 'Yes' if x else 'No')
    submit = st.form_submit_button('Predict cardiovascular disease class', type='primary')
if submit:
    if ap_hi <= ap_lo:
        st.error('Systolic blood pressure must be greater than diastolic blood pressure.')
    else:
        sample = pd.DataFrame([[age,gender,height,weight,ap_hi,ap_lo,cholesterol,gluc,smoke,alco,active]], columns=FEATURES)
        probability = float(model.predict_proba(sample)[0,1])
        predicted = int(model.predict(sample)[0])
        st.subheader('ANN prediction result')
        st.metric('Estimated probability of dataset class 1', f'{probability:.1%}')
        if predicted:
            st.warning('Predicted class: 1 — Cardiovascular disease present (dataset label)')
        else:
            st.success('Predicted class: 0 — Cardiovascular disease absent (dataset label)')
        st.caption('The output is a dataset-based model estimate; it is not a validated personal clinical risk assessment.')
if (ROOT / 'metrics.json').exists():
    st.divider()
    st.subheader('Model evaluation on held-out test data')
    m = json.loads((ROOT / 'metrics.json').read_text())
    cols = st.columns(5)
    for col, key in zip(cols, ['accuracy','precision','recall','f1','roc_auc']):
        col.metric(key.upper().replace('_','-'), f'{m[key]:.3f}')
    cm = m['confusion_matrix']
    st.write('Confusion matrix (rows = actual class, columns = predicted class)')
    st.dataframe(pd.DataFrame(cm, index=['Actual 0','Actual 1'], columns=['Predicted 0','Predicted 1']))
    st.caption(f"Training dataset after cleaning: {m['records_after_cleaning']:,} rows; test set: {m['test_records']:,} rows.")
