import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(
    page_title="MediCore AI | Breast Cancer Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_resource
def load_model():
    data = load_breast_cancer(as_frame=True)
    X = data.frame[list(data.feature_names)]
    y = (data.frame["target"] == 0).astype(int)  # 1 = malignant, 0 = benign

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    q1, q3 = X_train.quantile(0.25), X_train.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train.clip(lower, upper, axis=1))
    X_test_s = scaler.transform(X_test.clip(lower, upper, axis=1))

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_s, y_train)

    pred = model.predict(X_test_s)
    metrics = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1-score": f1_score(y_test, pred),
    }
    return model, scaler, lower, upper, X, metrics


def predict(model, scaler, lower, upper, values):
    row = pd.DataFrame([values], columns=lower.index).clip(lower, upper, axis=1)
    return float(model.predict_proba(scaler.transform(row))[0][1])


model, scaler, lower, upper, X, metrics = load_model()

# -------------------- VISUAL SYSTEM --------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root { --bg:#050b12; --panel:#0b1722; --panel2:#0d1d29; --line:rgba(93,220,221,.14); --mint:#00d69a; --cyan:#25d9e8; --muted:#8ba2b3; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background: radial-gradient(ellipse at 78% 0%, rgba(0,196,177,.12), transparent 35%), radial-gradient(ellipse at 5% 35%, rgba(0,111,180,.10), transparent 34%), var(--bg); color:#edf7fb; }
[data-testid="stHeader"] { background:rgba(5,11,18,.78); }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#07131d,#08101a 75%); border-right:1px solid var(--line); }
[data-testid="stSidebar"] > div { padding-top:1.2rem; }
.block-container { padding-top:1.4rem; padding-bottom:3rem; max-width:1500px; }
h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; color:#f3fbff !important; letter-spacing:-.6px; }
p, label, .stCaption { color:#9bb0bf; }
div[data-testid="stMetric"] { background:linear-gradient(145deg,rgba(15,35,48,.96),rgba(8,20,30,.96)); border:1px solid var(--line); border-radius:17px; padding:17px 18px; box-shadow:0 10px 35px rgba(0,0,0,.16); }
div[data-testid="stMetric"] label { color:#89a4b4 !important; font-size:.78rem !important; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color:#f4ffff !important; font-family:'Space Grotesk'; }
.stButton > button, .stFormSubmitButton > button { background:linear-gradient(100deg,#00bd82,#00d8ad) !important; color:#03130f !important; border:0 !important; border-radius:10px !important; font-weight:800 !important; min-height:46px; box-shadow:0 0 22px rgba(0,214,154,.22); }
.stButton > button:hover, .stFormSubmitButton > button:hover { box-shadow:0 0 28px rgba(0,214,154,.42); transform:translateY(-1px); }
div[data-baseweb="input"] > div, div[data-baseweb="select"] > div { background:#0a1823 !important; border:1px solid rgba(112,178,194,.22) !important; border-radius:9px !important; color:#e9fbff !important; }
input { color:#e9fbff !important; }
[data-testid="stExpander"] { background:rgba(10,25,37,.72); border:1px solid var(--line); border-radius:13px; }
hr { border-color:rgba(104,173,190,.13); }
section[data-testid="stSidebar"] .stRadio label { color:#c5d8e3 !important; }
section[data-testid="stSidebar"] .stRadio [role="radiogroup"] { gap:7px; }
div[data-testid="stProgressBar"] > div > div { background:linear-gradient(90deg,#00b987,#29dce5); }
.hero { background:linear-gradient(112deg,rgba(8,28,39,.97),rgba(5,17,27,.88) 55%,rgba(0,105,105,.18)); border:1px solid rgba(48,214,205,.22); border-radius:22px; padding:30px 32px; position:relative; overflow:hidden; box-shadow:0 18px 60px rgba(0,0,0,.24); }
.hero:after { content:''; position:absolute; width:360px;height:360px;right:-95px;top:-160px;border-radius:50%;background:radial-gradient(circle,rgba(0,221,190,.20),transparent 68%); }
.eyebrow { color:#00d6a0; font-size:11px; letter-spacing:2.1px; font-weight:800; text-transform:uppercase; }
.hero h1 { font-size:clamp(2rem,4vw,3.25rem) !important; line-height:1.05; margin:12px 0 10px; }
.hero p { max-width:620px; font-size:1rem; line-height:1.65; color:#9bb3c0; margin-bottom:0; }
.glow { color:#00d69a; }
.panel { background:linear-gradient(145deg,rgba(13,31,44,.94),rgba(7,18,28,.96)); border:1px solid rgba(95,190,201,.16); border-radius:17px; padding:19px 20px; min-height:120px; }
.panel-kicker { color:#89a5b4; font-size:11px; text-transform:uppercase; letter-spacing:1.3px; }
.panel-value { font-family:'Space Grotesk'; font-size:27px; color:#f1fcff; font-weight:700; margin:8px 0 3px; }
.panel-note { color:#00d69a; font-size:12px; }
.section-title { font-family:'Space Grotesk'; font-size:21px; font-weight:700; color:#edfaff; margin:20px 0 12px; }
.small-note { color:#8098a8; font-size:12px; }
.result-card { border-radius:18px; padding:23px; background:linear-gradient(140deg,rgba(10,40,49,.98),rgba(7,21,31,.98)); border:1px solid rgba(0,214,154,.30); }
.result-label { color:#83a6b2; text-transform:uppercase; letter-spacing:1.4px; font-size:11px; }
.result-value { font-family:'Space Grotesk'; font-size:32px; font-weight:700; margin:7px 0; }
div[data-testid="stPlotlyChart"], div[data-testid="stVegaLiteChart"] { background:transparent; }
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin:4px 0 28px 4px">
      <div style="width:38px;height:38px;border-radius:12px;background:linear-gradient(135deg,#00d69a,#18b6d0);display:grid;place-items:center;color:#032019;font-size:21px;font-weight:900">✚</div>
      <div><div style="font-family:Space Grotesk;font-size:18px;font-weight:700;color:#efffff">MediCore <span style="color:#00d69a">AI</span></div><div style="font-size:10px;color:#7894a4;letter-spacing:1.5px">DIAGNOSTIC INTELLIGENCE</div></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div class='small-note' style='margin:0 0 9px 7px'>WORKSPACE</div>", unsafe_allow_html=True)
    page = st.radio(
        "Navigation",
        ["Overview", "AI Prediction", "Model Analytics"],
        label_visibility="collapsed",
        format_func=lambda x: {"Overview":"◈   Overview","AI Prediction":"⌁   AI Prediction","Model Analytics":"▥   Model Analytics"}[x],
    )
    st.markdown("---")
    st.markdown("<div class='small-note'>SYSTEM STATUS</div>", unsafe_allow_html=True)
    st.markdown("<div style='margin-top:10px;padding:12px;border:1px solid rgba(0,214,154,.2);border-radius:12px;background:rgba(0,214,154,.05)'><span style='color:#00d69a'>●</span> <span style='color:#cde5e9;font-size:13px'>Prediction engine ready</span><div class='small-note' style='margin:5px 0 0 15px'>Logistic Regression · v1.0</div></div>", unsafe_allow_html=True)
    st.markdown("<div style='height:34px'></div><div class='small-note'>EDUCATIONAL PROJECT</div><div style='font-size:12px;color:#6f8797;margin-top:5px'>Breast Cancer Wisconsin<br>Diagnostic Dataset</div>", unsafe_allow_html=True)

# Header
left, right = st.columns([5, 1])
with left:
    st.markdown("<div class='eyebrow'>MEDICAL AI PLATFORM&nbsp;&nbsp; / &nbsp;&nbsp; BREAST CANCER ANALYSIS</div>", unsafe_allow_html=True)
with right:
    st.markdown("<div style='text-align:right;color:#7f9baa;font-size:12px;padding-top:4px'>● SYSTEM ONLINE</div>", unsafe_allow_html=True)

if page == "Overview":
    st.markdown("""
    <div class="hero">
      <div class="eyebrow">INTELLIGENCE THAT SUPPORTS BETTER DECISIONS</div>
      <h1>Smarter Analysis.<br>Faster <span class="glow">Insights.</span></h1>
      <p>Explore an AI-powered breast tumour classification demo. Review model performance, enter tumour measurements, and view a probability-based prediction in one workspace.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Platform overview</div>", unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    cards = [
        (c1,"DATA RECORDS","569","Wisconsin diagnostic samples"),
        (c2,"INPUT FEATURES","30","Tumour measurement variables"),
        (c3,"TEST ACCURACY",f"{metrics['Accuracy']*100:.1f}%","Held-out test set"),
        (c4,"AI MODEL","LR","Logistic Regression"),
    ]
    for col, title, value, note in cards:
        with col:
            st.markdown(f"<div class='panel'><div class='panel-kicker'>{title}</div><div class='panel-value'>{value}</div><div class='panel-note'>{note}</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Clinical intelligence workspace</div>", unsafe_allow_html=True)
    a,b = st.columns([1.25,1])
    with a:
        st.markdown("<div class='panel' style='min-height:245px'><div class='panel-kicker'>AI DIAGNOSTICS</div><div style='font-family:Space Grotesk;font-size:23px;font-weight:700;margin:12px 0;color:#effbff'>Tumour pattern classification</div><div style='color:#91aab9;line-height:1.7;font-size:14px'>The model analyses 30 numeric measurements and estimates the probability of a malignant class. Inputs are processed using the same preprocessing pipeline used during training.</div><div style='display:flex;gap:9px;flex-wrap:wrap;margin-top:22px'><span style='border:1px solid rgba(0,214,154,.25);padding:7px 10px;border-radius:8px;color:#00d69a;font-size:11px'>● IQR outlier capping</span><span style='border:1px solid rgba(37,217,232,.25);padding:7px 10px;border-radius:8px;color:#38d9e8;font-size:11px'>● StandardScaler</span><span style='border:1px solid rgba(37,217,232,.25);padding:7px 10px;border-radius:8px;color:#38d9e8;font-size:11px'>● Logistic Regression</span></div></div>", unsafe_allow_html=True)
    with b:
        st.markdown("<div class='panel' style='min-height:245px'><div class='panel-kicker'>MODEL PERFORMANCE SNAPSHOT</div>", unsafe_allow_html=True)
        for name, val in metrics.items():
            st.markdown(f"<div style='display:flex;justify-content:space-between;margin:13px 0 5px;color:#b8cbd4;font-size:12px'><span>{name}</span><b style='color:#eaffff'>{val*100:.1f}%</b></div>", unsafe_allow_html=True)
            st.progress(float(val))
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div class='small-note' style='margin-top:18px'>Use the AI Prediction page to enter measurements. This project is an educational demonstration, not a clinical diagnostic system.</div>", unsafe_allow_html=True)

elif page == "AI Prediction":
    st.markdown("<div class='hero'><div class='eyebrow'>AI-POWERED CLASSIFICATION</div><h1>Patient <span class='glow'>Input.</span><br>Model Insights.</h1><p>Enter the 30 tumour measurements below. Dataset median values are prefilled as a starting point; change them to explore how the model responds.</p></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Tumour measurement input</div>", unsafe_allow_html=True)
    st.markdown("<div class='small-note'>All fields are numeric. Measurements are grouped into mean, standard error and worst-value features.</div>", unsafe_allow_html=True)

    groups = [
        ("MEAN FEATURES", "mean ", "Average measurement values"),
        ("STANDARD ERROR FEATURES", " error", "Variation / standard error values"),
        ("WORST FEATURES", "worst ", "Largest measurement values"),
    ]
    values = {}
    with st.form("cancer_form"):
        for title, key, description in groups:
            cols_in_group = [c for c in X.columns if (c.startswith(key) if key.endswith(" ") else c.endswith(key))]
            with st.expander(f"{title}  ·  {len(cols_in_group)} FEATURES", expanded=(title == "MEAN FEATURES")):
                st.caption(description)
                input_cols = st.columns(3)
                for idx, feature in enumerate(cols_in_group):
                    with input_cols[idx % 3]:
                        values[feature] = st.number_input(
                            feature.replace(" ", " ").title(),
                            min_value=0.0,
                            value=float(X[feature].median()),
                            format="%.5f",
                            key=f"input_{feature}",
                        )
        submitted = st.form_submit_button("RUN AI ANALYSIS   →", use_container_width=True)

    if submitted:
        p_mal = predict(model, scaler, lower, upper, values)
        p_ben = 1 - p_mal
        st.markdown("<div class='section-title'>Analysis result</div>", unsafe_allow_html=True)
        r1,r2 = st.columns([1.05,1])
        with r1:
            if p_mal >= 0.5:
                label, accent, desc = "Malignant pattern", "#ff8b72", "The model assigns a higher probability to the malignant class."
            else:
                label, accent, desc = "Benign pattern", "#00d69a", "The model assigns a higher probability to the benign class."
            st.markdown(f"<div class='result-card'><div class='result-label'>MODEL OUTPUT</div><div class='result-value' style='color:{accent}'>{label}</div><div style='color:#9bb3c0;font-size:13px;line-height:1.6'>{desc}</div><div style='margin-top:19px;color:#e9faff;font-size:13px'>Class threshold <b>50%</b></div></div>", unsafe_allow_html=True)
        with r2:
            st.markdown("<div class='panel' style='min-height:190px'><div class='panel-kicker'>CLASS PROBABILITY</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='display:flex;justify-content:space-between;margin:15px 0 6px;font-size:13px;color:#d6e8ed'><span>Benign</span><b>{p_ben*100:.1f}%</b></div>", unsafe_allow_html=True)
            st.progress(float(p_ben))
            st.markdown(f"<div style='display:flex;justify-content:space-between;margin:17px 0 6px;font-size:13px;color:#d6e8ed'><span>Malignant</span><b>{p_mal*100:.1f}%</b></div>", unsafe_allow_html=True)
            st.progress(float(p_mal))
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<div class='small-note' style='margin-top:14px'>Probability values are model outputs, not a medical diagnosis or a substitute for professional evaluation.</div>", unsafe_allow_html=True)
    else:
        st.info("Enter measurements and select **RUN AI ANALYSIS** to view the model output.")

elif page == "Model Analytics":
    st.markdown("<div class='hero'><div class='eyebrow'>MODEL TRANSPARENCY</div><h1>Performance <span class='glow'>Analytics.</span></h1><p>Evaluation metrics calculated on the held-out test set after an 80:20 stratified train-test split.</p></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Evaluation metrics</div>", unsafe_allow_html=True)
    cols = st.columns(4)
    for col, (name, val) in zip(cols, metrics.items()):
        with col:
            st.metric(name, f"{val*100:.2f}%")
    st.markdown("<div class='section-title'>Metric comparison</div>", unsafe_allow_html=True)
    chart_df = pd.DataFrame({"Score (%)":[v*100 for v in metrics.values()]}, index=list(metrics.keys()))
    st.bar_chart(chart_df, color="#00c99a", height=280)
    st.markdown("<div class='section-title'>Model workflow</div>", unsafe_allow_html=True)
    w1,w2,w3,w4 = st.columns(4)
    workflow = [
        (w1,"01","DATASET","569 records · 30 features"),
        (w2,"02","PREPROCESSING","IQR capping · scaling"),
        (w3,"03","TRAINING","Logistic Regression"),
        (w4,"04","EVALUATION","Stratified 80:20 split"),
    ]
    for col, num, name, detail in workflow:
        with col:
            st.markdown(f"<div class='panel'><div class='eyebrow'>{num}</div><div style='font-family:Space Grotesk;font-weight:700;color:#effaff;margin:10px 0'>{name}</div><div class='small-note'>{detail}</div></div>", unsafe_allow_html=True)

st.markdown("<hr><div style='display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;color:#627b8a;font-size:11px'><span>MEDICORE AI · BREAST CANCER CLASSIFICATION DEMO</span><span>FOR EDUCATIONAL USE ONLY · NOT FOR CLINICAL DECISIONS</span></div>", unsafe_allow_html=True)
