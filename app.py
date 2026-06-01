"""
Customer Churn Predictor — Demo ⑦
Upload customer data → K-means segmentation + XGBoost churn scoring + SHAP explainability.

Business impact: Identify high-risk customers before they leave, prioritize retention spend.
"""

import os, io, json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from openai import OpenAI
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import shap

st.set_page_config(page_title="Customer Churn Predictor", page_icon="🎯", layout="wide")

st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background:var(--background-color); }
  [data-testid="stSidebar"] { background:var(--secondary-background-color); border-right:1px solid rgba(120,130,150,0.35); }
  [data-testid="stAppViewContainer"] .main .block-container { max-width:1220px; padding-top:1.2rem; }
  [data-testid="stAppViewContainer"], [data-testid="stSidebar"] { font-size:16px; }
  p, label, [data-testid="stMarkdownContainer"] p { font-size:0.95rem; }
  .section-tag {
    display:inline-block;background:var(--secondary-background-color);color:var(--text-color) !important;
    border:1px solid rgba(120,130,150,0.35);font-size:0.76rem;font-weight:700;letter-spacing:0.1em;
    text-transform:uppercase;padding:0.3rem 0.8rem;border-radius:4px;margin-bottom:1rem;
  }
  .stButton>button {
    background:var(--secondary-background-color);border:1px solid rgba(120,130,150,0.45);
    color:var(--text-color) !important;border-radius:8px;min-height:42px;font-weight:600;
  }
  .stButton>button:hover { border-color:var(--primary-color); }
  .stButton>button[data-testid="baseButton-primary"] { background:#4f46e5 !important;border-color:#4f46e5 !important;color:white !important; }
  .stButton>button[kind="primary"] { background:#4f46e5 !important;border-color:#4f46e5 !important;color:white !important; }
  [data-testid="stDataFrame"] { border:1px solid rgba(120,130,150,0.3);border-radius:8px; }
  [data-testid="stMetric"] { background:var(--secondary-background-color);border-radius:8px;padding:0.8rem 1rem; }
  [data-testid="stFileUploader"] {
    border:2px dashed rgba(120,130,150,0.45) !important;border-radius:10px !important;
    padding:1.5rem !important;background:var(--secondary-background-color) !important;
  }
  [data-testid="stFileUploaderDropzone"] { background:var(--secondary-background-color) !important;border:0 !important; }
  [data-testid="stFileUploaderDropzone"] * { color:var(--text-color) !important; }
  [data-testid="stDownloadButton"]>button {
    background:var(--secondary-background-color) !important;border:1px solid rgba(120,130,150,0.45) !important;
    color:var(--text-color) !important;border-radius:8px !important;min-height:42px;font-weight:600;
  }
  .privacy-box {
    background:var(--secondary-background-color);border:1px solid rgba(120,130,150,0.45);border-radius:8px;
    padding:0.7rem 1rem;color:var(--text-color) !important;font-size:0.83rem;line-height:1.7;margin-bottom:1rem;
  }
</style>
""", unsafe_allow_html=True)

COLORS = ["#6366f1","#06b6d4","#34d399","#f59e0b","#f43f5e","#a78bfa"]

# ── Sample data ────────────────────────────────────────────────────────────────
def generate_sample(n=800, seed=42):
    np.random.seed(seed)
    tenure      = np.random.exponential(18, n).clip(1, 60).astype(int)
    orders      = (tenure * np.random.uniform(0.3, 1.2, n)).clip(1).astype(int)
    avg_order   = np.random.normal(85, 30, n).clip(10).round(2)
    days_since  = np.random.exponential(30, n).clip(1, 365).astype(int)
    support     = np.random.poisson(1.2, n)
    returns     = np.random.binomial(orders, 0.08)
    # Churn probability depends on recency + support tickets
    churn_prob  = (days_since / 365) * 0.6 + (support / 10) * 0.2 + (returns / (orders + 1)) * 0.2
    churn_prob  = churn_prob.clip(0, 1)
    churned     = np.random.binomial(1, churn_prob)
    return pd.DataFrame({
        "customer_id": range(1, n+1),
        "tenure_months": tenure,
        "total_orders": orders,
        "avg_order_value": avg_order,
        "days_since_last_order": days_since,
        "support_tickets": support,
        "return_count": returns,
        "churned": churned,
    })

SAMPLE_DF = generate_sample()
SAMPLE_CSV = SAMPLE_DF.to_csv(index=False)

FEATURE_COLS = ["tenure_months","total_orders","avg_order_value",
                "days_since_last_order","support_tickets","return_count"]

# ── ML pipeline ────────────────────────────────────────────────────────────────
def get_client():
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        st.error("⚠️ API key not configured.")
        st.stop()
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

def run_segmentation(df: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
    feats = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feats].fillna(df[feats].median())
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df = df.copy()
    df["segment"] = km.fit_predict(Xs).astype(str)
    return df

def run_churn_model(df: pd.DataFrame):
    feats = [c for c in FEATURE_COLS if c in df.columns]
    if "churned" not in df.columns:
        return df, None, None, None
    X = df[feats].fillna(df[feats].median())
    y = df["churned"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                          use_label_encoder=False, eval_metric="logloss", random_state=42)
    model.fit(X_tr, y_tr)
    auc = roc_auc_score(y_te, model.predict_proba(X_te)[:,1])
    proba = model.predict_proba(X)[:,1]
    df = df.copy()
    df["churn_prob"] = proba.round(3)
    df["risk"] = pd.cut(proba, bins=[0, 0.3, 0.6, 1.0],
                        labels=["Low", "Medium", "High"])
    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)
    return df, model, auc, (shap_vals, X, feats)

def ai_recommendation(client, summary: dict) -> str:
    system = (
        "You are a senior CRM analyst. Given customer churn model results, "
        "write a 4-6 sentence executive summary covering: "
        "overall churn risk distribution, top churn drivers from SHAP analysis, "
        "which customer segments need immediate attention, "
        "and 2-3 specific retention actions. Be direct and quantitative."
    )
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role":"system","content":system},
                  {"role":"user","content":json.dumps(summary, indent=2)}],
        temperature=0.3, max_tokens=400,
    )
    return resp.choices[0].message.content.strip()

# ── Charts ─────────────────────────────────────────────────────────────────────
def plot_risk_dist(df):
    counts = df["risk"].value_counts().reindex(["High","Medium","Low"]).fillna(0)
    colors_map = {"High":"#f43f5e","Medium":"#f59e0b","Low":"#34d399"}
    fig = go.Figure(go.Bar(
        x=counts.index, y=counts.values,
        marker_color=[colors_map[c] for c in counts.index],
        text=counts.values.astype(int), textposition="outside",
    ))
    fig.update_layout(title="Customer Risk Distribution", height=300,
                      xaxis_title="Risk Level", yaxis_title="Customers",
                      showlegend=False, margin=dict(t=50,b=30))
    return fig

def plot_shap_bar(shap_vals, feats):
    mean_abs = np.abs(shap_vals).mean(axis=0)
    idx = np.argsort(mean_abs)[::-1]
    fig = go.Figure(go.Bar(
        x=mean_abs[idx], y=[feats[i] for i in idx],
        orientation="h",
        marker_color="#6366f1",
    ))
    fig.update_layout(title="Feature Importance (SHAP)", height=320,
                      xaxis_title="Mean |SHAP value|",
                      yaxis=dict(autorange="reversed"),
                      margin=dict(t=50,b=30,l=160))
    return fig

def plot_segment_churn(df):
    seg = df.groupby("segment").agg(
        customers=("customer_id","count"),
        churn_rate=("churned","mean"),
        avg_churn_prob=("churn_prob","mean"),
    ).reset_index()
    fig = px.bar(seg, x="segment", y="churn_rate",
                 color="avg_churn_prob",
                 color_continuous_scale=["#34d399","#f59e0b","#f43f5e"],
                 text=seg["churn_rate"].round(2),
                 title="Churn Rate by Segment",
                 labels={"churn_rate":"Churn Rate","segment":"Segment"})
    fig.update_layout(height=300, margin=dict(t=50,b=30), showlegend=False)
    fig.update_traces(texttemplate="%{text:.0%}", textposition="outside")
    return fig

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎯 Churn Predictor")
    st.divider()
    st.markdown("""
<div style="background:var(--secondary-background-color);border:1px solid rgba(120,130,150,0.35);
border-radius:8px;padding:0.8rem 1rem;font-size:0.83rem;line-height:1.75">
<b>Demo storyline</b><br>
1) Start with risk distribution — how many high-risk customers?<br>
2) SHAP chart — what drives churn in this dataset?<br>
3) Segment view — which group needs immediate action?<br>
4) AI recommendation — translate findings into CRM actions<br><br>
<b>Suggested flow (2-4 min)</b>: risk dist → SHAP → segment → AI summary
</div>""", unsafe_allow_html=True)
    st.divider()
    st.markdown("**Required columns**")
    st.markdown("`customer_id` `churned` (0/1) + numeric feature columns")
    n_clusters = st.slider("Segments (K-means)", 2, 5, 3)
    st.divider()
    st.markdown("**Business impact**")
    st.markdown("Prioritize retention spend on high-risk, high-value customers before they leave.")
    st.divider()
    if st.button("Reset", use_container_width=True):
        for k in ["churn_sample","churn_result","churn_narrative"]:
            st.session_state.pop(k, None)
        st.rerun()
    st.divider()
    st.markdown("Built by [Joseph Wang](https://josephjwang.com)")

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='background:linear-gradient(90deg,#6366f1,#f43f5e);
-webkit-background-clip:text;-webkit-text-fill-color:transparent;
font-size:2.2rem;font-weight:700;margin-bottom:0.2rem'>🎯 Customer Churn Predictor</h1>
<p style='color:var(--text-color);opacity:0.7;font-size:1rem;margin-bottom:1.5rem'>
Upload customer data → K-means segmentation → XGBoost churn scoring → SHAP explainability</p>
""", unsafe_allow_html=True)

# ── Step 1: Data ───────────────────────────────────────────────────────────────
st.markdown('<span class="section-tag">Step 1 — Load data</span>', unsafe_allow_html=True)

mode = st.radio("Source", ["Use sample data", "Upload my own CSV"],
                horizontal=True, label_visibility="collapsed")
df = None

if mode == "Use sample data":
    st.session_state["churn_sample"] = True
    df = SAMPLE_DF.copy()
    st.info(f"Sample loaded: {len(df):,} customers · churn rate {df['churned'].mean():.1%}")
else:
    st.markdown("""<div class="privacy-box">
    🔒 <b>Your data stays private.</b> Processed in-memory only. Nothing stored or logged.
    </div>""", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            if "customer_id" not in df.columns:
                df.insert(0, "customer_id", range(1, len(df)+1))
            st.success(f"Loaded {len(df):,} rows · {len(df.columns)} columns")
        except Exception as e:
            st.error(f"Could not read file: {e}")

st.download_button("Download sample CSV", SAMPLE_CSV.encode(), "sample_customers.csv", "text/csv")

# ── Analysis ───────────────────────────────────────────────────────────────────
if df is not None:
    feats_available = [c for c in FEATURE_COLS if c in df.columns]
    if len(feats_available) < 2:
        st.error(f"Need at least 2 feature columns from: {FEATURE_COLS}")
        st.stop()

    with st.expander("Preview data"):
        st.dataframe(df.head(10), use_container_width=True, hide_index=True)

    # Run models
    with st.spinner("Running segmentation and churn model…"):
        df = run_segmentation(df, n_clusters)
        df, model, auc, shap_data = run_churn_model(df)

    # ── Step 2: KPIs ──────────────────────────────────────────────────────────
    st.divider()
    st.markdown('<span class="section-tag">Step 2 — Overview</span>', unsafe_allow_html=True)

    has_churn_col = "churned" in df.columns
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Customers", f"{len(df):,}")
    if has_churn_col:
        m2.metric("Churn Rate", f"{df['churned'].mean():.1%}")
    if "churn_prob" in df.columns:
        high_risk = (df["risk"] == "High").sum()
        m3.metric("High Risk", f"{high_risk:,}", delta=f"{high_risk/len(df):.1%} of base", delta_color="inverse")
        m4.metric("Avg Churn Prob", f"{df['churn_prob'].mean():.1%}")
    if auc:
        m5.metric("Model AUC", f"{auc:.3f}")

    # ── Step 3: Risk distribution ──────────────────────────────────────────────
    if "risk" in df.columns:
        st.divider()
        st.markdown('<span class="section-tag">Step 3 — Risk distribution</span>', unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(plot_risk_dist(df), use_container_width=True)
        with col_b:
            if shap_data:
                shap_vals, X_shap, feats = shap_data
                st.plotly_chart(plot_shap_bar(shap_vals, feats), use_container_width=True)

    # ── Step 4: Segments ───────────────────────────────────────────────────────
    st.divider()
    st.markdown('<span class="section-tag">Step 4 — Customer segments</span>', unsafe_allow_html=True)
    if has_churn_col and "churn_prob" in df.columns:
        st.plotly_chart(plot_segment_churn(df), use_container_width=True)

    seg_summary = df.groupby("segment").agg(
        customers=("customer_id","count"),
        avg_churn_prob=("churn_prob","mean") if "churn_prob" in df.columns else ("customer_id","count"),
        **{f: (f, "mean") for f in feats_available[:3]}
    ).round(2).reset_index()
    st.dataframe(seg_summary, use_container_width=True, hide_index=True)

    # ── Step 5: High-risk list ─────────────────────────────────────────────────
    if "churn_prob" in df.columns:
        st.divider()
        st.markdown('<span class="section-tag">Step 5 — High-risk customers</span>', unsafe_allow_html=True)
        high_risk_df = df[df["risk"]=="High"].sort_values("churn_prob", ascending=False).head(20)
        show_cols = ["customer_id","churn_prob","risk","segment"] + feats_available[:4]
        show_cols = [c for c in show_cols if c in high_risk_df.columns]
        st.dataframe(high_risk_df[show_cols], use_container_width=True, hide_index=True)

        csv_out = df[show_cols + [c for c in ["churned"] if c in df.columns]].to_csv(index=False).encode()
        st.download_button("Download scored customer list", csv_out, "churn_scores.csv", "text/csv")

    # ── Step 6: AI recommendation ─────────────────────────────────────────────
    st.divider()
    st.markdown('<span class="section-tag">Step 6 — AI retention recommendations</span>', unsafe_allow_html=True)

    if st.button("Generate AI Recommendations", type="primary"):
        top_features = []
        if shap_data:
            shap_vals, X_shap, feats = shap_data
            mean_abs = np.abs(shap_vals).mean(axis=0)
            idx = np.argsort(mean_abs)[::-1][:3]
            top_features = [feats[i] for i in idx]

        summary = {
            "total_customers": int(len(df)),
            "overall_churn_rate": round(float(df["churned"].mean()), 3) if has_churn_col else "unknown",
            "risk_distribution": {
                "high": int((df["risk"]=="High").sum()),
                "medium": int((df["risk"]=="Medium").sum()),
                "low": int((df["risk"]=="Low").sum()),
            } if "risk" in df.columns else {},
            "model_auc": round(float(auc), 3) if auc else None,
            "top_churn_drivers_shap": top_features,
            "avg_churn_prob": round(float(df["churn_prob"].mean()), 3) if "churn_prob" in df.columns else None,
            "n_segments": n_clusters,
        }
        with st.spinner("Generating retention strategy…"):
            client = get_client()
            rec = ai_recommendation(client, summary)
            st.session_state["churn_narrative"] = rec

    if st.session_state.get("churn_narrative"):
        st.markdown(
            f"<div style='background:var(--secondary-background-color);border-left:3px solid #6366f1;"
            f"border-radius:0 8px 8px 0;padding:1rem 1.2rem;line-height:1.8'>"
            f"{st.session_state['churn_narrative']}</div>",
            unsafe_allow_html=True,
        )
