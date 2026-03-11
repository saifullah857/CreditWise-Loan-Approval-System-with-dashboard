"""
╔══════════════════════════════════════════════════════════════╗
║       LOANIQ  ·  Loan Approval Intelligence Dashboard  v3   ║
║       Run:  streamlit run loan_dashboard.py                 ║
║       Deps: pip install streamlit plotly pandas scikit-learn numpy
╠══════════════════════════════════════════════════════════════╣
║  Fixes vs v1/v2                                              ║
║  1. LAYOUT_KW plain dict (go.Layout.to_plotly_json is str)  ║
║  2. hex_to_rgba() for valid Plotly fillcolor                 ║
║  3. @st.cache_resource for sklearn objects                   ║
║  4. OHE sparse kwarg compat (sklearn <1.2 / >=1.2)          ║
║  5. Dependents included + cast to float                      ║
║  6. Empty-filter guard on every KPI / chart                  ║
║  7. CM / heatmap arrays stored as .tolist() (JSON-safe)      ║
║  8. fillna(0) + float cast before scaler.transform          ║
║  9. data.csv not-found -> friendly st.error, no traceback    ║
║ 10. xaxis/yaxis NOT injected into polar / sunburst layouts   ║
║ 11. Loan_Term cast to float (model expects numeric)          ║
║ 12. Duplicate legend entries suppressed in box-plot grid     ║
╚══════════════════════════════════════════════════════════════╝
"""

import warnings
warnings.filterwarnings("ignore")

import inspect
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix,
)

# ══════════════════════════════════════════════
#  PAGE CONFIG  (must be the very first st call)
# ══════════════════════════════════════════════
st.set_page_config(
    page_title="LoanIQ · Approval Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════
#  GLOBAL CSS
# ══════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

:root{
  --bg:#070b14; --sf:#0d1624; --sf2:#111e30;
  --bd:#1c2d4a; --bd2:#243650;
  --a1:#00e5ff; --a2:#ff6b35; --a3:#7c3aed;
  --ok:#22d3a0; --er:#f43f5e;
  --tx:#e2e8f0; --mu:#64748b;
  --fh:'Syne',sans-serif; --fb:'DM Mono',monospace;
}

html,body,[class*="css"],.stApp{
  background-color:var(--bg)!important;
  color:var(--tx)!important;
  font-family:var(--fb);
}

section[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#080e1c 0%,#0c1628 100%)!important;
  border-right:1px solid var(--bd)!important;
}
section[data-testid="stSidebar"] *{color:var(--tx)!important;}

div[data-testid="metric-container"]{
  background:var(--sf); border:1px solid var(--bd);
  border-radius:14px; padding:18px 22px;
  position:relative; overflow:hidden;
  transition:transform .25s,box-shadow .25s;
}
div[data-testid="metric-container"]:hover{
  transform:translateY(-4px);
  box-shadow:0 10px 40px rgba(0,229,255,.14);
}
div[data-testid="metric-container"]::before{
  content:''; position:absolute; inset:0; pointer-events:none;
  background:linear-gradient(135deg,rgba(0,229,255,.05) 0%,transparent 55%);
}
div[data-testid="metric-container"] label{
  color:var(--mu)!important; font-family:var(--fb)!important;
  font-size:.7rem!important; letter-spacing:.08em!important; text-transform:uppercase;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"]{
  font-family:var(--fh)!important; font-size:1.65rem!important;
  font-weight:700!important; color:var(--tx)!important;
}
div[data-testid="metric-container"] [data-testid="stMetricDelta"]{
  font-size:.76rem!important; color:var(--ok)!important;
}

h1,h2,h3,h4{font-family:var(--fh)!important; letter-spacing:-.03em;}

div[data-baseweb="select"]>div{
  background:var(--sf)!important; border-color:var(--bd2)!important;
  color:var(--tx)!important; border-radius:8px!important;
}

.stButton>button,.stFormSubmitButton>button{
  background:linear-gradient(135deg,#00b8d4,#6d28d9)!important;
  color:#fff!important; font-family:var(--fh)!important; font-weight:700!important;
  font-size:.95rem!important; border:none!important; border-radius:10px!important;
  padding:12px 28px!important; letter-spacing:.02em;
  transition:opacity .2s,transform .2s!important;
}
.stButton>button:hover,.stFormSubmitButton>button:hover{
  opacity:.87!important; transform:translateY(-2px)!important;
}

hr{border-color:var(--bd)!important;}

.cc{
  background:var(--sf); border:1px solid var(--bd);
  border-radius:16px; padding:20px; margin-bottom:18px;
  transition:box-shadow .25s;
}
.cc:hover{box-shadow:0 4px 24px rgba(0,229,255,.07);}

.hero{
  background:linear-gradient(135deg,#0c1830 0%,#0a1020 45%,#10082a 100%);
  border:1px solid var(--bd); border-radius:22px;
  padding:40px 52px; margin-bottom:28px;
  position:relative; overflow:hidden;
}
.hero::before{
  content:''; position:absolute; left:-80px; top:-100px;
  width:350px; height:350px;
  background:radial-gradient(circle,rgba(0,229,255,.07) 0%,transparent 65%);
  pointer-events:none;
}
.hero::after{
  content:''; position:absolute; right:-60px; bottom:-80px;
  width:260px; height:260px;
  background:radial-gradient(circle,rgba(124,58,237,.10) 0%,transparent 65%);
  pointer-events:none;
}
.hero h1{
  font-size:2.5rem!important; font-weight:800;
  background:linear-gradient(100deg,#e2e8f0 25%,#00e5ff 80%,#a78bfa 100%);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
  margin-bottom:8px; line-height:1.15;
}
.hero p{color:var(--mu); font-size:.88rem; line-height:1.6; margin:0;}
.pill{
  display:inline-block;
  background:rgba(0,229,255,.08); border:1px solid rgba(0,229,255,.2);
  color:var(--a1); padding:3px 12px; border-radius:50px;
  font-size:.73rem; margin:10px 4px 0 0; font-family:var(--fb);
}

.badge-yes{
  background:rgba(34,211,160,.14); border:1px solid #22d3a0; color:#22d3a0;
  padding:7px 22px; border-radius:50px; font-weight:700;
  font-size:1.05rem; display:inline-block; letter-spacing:.04em;
}
.badge-no{
  background:rgba(244,63,94,.14); border:1px solid #f43f5e; color:#f43f5e;
  padding:7px 22px; border-radius:50px; font-weight:700;
  font-size:1.05rem; display:inline-block; letter-spacing:.04em;
}

.sec-lbl{
  color:var(--a1); font-family:var(--fh); font-size:.68rem; font-weight:700;
  letter-spacing:.15em; text-transform:uppercase; margin-bottom:6px;
}

table{width:100%!important; border-collapse:collapse;}
th{
  background:var(--sf2)!important; color:var(--a1)!important;
  font-family:var(--fh)!important; font-size:.78rem!important;
  letter-spacing:.07em!important; text-transform:uppercase; padding:10px 16px!important;
}
td{
  padding:9px 16px!important; border-bottom:1px solid var(--bd)!important;
  color:var(--tx)!important; font-size:.86rem!important;
}
tr:hover td{background:rgba(0,229,255,.03)!important;}

::-webkit-scrollbar{width:5px;}
::-webkit-scrollbar-track{background:var(--bg);}
::-webkit-scrollbar-thumb{background:#1c2d4a; border-radius:3px;}
::-webkit-scrollbar-thumb:hover{background:#243650;}

@keyframes fadeUp{from{opacity:0;transform:translateY(14px);}to{opacity:1;transform:none;}}
.stApp>section>div>div{animation:fadeUp .38s ease both;}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════

def hex_to_rgba(hex_color: str, alpha: float = 0.10) -> str:
    """'#rrggbb' -> 'rgba(r,g,b,alpha)'  (Plotly requires this; hex-in-rgba is invalid)."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def make_ohe(**kwargs) -> OneHotEncoder:
    """OHE with sparse_output=False, compatible with sklearn <1.2 and >=1.2."""
    key = "sparse_output" if "sparse_output" in inspect.signature(
        OneHotEncoder.__init__).parameters else "sparse"
    return OneHotEncoder(**{key: False}, **kwargs)


def layout(**extra) -> dict:
    """
    Shared Plotly layout dict.  xaxis/yaxis are only injected when 'polar' is
    absent, preventing interference with polar/ternary charts.
    """
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,22,36,0.55)",
        font=dict(family="DM Mono, monospace", color="#e2e8f0", size=11),
        colorway=["#00e5ff","#ff6b35","#7c3aed","#22d3a0","#f43f5e","#fbbf24"],
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1c2d4a", font=dict(size=11)),
        margin=dict(t=48, b=36, l=36, r=20),
    )
    if "polar" not in extra:
        base["xaxis"] = dict(gridcolor="#1c2d4a", linecolor="#1c2d4a", zeroline=False)
        base["yaxis"] = dict(gridcolor="#1c2d4a", linecolor="#1c2d4a", zeroline=False)
    base.update(extra)
    return base


OHE_COLS = ["Employment_Status","Marital_Status","Loan_Purpose",
            "Property_Area","Gender","Employer_Category"]
MODEL_COLORS = ["#00e5ff","#ff6b35","#7c3aed"]


# ══════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════
@st.cache_data
def load_and_preprocess() -> pd.DataFrame:
    try:
        df = pd.read_csv("data.csv")
    except FileNotFoundError:
        st.error(
            "**data.csv not found.** "
            "Place `data.csv` in the same folder as `loan_dashboard.py` and reload."
        )
        st.stop()

    num_cols = df.select_dtypes(include=["number"]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns
    df[num_cols] = SimpleImputer(strategy="mean").fit_transform(df[num_cols])
    df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])
    return df


# ══════════════════════════════════════════════
#  MODEL TRAINING  (@st.cache_resource avoids pickling sklearn objects)
# ══════════════════════════════════════════════
@st.cache_resource
def get_trained_models():
    df_raw = load_and_preprocess()
    df = df_raw.copy()
    df.drop(columns=["Applicant_ID"], errors="ignore", inplace=True)

    le = LabelEncoder()
    df["Education_Level"] = le.fit_transform(df["Education_Level"])
    df["Loan_Approved"]   = le.fit_transform(df["Loan_Approved"])

    ohe = make_ohe(drop="first", handle_unknown="ignore")
    enc    = ohe.fit_transform(df[OHE_COLS])
    enc_df = pd.DataFrame(enc, columns=ohe.get_feature_names_out(OHE_COLS), index=df.index)
    df = pd.concat([df.drop(columns=OHE_COLS), enc_df], axis=1)

    df["DTI_Ratio_sq"]    = df["DTI_Ratio"]    ** 2
    df["Credit_Score_sq"] = df["Credit_Score"] ** 2

    X = df.drop(columns=["Loan_Approved","Credit_Score","DTI_Ratio"])
    y = df["Loan_Approved"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    results = {}
    for name, mdl in [
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)),
        ("K-Nearest Neighbors", KNeighborsClassifier(n_neighbors=5)),
        ("Naive Bayes",         GaussianNB()),
    ]:
        mdl.fit(Xtr, y_train)
        yp = mdl.predict(Xte)
        results[name] = dict(
            model     = mdl,
            accuracy  = round(accuracy_score(y_test, yp)  * 100, 2),
            precision = round(precision_score(y_test, yp) * 100, 2),
            recall    = round(recall_score(y_test, yp)    * 100, 2),
            f1        = round(f1_score(y_test, yp)        * 100, 2),
            cm        = confusion_matrix(y_test, yp).tolist(),  # JSON-safe
        )

    return results, X.columns.tolist(), scaler, ohe, le


# ── bootstrap
df_raw = load_and_preprocess()
model_results, feature_names, scaler, ohe, le = get_trained_models()


# ══════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:24px 0 12px'>
      <div style='font-family:Syne,sans-serif;font-size:1.55rem;font-weight:800;
                  background:linear-gradient(90deg,#e2e8f0,#00e5ff);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
        💳 LoanIQ
      </div>
      <div style='color:#64748b;font-size:.68rem;margin-top:5px;letter-spacing:.1em;'>
        LOAN APPROVAL INTELLIGENCE
      </div>
    </div>
    <hr style='border-color:#1c2d4a;margin:0 0 14px'>
    """, unsafe_allow_html=True)

    nav = st.radio("nav",
                   ["📊 Overview","🔍 EDA","🤖 Models","🎯 Predictor"],
                   label_visibility="collapsed")

    st.markdown("""
    <hr style='border-color:#1c2d4a;margin:14px 0 8px'>
    <div class='sec-lbl' style='padding:0 4px'>Filters</div>
    """, unsafe_allow_html=True)

    area_opts = sorted(df_raw["Property_Area"].dropna().unique())
    emp_opts  = sorted(df_raw["Employment_Status"].dropna().unique())

    area_filter = st.multiselect("Property Area",     area_opts, default=list(area_opts))
    emp_filter  = st.multiselect("Employment Status", emp_opts,  default=list(emp_opts))
    cr_min = int(df_raw["Credit_Score"].min())
    cr_max = int(df_raw["Credit_Score"].max())
    credit_range = st.slider("Credit Score Range", cr_min, cr_max, (cr_min, cr_max))

    # guard empty selections
    if not area_filter: area_filter = list(area_opts)
    if not emp_filter:  emp_filter  = list(emp_opts)

    df_f = df_raw[
        df_raw["Property_Area"].isin(area_filter)    &
        df_raw["Employment_Status"].isin(emp_filter) &
        df_raw["Credit_Score"].between(*credit_range)
    ].copy()

    pct_shown = len(df_f) / max(len(df_raw), 1) * 100
    null_pct  = df_raw.isnull().mean().mean() * 100

    st.markdown(f"""
    <div style='background:#0d1624;border:1px solid #1c2d4a;border-radius:12px;
                padding:14px;margin-top:14px;text-align:center;'>
      <div style='color:#64748b;font-size:.66rem;letter-spacing:.1em;text-transform:uppercase;'>
        Filtered Records</div>
      <div style='font-family:Syne,sans-serif;font-size:1.7rem;font-weight:800;
                  color:#00e5ff;margin:4px 0 2px;'>{len(df_f):,}</div>
      <div style='color:#64748b;font-size:.7rem;'>{pct_shown:.0f}% of dataset</div>
    </div>
    <div style='background:#0d1624;border:1px solid #1c2d4a;border-radius:12px;
                padding:10px 14px;margin-top:8px;display:flex;align-items:center;gap:10px;'>
      <div style='width:9px;height:9px;border-radius:50%;flex-shrink:0;
                  background:{"#22d3a0" if null_pct<5 else "#fbbf24"};'></div>
      <div style='font-size:.7rem;color:#64748b;'>
        Raw null rate&nbsp;<span style='color:#e2e8f0;'>{null_pct:.1f}%</span>&nbsp;(imputed)
      </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  HERO
# ══════════════════════════════════════════════
st.markdown("""
<div class='hero'>
  <h1>Loan Approval Intelligence</h1>
  <p>End-to-end ML pipeline — cleaning · EDA · multi-model training · live predictor</p>
  <span class='pill'>Logistic Regression</span>
  <span class='pill'>K-Nearest Neighbors</span>
  <span class='pill'>Naive Bayes</span>
  <span class='pill'>Plotly Interactive</span>
</div>""", unsafe_allow_html=True)

# fallback if filter returns nothing
if len(df_f) == 0:
    st.warning("⚠️  No records match the current filters — showing the full dataset.")
    df_f = df_raw.copy()


# ══════════════════════════════════════════════
#  PAGE: OVERVIEW
# ══════════════════════════════════════════════
if nav == "📊 Overview":

    n_total    = len(df_f)
    n_approved = (df_f["Loan_Approved"] == "Yes").sum()
    n_denied   = (df_f["Loan_Approved"] == "No").sum()
    pct_app    = n_approved / max(n_total, 1) * 100
    avg_income = df_f["Applicant_Income"].mean()
    avg_loan   = df_f["Loan_Amount"].mean()

    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("📋 Applicants",   f"{n_total:,}")
    k2.metric("✅ Approved",      f"{n_approved:,}", delta=f"{pct_app:.1f}%")
    k3.metric("❌ Denied",        f"{n_denied:,}")
    k4.metric("💰 Avg Income",    f"${avg_income:,.0f}")
    k5.metric("🏦 Avg Loan",      f"${avg_loan:,.0f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Donut + income bar
    ca, cb = st.columns([1.05, 1])
    with ca:
        st.markdown("<div class='cc'>", unsafe_allow_html=True)
        counts = df_f["Loan_Approved"].value_counts()
        clr_map = {"Yes":"#22d3a0","No":"#f43f5e"}
        fig = go.Figure(go.Pie(
            labels=counts.index, values=counts.values, hole=0.62,
            marker=dict(
                colors=[clr_map.get(l,"#64748b") for l in counts.index],
                line=dict(color="#070b14", width=3),
            ),
            textfont=dict(size=13), pull=[0.03]*len(counts),
        ))
        fig.update_layout(**layout(
            title=dict(text="Loan Approval Distribution", font=dict(size=15,family="Syne")),
            annotations=[dict(
                text=f"<b>{pct_app:.0f}%</b><br>Approved",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=17, color="#00e5ff", family="Syne"),
            )],
        ))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with cb:
        st.markdown("<div class='cc'>", unsafe_allow_html=True)
        ig = df_f.groupby("Loan_Approved")["Applicant_Income"].mean().reset_index()
        fig = go.Figure(go.Bar(
            x=ig["Loan_Approved"],
            y=ig["Applicant_Income"],
            marker_color=[clr_map.get(v,"#64748b") for v in ig["Loan_Approved"]],
            text=[f"${v:,.0f}" for v in ig["Applicant_Income"]],
            textposition="outside", width=0.45,
        ))
        fig.update_layout(**layout(
            title=dict(text="Avg Income by Loan Status", font=dict(size=15,family="Syne")),
            yaxis=dict(gridcolor="#1c2d4a", linecolor="#1c2d4a", zeroline=False,
                       range=[0, ig["Applicant_Income"].max()*1.28]),
            xaxis=dict(gridcolor="#1c2d4a", linecolor="#1c2d4a"),
            yaxis_title="Avg Income ($)", xaxis_title="",
        ))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Loan purpose
    st.markdown("<div class='cc'>", unsafe_allow_html=True)
    pur = df_f.groupby(["Loan_Purpose","Loan_Approved"]).size().reset_index(name="count")
    fig = px.bar(pur, x="Loan_Purpose", y="count", color="Loan_Approved", barmode="group",
                 color_discrete_map={"Yes":"#22d3a0","No":"#f43f5e"},
                 template="none", text_auto=True)
    fig.update_layout(**layout(
        title=dict(text="Applications by Loan Purpose", font=dict(size=15,family="Syne")),
        xaxis_title="", yaxis_title="Count", legend_title_text="Approved",
    ))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Area sunburst + employment stacked
    cc, cd = st.columns(2)
    with cc:
        st.markdown("<div class='cc'>", unsafe_allow_html=True)
        ag = df_f.groupby(["Property_Area","Loan_Approved"]).size().reset_index(name="n")
        fig = px.sunburst(ag, path=["Property_Area","Loan_Approved"], values="n",
                          color="Loan_Approved",
                          color_discrete_map={"Yes":"#22d3a0","No":"#f43f5e"})
        fig.update_layout(**layout(
            title=dict(text="Area × Approval Sunburst", font=dict(size=15,family="Syne")),
        ))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with cd:
        st.markdown("<div class='cc'>", unsafe_allow_html=True)
        eg = df_f.groupby(["Employment_Status","Loan_Approved"]).size().reset_index(name="n")
        fig = px.bar(eg, y="Employment_Status", x="n", color="Loan_Approved",
                     barmode="stack", orientation="h",
                     color_discrete_map={"Yes":"#00e5ff","No":"#7c3aed"}, template="none")
        fig.update_layout(**layout(
            title=dict(text="Employment Status vs Approval", font=dict(size=15,family="Syne")),
            xaxis_title="Count", yaxis_title="", legend_title_text="Approved",
        ))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Gender + marital
    ce, cf = st.columns(2)
    with ce:
        st.markdown("<div class='cc'>", unsafe_allow_html=True)
        gg = df_f.groupby(["Gender","Loan_Approved"]).size().reset_index(name="n")
        fig = px.bar(gg, x="Gender", y="n", color="Loan_Approved", barmode="group",
                     color_discrete_map={"Yes":"#22d3a0","No":"#f43f5e"},
                     template="none", text_auto=True)
        fig.update_layout(**layout(
            title=dict(text="Gender vs Approval", font=dict(size=15,family="Syne")),
            xaxis_title="", yaxis_title="Count",
        ))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with cf:
        st.markdown("<div class='cc'>", unsafe_allow_html=True)
        mg = df_f.groupby(["Marital_Status","Loan_Approved"]).size().reset_index(name="n")
        fig = px.bar(mg, x="Marital_Status", y="n", color="Loan_Approved", barmode="group",
                     color_discrete_map={"Yes":"#22d3a0","No":"#ff6b35"},
                     template="none", text_auto=True)
        fig.update_layout(**layout(
            title=dict(text="Marital Status vs Approval", font=dict(size=15,family="Syne")),
            xaxis_title="", yaxis_title="Count",
        ))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  PAGE: EDA
# ══════════════════════════════════════════════
elif nav == "🔍 EDA":

    st.markdown("### 🔬 Exploratory Data Analysis")

    # Credit score density
    st.markdown("<div class='cc'>", unsafe_allow_html=True)
    fig = go.Figure()
    for lbl, clr in [("Yes","#22d3a0"),("No","#f43f5e")]:
        sub = df_f.loc[df_f["Loan_Approved"]==lbl, "Credit_Score"]
        if not sub.empty:
            fig.add_trace(go.Histogram(x=sub, name=f"Approved: {lbl}",
                marker_color=clr, opacity=0.72, nbinsx=28,
                histnorm="probability density"))
    fig.update_layout(**layout(
        title=dict(text="Credit Score Distribution by Approval",
                   font=dict(size=15,family="Syne")),
        barmode="overlay", xaxis_title="Credit Score", yaxis_title="Density",
    ))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Box plots 2x3  — legend de-duped via legend_added set
    NUM_FEATS = ["Applicant_Income","Credit_Score","DTI_Ratio","Savings","Age","Loan_Amount"]
    fig_box   = make_subplots(rows=2, cols=3, subplot_titles=NUM_FEATS,
                               horizontal_spacing=0.09, vertical_spacing=0.16)
    AC = {"Yes":"#22d3a0","No":"#f43f5e"}
    legend_added: set = set()
    for (r,c), feat in zip([(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)], NUM_FEATS):
        for lbl in ["Yes","No"]:
            sub = df_f.loc[df_f["Loan_Approved"]==lbl, feat]
            show_leg = lbl not in legend_added
            fig_box.add_trace(
                go.Box(y=sub, name=lbl, marker_color=AC[lbl], line_color=AC[lbl],
                       boxmean=True, showlegend=show_leg), row=r, col=c)
            legend_added.add(lbl)
        fig_box.update_yaxes(gridcolor="#1c2d4a", linecolor="#1c2d4a",
                              title_text=feat, row=r, col=c)
        fig_box.update_xaxes(gridcolor="#1c2d4a", linecolor="#1c2d4a",
                              showticklabels=False, row=r, col=c)
    fig_box.update_layout(**layout(
        title=dict(text="Feature Distributions vs Loan Approval",
                   font=dict(size=15,family="Syne")),
        height=560,
    ))
    st.markdown("<div class='cc'>", unsafe_allow_html=True)
    st.plotly_chart(fig_box, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Scatter income vs loan amount
    st.markdown("<div class='cc'>", unsafe_allow_html=True)
    fig = px.scatter(df_f, x="Applicant_Income", y="Loan_Amount",
                     color="Loan_Approved", size="Credit_Score",
                     color_discrete_map={"Yes":"#22d3a0","No":"#f43f5e"},
                     hover_data=["Age","DTI_Ratio","Existing_Loans"],
                     opacity=0.60, template="none")
    fig.update_layout(**layout(
        title=dict(text="Income vs Loan Amount  (bubble = Credit Score)",
                   font=dict(size=15,family="Syne")),
        xaxis_title="Applicant Income ($)", yaxis_title="Loan Amount ($)",
    ))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Correlation heatmap  — .tolist() for JSON safety
    st.markdown("<div class='cc'>", unsafe_allow_html=True)
    HM = ["Applicant_Income","Coapplicant_Income","Credit_Score",
          "DTI_Ratio","Savings","Loan_Amount","Age","Existing_Loans"]
    corr = df_f[HM].corr().round(2)
    fig = go.Figure(go.Heatmap(
        z=corr.values.tolist(), x=corr.columns.tolist(), y=corr.index.tolist(),
        colorscale=[[0,"#7c3aed"],[0.45,"#111e30"],[1,"#00e5ff"]],
        zmin=-1, zmax=1,
        text=corr.values.tolist(), texttemplate="%{text:.2f}",
        hoverongaps=False,
    ))
    fig.update_layout(**layout(
        title=dict(text="Correlation Heatmap", font=dict(size=15,family="Syne")),
        height=480,
        xaxis=dict(tickangle=-38, gridcolor="#1c2d4a", linecolor="#1c2d4a"),
        yaxis=dict(gridcolor="#1c2d4a", linecolor="#1c2d4a"),
    ))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Violin (age) + DTI bar
    cg, ch = st.columns(2)
    with cg:
        st.markdown("<div class='cc'>", unsafe_allow_html=True)
        fig = px.violin(df_f, x="Loan_Approved", y="Age", color="Loan_Approved",
                        color_discrete_map={"Yes":"#00e5ff","No":"#ff6b35"},
                        box=True, points="outliers", template="none")
        fig.update_layout(**layout(
            title=dict(text="Age Distribution by Approval", font=dict(size=15,family="Syne")),
            showlegend=False, xaxis_title="", yaxis_title="Age",
        ))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with ch:
        st.markdown("<div class='cc'>", unsafe_allow_html=True)
        dg = df_f.groupby("Loan_Approved")["DTI_Ratio"].mean().reset_index()
        clrs = [{"Yes":"#22d3a0","No":"#f43f5e"}.get(v,"#64748b") for v in dg["Loan_Approved"]]
        fig = go.Figure(go.Bar(
            x=dg["Loan_Approved"], y=dg["DTI_Ratio"],
            marker_color=clrs,
            text=[f"{v:.3f}" for v in dg["DTI_Ratio"]],
            textposition="outside", width=0.4,
        ))
        fig.update_layout(**layout(
            title=dict(text="Avg DTI Ratio by Approval", font=dict(size=15,family="Syne")),
            xaxis_title="", yaxis_title="DTI Ratio",
            yaxis=dict(gridcolor="#1c2d4a", linecolor="#1c2d4a", zeroline=False,
                       range=[0, dg["DTI_Ratio"].max()*1.3]),
        ))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Savings vs collateral
    st.markdown("<div class='cc'>", unsafe_allow_html=True)
    fig = px.scatter(df_f, x="Savings", y="Collateral_Value", color="Loan_Approved",
                     color_discrete_map={"Yes":"#22d3a0","No":"#f43f5e"},
                     opacity=0.55, template="none",
                     hover_data=["Loan_Amount","Credit_Score","Age"])
    fig.update_layout(**layout(
        title=dict(text="Savings vs Collateral Value", font=dict(size=15,family="Syne")),
        xaxis_title="Savings ($)", yaxis_title="Collateral Value ($)",
    ))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  PAGE: MODELS
# ══════════════════════════════════════════════
elif nav == "🤖 Models":

    st.markdown("### 🤖 Model Training & Evaluation")

    model_names = list(model_results.keys())
    metrics     = ["accuracy","precision","recall","f1"]
    met_labels  = ["Accuracy","Precision","Recall","F1-Score"]

    # Radar  — fillcolor via hex_to_rgba (not raw hex notation)
    st.markdown("<div class='cc'>", unsafe_allow_html=True)
    fig_radar = go.Figure()
    for (name, res), clr in zip(model_results.items(), MODEL_COLORS):
        vals = [res[m] for m in metrics] + [res["accuracy"]]   # close polygon
        fig_radar.add_trace(go.Scatterpolar(
            r=vals,
            theta=met_labels + [met_labels[0]],
            name=name,
            line=dict(color=clr, width=2.8),
            fill="toself",
            fillcolor=hex_to_rgba(clr, 0.12),
        ))
    fig_radar.update_layout(**layout(
        polar=dict(
            bgcolor="rgba(13,22,36,0.7)",
            radialaxis=dict(
                visible=True, range=[0, 100],
                gridcolor="#1c2d4a", linecolor="#1c2d4a",
                tickfont=dict(size=10, color="#64748b"),
                ticksuffix="%",
            ),
            angularaxis=dict(
                gridcolor="#1c2d4a", linecolor="#1c2d4a",
                tickfont=dict(size=12, family="Syne"),
            ),
        ),
        title=dict(text="Model Performance Radar", font=dict(size=15,family="Syne")),
        height=440,
    ))
    st.plotly_chart(fig_radar, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Grouped metric bar
    st.markdown("<div class='cc'>", unsafe_allow_html=True)
    bar_rows = [
        {"Model":name, "Metric":ml, "Score":res[m]}
        for name, res in model_results.items()
        for m, ml in zip(metrics, met_labels)
    ]
    fig = px.bar(pd.DataFrame(bar_rows), x="Metric", y="Score", color="Model",
                 barmode="group",
                 color_discrete_map=dict(zip(model_names, MODEL_COLORS)),
                 template="none", text_auto=".1f")
    fig.update_layout(**layout(
        title=dict(text="All Metrics Comparison", font=dict(size=15,family="Syne")),
        yaxis_range=[0, 110], yaxis_title="Score (%)",
    ))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Score table
    st.markdown("#### 📋 Detailed Score Table")
    st.table(pd.DataFrame([
        {"Model":n, "Accuracy":f"{r['accuracy']:.2f} %",
         "Precision":f"{r['precision']:.2f} %",
         "Recall":f"{r['recall']:.2f} %", "F1-Score":f"{r['f1']:.2f} %"}
        for n, r in model_results.items()
    ]).set_index("Model"))

    # Confusion matrices  — cm already list-of-lists
    st.markdown("#### 🔲 Confusion Matrices")
    cm_cols = st.columns(3)
    for i, (name, res) in enumerate(model_results.items()):
        with cm_cols[i]:
            st.markdown("<div class='cc'>", unsafe_allow_html=True)
            cm = res["cm"]
            fig = go.Figure(go.Heatmap(
                z=cm, x=["Pred: No","Pred: Yes"], y=["Actual: No","Actual: Yes"],
                colorscale=[[0,"#0d1624"],[0.5,"#1c2d4a"],[1,"#00e5ff"]],
                showscale=False, text=cm, texttemplate="<b>%{text}</b>",
                hoverongaps=False,
            ))
            fig.update_layout(**layout(
                title=dict(text=name, font=dict(size=13,family="Syne")),
                height=270, margin=dict(t=52,b=22,l=22,r=22),
            ))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # Best model callout
    best     = max(model_results, key=lambda k: model_results[k]["f1"])
    best_f1  = model_results[best]["f1"]
    best_acc = model_results[best]["accuracy"]
    st.markdown(f"""
    <div style='background:rgba(0,229,255,.05);border:1px solid #00e5ff;
                border-radius:14px;padding:22px 30px;margin-top:8px;
                display:flex;align-items:center;gap:18px;'>
      <div style='font-size:2rem;'>🏆</div>
      <div>
        <div style='font-family:Syne,sans-serif;font-size:.68rem;color:#00e5ff;
                    letter-spacing:.12em;text-transform:uppercase;margin-bottom:4px;'>
          Best Model by F1-Score</div>
        <div style='font-family:Syne,sans-serif;font-size:1.2rem;font-weight:800;
                    color:#e2e8f0;'>{best}</div>
        <div style='color:#64748b;font-size:.8rem;margin-top:3px;'>
          F1&nbsp;<span style='color:#22d3a0;font-weight:700;'>{best_f1:.2f}%</span>
          &ensp;·&ensp;
          Accuracy&nbsp;<span style='color:#00e5ff;font-weight:700;'>{best_acc:.2f}%</span>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  PAGE: PREDICTOR
# ══════════════════════════════════════════════
elif nav == "🎯 Predictor":

    st.markdown("### 🎯 Live Loan Approval Predictor")
    st.markdown(
        "<p style='color:#64748b;'>Enter applicant details and hit "
        "<b>Run Prediction</b> to get a real-time verdict from all three trained models.</p>",
        unsafe_allow_html=True)

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("<div class='sec-lbl'>💼 Financial Details</div>",
                        unsafe_allow_html=True)
            app_income   = st.number_input("Applicant Income ($)",     1000,100000,10000,500)
            coapp_income = st.number_input("Co-applicant Income ($)",     0, 50000, 2000,500)
            loan_amount  = st.number_input("Loan Amount ($)",          1000,500000,25000,1000)
            loan_term    = st.selectbox("Loan Term (months)", [12,24,36,48,60,72,84])
            savings      = st.number_input("Savings ($)",                 0,200000,10000,1000)
            collateral   = st.number_input("Collateral Value ($)",        0,500000,20000,1000)

        with c2:
            st.markdown("<div class='sec-lbl'>📋 Credit & Risk</div>",
                        unsafe_allow_html=True)
            credit_score   = st.slider("Credit Score",   300, 850, 680)
            dti_ratio      = st.slider("DTI Ratio",      0.0, 1.0, 0.35, 0.01)
            existing_loans = st.selectbox("Existing Loans", [0,1,2,3,4])
            age            = st.slider("Age",             18,  70,  35)
            dependents     = st.selectbox("Dependents",  [0,1,2,3])

        with c3:
            st.markdown("<div class='sec-lbl'>🧑 Personal Details</div>",
                        unsafe_allow_html=True)
            employment = st.selectbox("Employment Status",
                                      ["Salaried","Self-employed","Contract","Unemployed"])
            marital    = st.selectbox("Marital Status",  ["Married","Single"])
            education  = st.selectbox("Education Level", ["Graduate","Not Graduate"])
            gender     = st.selectbox("Gender",          ["Male","Female"])
            employer   = st.selectbox("Employer Category",
                                      ["Private","Government","MNC","Business","Unemployed"])
            area       = st.selectbox("Property Area",   ["Urban","Semiurban","Rural"])
            purpose    = st.selectbox("Loan Purpose",
                                      ["Home","Car","Personal","Business","Education"])

        submitted = st.form_submit_button("🚀  Run Prediction", use_container_width=True)

    if submitted:
        # All numerics explicitly cast to float to match scaler expectations
        raw = {
            "Applicant_Income":   float(app_income),
            "Coapplicant_Income": float(coapp_income),
            "Age":                float(age),
            "Credit_Score":       float(credit_score),
            "Existing_Loans":     float(existing_loans),
            "DTI_Ratio":          float(dti_ratio),
            "Savings":            float(savings),
            "Collateral_Value":   float(collateral),
            "Loan_Amount":        float(loan_amount),
            "Loan_Term":          float(loan_term),
            "Education_Level":    1.0 if education == "Graduate" else 0.0,
            "Dependents":         float(dependents),
            "Employment_Status":  employment,
            "Marital_Status":     marital,
            "Loan_Purpose":       purpose,
            "Property_Area":      area,
            "Gender":             gender,
            "Employer_Category":  employer,
        }
        row_df = pd.DataFrame([raw])

        # OHE encode
        enc_arr  = ohe.transform(row_df[OHE_COLS])
        enc_part = pd.DataFrame(enc_arr, columns=ohe.get_feature_names_out(OHE_COLS))
        row_df   = pd.concat([row_df.drop(columns=OHE_COLS).reset_index(drop=True),
                               enc_part], axis=1)

        # Engineered features (must mirror training)
        row_df["DTI_Ratio_sq"]    = row_df["DTI_Ratio"]    ** 2
        row_df["Credit_Score_sq"] = row_df["Credit_Score"] ** 2
        row_df.drop(columns=["Credit_Score","DTI_Ratio"], errors="ignore", inplace=True)

        # Align columns, fill any gaps, ensure float dtype
        for col_name in feature_names:
            if col_name not in row_df.columns:
                row_df[col_name] = 0.0
        row_df  = row_df[feature_names].fillna(0.0).astype(float)
        X_pred  = scaler.transform(row_df)

        # Individual verdicts
        st.markdown("---")
        st.markdown("#### 📊 Individual Model Verdicts")
        pred_cols  = st.columns(3)
        model_list = list(model_results.items())
        vote = 0

        for i, (name, res) in enumerate(model_list):
            pred  = int(res["model"].predict(X_pred)[0])
            proba = (res["model"].predict_proba(X_pred)[0]
                     if hasattr(res["model"], "predict_proba") else [0.5, 0.5])
            vote += pred
            conf  = max(proba) * 100
            badge = ("<span class='badge-yes'>✅ APPROVED</span>"
                     if pred == 1 else "<span class='badge-no'>❌ DENIED</span>")
            with pred_cols[i]:
                st.markdown(f"""
                <div class='cc' style='text-align:center;padding:24px 16px;'>
                  <div style='font-family:Syne,sans-serif;font-size:.82rem;
                              color:#64748b;margin-bottom:12px;
                              letter-spacing:.06em;text-transform:uppercase;'>{name}</div>
                  {badge}
                  <div style='margin-top:14px;font-size:.76rem;color:#64748b;'>
                    Confidence
                    <span style='color:#00e5ff;font-weight:700;font-size:.95rem;'>
                      &ensp;{conf:.1f}%
                    </span>
                  </div>
                </div>""", unsafe_allow_html=True)

        # Majority-vote verdict
        vc   = "#22d3a0" if vote >= 2 else "#f43f5e"
        vbg  = "rgba(34,211,160,.06)" if vote >= 2 else "rgba(244,63,94,.06)"
        vlbl = "✅  LOAN APPROVED"     if vote >= 2 else "❌  LOAN DENIED"
        vsub = ("Majority of models recommend approval."
                if vote >= 2 else "Majority of models recommend denial.")
        st.markdown(f"""
        <div style='background:{vbg};border:1px solid {vc};border-radius:18px;
                    padding:32px;text-align:center;margin-top:20px;'>
          <div style='font-family:Syne,sans-serif;font-size:2rem;font-weight:800;
                      color:{vc};letter-spacing:-.02em;'>{vlbl}</div>
          <div style='color:#64748b;margin-top:8px;font-size:.84rem;'>{vsub}</div>
          <div style='color:#64748b;margin-top:4px;font-size:.78rem;'>
            {vote} / {len(model_list)} models voted to approve
          </div>
        </div>""", unsafe_allow_html=True)

        # Stacked probability chart
        proba_rows = []
        for name, res in model_list:
            if hasattr(res["model"], "predict_proba"):
                p = res["model"].predict_proba(X_pred)[0]
                proba_rows += [
                    {"Model":name,"Probability":round(p[1]*100,1),"Outcome":"Approved"},
                    {"Model":name,"Probability":round(p[0]*100,1),"Outcome":"Denied"},
                ]
        if proba_rows:
            st.markdown("<div class='cc' style='margin-top:22px;'>", unsafe_allow_html=True)
            fig = px.bar(pd.DataFrame(proba_rows), x="Model", y="Probability",
                         color="Outcome", barmode="stack",
                         color_discrete_map={"Approved":"#22d3a0","Denied":"#f43f5e"},
                         template="none", text_auto=".1f")
            fig.update_layout(**layout(
                title=dict(text="Approval Probability by Model",
                           font=dict(size=15,family="Syne")),
                yaxis_range=[0, 108], yaxis_title="Probability (%)",
            ))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Input summary expander
        with st.expander("📋 View submitted values"):
            num_vals = {k: v for k, v in raw.items() if k not in OHE_COLS}
            cat_vals = {k: v for k, v in raw.items() if k in OHE_COLS}
            ci, cj   = st.columns(2)
            ci.json(num_vals)
            cj.json(cat_vals)


# ══════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════
st.markdown("""
<hr style='border-color:#1c2d4a;margin-top:44px'>
<div style='text-align:center;color:#243650;font-size:.7rem;padding-bottom:22px;
            font-family:DM Mono,monospace;letter-spacing:.06em;'>
  LoanIQ v3 &nbsp;·&nbsp; Streamlit + Plotly &nbsp;·&nbsp; Scikit-learn
</div>""", unsafe_allow_html=True)