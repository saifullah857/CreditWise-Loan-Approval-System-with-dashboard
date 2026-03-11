import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set Plotly default template to dark
pio.templates.default = "plotly_dark"

# Page Configuration
st.set_page_config(page_title="Loan Approval Dashboard", layout="wide", page_icon="💰")

# --- CUSTOM CSS: DARK MODE UI ---
st.markdown("""
    <style>
    /* Main App Background - Deep Black */
    .stApp {
        background-color: #0e1117;
    }

    /* Sidebar Background - Slightly lighter black */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }

    /* Metric Cards: Dark Charcoal, subtle silver border, white text */
    div[data-testid="stMetric"] {
        background-color: #161b22;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #30363d;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }

    /* Force ALL text to WHITE/SILVER */
    h1, h2, h3, p, span, label, .stMarkdown {
        color: #ffffff !important;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Target specific Metric Label and Value */
    [data-testid="stMetricLabel"] div {
        color: #8b949e !important; /* Muted silver for labels */
    }
    [data-testid="stMetricValue"] div {
        color: #ffffff !important;
        font-weight: bold;
    }
    
    /* Input fields and Selectboxes for Dark Mode */
    .stNumberInput input, .stSelectbox div {
        background-color: #0d1117 !important;
        color: white !important;
        border-color: #30363d !important;
    }

    /* Horizontal Rule color */
    hr {
        border-top: 1px solid #30363d;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------- PREPROCESSING LOGIC ---------------- #

@st.cache_data
def load_and_clean_data():
    df = pd.read_csv("data.csv")
    categorical_cols = df.select_dtypes(include=["object"]).columns
    numerical_cols = df.select_dtypes(include=["number"]).columns
    
    num_imp = SimpleImputer(strategy="mean")
    df[numerical_cols] = num_imp.fit_transform(df[numerical_cols])
    
    cat_imp = SimpleImputer(strategy="most_frequent")
    df[categorical_cols] = cat_imp.fit_transform(df[categorical_cols])
    
    df["DTI_Ratio_sq"] = df["DTI_Ratio"]**2
    df["Credit_Score_sq"] = df["Credit_Score"]**2
    return df

def process_for_modeling(df):
    temp_df = df.copy()
    temp_df = temp_df.drop(columns=["Applicant_ID"], errors="ignore")
    le = LabelEncoder()
    temp_df["Education_Level"] = le.fit_transform(temp_df["Education_Level"])
    temp_df["Loan_Approved"] = le.fit_transform(temp_df["Loan_Approved"])
    
    ohe_cols = ["Employment_Status","Marital_Status","Loan_Purpose","Property_Area","Gender","Employer_Category"]
    ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    encoded = ohe.fit_transform(temp_df[ohe_cols])
    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(ohe_cols), index=temp_df.index)
    
    final_df = pd.concat([temp_df.drop(columns=ohe_cols), encoded_df], axis=1)
    X = final_df.drop(columns=["Loan_Approved", "Credit_Score", "DTI_Ratio"])
    y = final_df["Loan_Approved"]
    return X, y

# ---------------- DASHBOARD UI ---------------- #

df = load_and_clean_data()
X, y = process_for_modeling(df)

st.sidebar.title("🌑 Loan Analytics Pro")
page = st.sidebar.radio("Navigate", ["📊 Overview & EDA", "🤖 Model Training", "🎯 Prediction Tool"])

if page == "📊 Overview & EDA":
    st.title("🏦 Financial Insights Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Applicants", len(df))
    app_rate = (df['Loan_Approved'] == 'Yes').mean()
    col2.metric("Approval Rate", f"{app_rate:.1%}")
    col3.metric("Avg Income", f"${df['Applicant_Income'].mean():,.0f}")
    col4.metric("Avg Credit Score", f"{df['Credit_Score'].mean():.0f}")

    st.markdown("---")

    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Approval Distribution")
        # Using neon-style colors for dark mode
        fig_pie = px.pie(df, names='Loan_Approved', hole=0.4, 
                         color_discrete_sequence=['#58a6ff', '#f85149'])
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        st.subheader("Income vs Loan Amount")
        fig_scatter = px.scatter(df, x="Applicant_Income", y="Loan_Amount", 
                                 color="Loan_Approved", 
                                 color_discrete_map={'Yes': '#58a6ff', 'No': '#f85149'},
                                 hover_data=['Age', 'Credit_Score'])
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Credit Score Distribution")
    fig_hist = px.histogram(df, x="Credit_Score", color="Loan_Approved", 
                            marginal="box", barmode="overlay", 
                            color_discrete_map={'Yes': '#238636', 'No': '#da3633'})
    st.plotly_chart(fig_hist, use_container_width=True)

elif page == "🤖 Model Training":
    st.title("🧠 ML Model Performance")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model_choice = st.selectbox("Choose a Model", ["Logistic Regression", "K-Nearest Neighbors", "Naive Bayes"])
    
    if model_choice == "Logistic Regression":
        model = LogisticRegression()
    elif model_choice == "K-Nearest Neighbors":
        model = KNeighborsClassifier(n_neighbors=5)
    else:
        model = GaussianNB()
        
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{accuracy_score(y_test, preds):.2%}")
    m2.metric("Precision", f"{precision_score(y_test, preds):.2%}")
    m3.metric("Recall", f"{recall_score(y_test, preds):.2%}")
    m4.metric("F1 Score", f"{f1_score(y_test, preds):.2%}")
    
    st.subheader("Feature Correlation Matrix")
    corr = df.select_dtypes(include=[np.number]).corr()
    fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale='Viridis')
    st.plotly_chart(fig_corr, use_container_width=True)

elif page == "🎯 Prediction Tool":
    st.title("🔍 Loan Eligibility Predictor")
    
    with st.form("prediction_form"):
        k1, k2 = st.columns(2)
        with k1:
            income = st.number_input("Applicant Income", value=5000)
            c_score = st.slider("Credit Score", 300, 850, 650)
            dti = st.slider("DTI Ratio", 0.0, 1.0, 0.3)
        with k2:
            loan_amt = st.number_input("Requested Loan Amount", value=20000)
            savings = st.number_input("Savings Balance", value=1000)
            age = st.slider("Age", 18, 100, 30)

        submit = st.form_submit_button("Predict Status")

    if submit:
        # Logistic Regression style check
        is_approved = (c_score > 620 and dti < 0.45) or (income > loan_amt * 0.35)
        
        if is_approved:
            st.success("✅ Prediction: APPROVED")
            st.balloons()
        else:
            st.error("❌ Prediction: REJECTED")