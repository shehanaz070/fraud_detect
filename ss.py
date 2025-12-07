# app.py
import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Fraud Detection System", layout="wide", initial_sidebar_state="collapsed")

# ------------------ Stylish CSS (Gradient + Glass cards + Animations) ------------------
st.markdown(
    """
    <style>
    :root{
      --glass-bg: rgba(255,255,255,0.06);
      --glass-border: rgba(255,255,255,0.12);
      --accent: linear-gradient(135deg, #7b61ff 0%, #34d4ff 100%);
      --card-radius: 14px;
      --glass-shadow: 0 8px 30px rgba(11,15,30,0.45);
    }

    body {
      background: linear-gradient(135deg,#090b2a 0%, #2b1a5f 35%, #1a6b8f 100%);
      color: #111111;  /* changed to dark color */
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    .main > div[class*="block-container"] {
      padding-top: 18px;
      padding-left: 18px;
      padding-right: 18px;
      padding-bottom: 60px;
    }

    .header {
      background: linear-gradient(90deg, rgba(123,97,255,0.12), rgba(52,212,255,0.06));
      border-radius: var(--card-radius);
      padding: 18px;
      border: 1px solid var(--glass-border);
      box-shadow: var(--glass-shadow);
      margin-bottom: 18px;
      display:flex;
      align-items:center;
      justify-content:space-between;
    }

    .title {
      font-size: 26px;
      font-weight:700;
      margin: 0;
      color: #111111;   /* DARK heading */
      display:flex;
      gap:12px;
      align-items:center;
    }

    .subtitle {
      color: #333333;  /* DARK subtitle */
      font-size:14px;
      margin:0;
    }

    .glass {
      background: var(--glass-bg);
      border: 1px solid var(--glass-border);
      border-radius: var(--card-radius);
      padding: 14px;
      box-shadow: var(--glass-shadow);
    }

    .muted {
      color: #555555;   /* slightly darker muted text */
      font-size:13px;
    }

    .small {
      font-size:13px;
    }

    .stat {
      font-weight:700;
      font-size:20px;
      color:#111111;  /* dark stat text */
    }

    /* floating animated accent */
    .accent-bar {
      height:6px;
      border-radius:6px;
      background: linear-gradient(90deg,#7b61ff,#34d4ff);
      animation: slide 4s linear infinite;
      margin-top:10px;
      box-shadow: 0 6px 18px rgba(52,212,255,0.08);
    }

    @keyframes slide {
      0% { transform: translateX(-10%); }
      50% { transform: translateX(10%); }
      100% { transform: translateX(-10%); }
    }

    /* responsive tweaks for Streamlit table */
    table.dataframe tbody tr th:only-of-type {
      vertical-align: middle;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------ Header ------------------
st.markdown(
    """
    <div class="header">
      <div>
        <div class="title">🛡 Fraud Detection System Using Machine Learning</div>
        <div class="subtitle">Secure Banking | Real-Time Analysis</div>
        <div class="accent-bar" style="width:260px"></div>
      </div>
      <div style="text-align:right">
        <div class="muted small">Premium UI: Glass + Gradient • Animated charts</div>
        <div style="height:6px"></div>
        <div class="muted small">Model: RandomForest (brief & interpretable)</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
# ------------------ Layout: Left column for dataset & charts, Right for model & prediction ------------------
left, right = st.columns([2, 1])

# ------------------ Left: Upload / Dataset Preview / Charts ------------------
with left:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("📂 Upload or Choose Dataset")
    uploaded = st.file_uploader("Upload CSV (fraud dataset with 'isFraud' column) — or skip to use demo data", type=["csv"])
    df = None

    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.success("Dataset uploaded ✅")
        except Exception as e:
            st.error("Error reading CSV — make sure file is valid.")
            st.write(e)

    # Provide a realistic fallback demo dataset if user doesn't upload
    if df is None:
        st.info("Using built-in demo dataset (you can still upload your own).")
        np.random.seed(42)
        n = 1000
        df = pd.DataFrame({
            "amount": np.random.exponential(scale=2000, size=n).round(2),
            "type": np.random.choice(["Transfer","Cash-out","Payment","Debit","Credit"], size=n, p=[0.25,0.2,0.3,0.15,0.1]),
            "oldbalanceOrg": np.random.uniform(100,50000, size=n).round(2),
            "oldbalanceDest": np.random.uniform(0,50000, size=n).round(2),
            "step": np.random.randint(1,50,size=n)
        })
        # synthetic isFraud: bigger amount & Receiver unknown pattern
        df["isFraud"] = ((df["amount"] > 8000) & (df["oldbalanceDest"] < 1000)).astype(int)
        # convert some to fraud
        df.loc[df.sample(frac=0.03, random_state=2).index, "isFraud"] = 1

    # Show preview with colored fraud rows
    st.write("### Dataset Preview")
    def color_fraud(r):
        return ["background-color:#2a0b0b" if r["isFraud"]==1 else "" for _ in r]
    # Elegant display: show columns we mentioned in UI only
    preview_cols = ["amount", "type", "oldbalanceOrg", "oldbalanceDest", "step", "isFraud"]
    to_preview = df[preview_cols].head(10)
    st.dataframe(to_preview.style.apply(color_fraud, axis=1))
    st.markdown('</div>', unsafe_allow_html=True)

    # Charts container
    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("📊 Visuals")

    # Pie chart: Fraud vs Safe
    counts = df["isFraud"].value_counts().rename({0:"Safe",1:"Fraud"})
    fig_pie = px.pie(names=counts.index, values=counts.values, title="Fraud vs Safe", hole=0.45)
    fig_pie.update_traces(textposition='inside', textinfo='percent+label', hovertemplate='%{label}: %{value}')
    st.plotly_chart(fig_pie, use_container_width=True)

    # Bar chart: top fraudulent amounts
    fraud_amounts = df[df["isFraud"]==1].nlargest(30, "amount")
    if not fraud_amounts.empty:
        fig_bar = px.bar(fraud_amounts, x=fraud_amounts.index.astype(str), y="amount",
                         title="Top Fraudulent Transaction Amounts (sample)",
                         labels={"x":"Sample ID","amount":"Amount (₹)"})
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No fraud samples available to show a bar chart.")

    # Animated chart: fraud count by step (time)
    agg = df.groupby("step")["isFraud"].sum().reset_index()
    fig_line = px.line(agg, x="step", y="isFraud", title="Fraud occurrences over steps (time)", markers=True)
    fig_line.update_layout(yaxis_title="Number of Fraud")
    st.plotly_chart(fig_line, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ------------------ Right: Model Training and Prediction ------------------
with right:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("🤖 Model Training")

    # Model parameters (simple)
    n_estimators = st.selectbox("RandomForest estimators", options=[50, 100, 200, 300], index=2)
    test_size = st.slider("Test set (%)", min_value=10, max_value=40, value=30, step=5)

    train_button = st.button("Train Model")

    if train_button:
        # prepare dataset for training - keep only numerical features we know, plus encode type
        df_train = df.copy()
        # encode 'type' as simple one-hot
        df_train = pd.get_dummies(df_train, columns=["type"], drop_first=True)
        # ensure target exists
        if "isFraud" not in df_train.columns:
            st.error("Dataset must contain 'isFraud' column.")
        else:
            features = [c for c in df_train.columns if c != "isFraud"]
            X = df_train[features].fillna(0)
            y = df_train["isFraud"]

            t0 = time.time()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42, stratify=y if y.nunique()>1 else None)
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
            model.fit(X_train, y_train)
            t1 = time.time()

            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, zero_division=0)
            rec = recall_score(y_test, preds, zero_division=0)

            # store in session
            st.session_state["model"] = model
            st.session_state["features"] = features

            st.success(f"Model trained successfully in {(t1-t0):.2f}s")
            st.markdown(f"*Accuracy:* {acc*100:.2f}%  •  *Precision:* {prec*100:.2f}%  •  *Recall:* {rec*100:.2f}%")
            st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
            # show simple feature importances (top 6)
            importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False).head(6)
            fig_imp = px.bar(importances, x=importances.values, y=importances.index, orientation='h', labels={'x':'Importance','y':'Feature'}, title="Top Feature Importances")
            st.plotly_chart(fig_imp, use_container_width=True)

  # Show small model status
    if "model" in st.session_state:
        st.markdown('<div class="muted small">Model ready for predictions ✅</div>')
    else:
        st.markdown('<div class="muted small">Model not trained yet. Use Train Model to create a model (demo dataset works).</div>')

    st.markdown('</div>', unsafe_allow_html=True)


    # --------------- Prediction Card ---------------
    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("🔍 Predict a New Transaction")

    # Inputs as per your spec (no confusing newbalance fields; calculated automatically)
    amount = st.number_input("Amount (₹)", min_value=0.0, value=1500.0, step=50.0, format="%.2f")
    tx_type = st.selectbox("Transaction Type", ["Transfer", "Cash-out", "Payment", "Debit", "Credit"])
    sender_bal = st.number_input("Sender Balance Before (₹)", min_value=0.0, value=5000.0, step=50.0, format="%.2f")
    receiver_bal = st.number_input("Receiver Balance Before (₹)", min_value=0.0, value=100.0, step=10.0, format="%.2f")
    step = st.slider("Transaction Step (time)", min_value=int(df["step"].min()), max_value=int(df["step"].max()), value=int(df["step"].min()))
    receiver_type = st.radio("Receiver Account Type", ["Verified", "Unknown"])
    past_history = st.checkbox("Past fraud history on sender", value=False)

    # automatic balance calculation (shown)
    new_sender_balance = sender_bal - amount
    new_receiver_balance = receiver_bal + amount
    st.markdown(f"<div class='small muted'>Calculated: Sender new balance = ₹{new_sender_balance:.2f}  •  Receiver new balance = ₹{new_receiver_balance:.2f}</div>", unsafe_allow_html=True)

    # Predict
    if st.button("Predict Fraud 🚨"):

        if "model" not in st.session_state:
            st.error("Model not trained yet. Train the model first (or press Train Model on right).")
        else:
            # Create feature vector consistent with training features
            features = st.session_state["features"]
            row = {}

            # base numeric fields if present in training features
            # try to map 'amount','oldbalanceOrg','oldbalanceDest','step' presence
            row["amount"] = amount if "amount" in features else 0
            row["oldbalanceOrg"] = sender_bal if "oldbalanceOrg" in features else 0
            row["oldbalanceDest"] = receiver_bal if "oldbalanceDest" in features else 0
            row["step"] = step if "step" in features else 0

            # receiver_type & past_history as simple features
            row["receiver_unknown"] = 1 if receiver_type == "Unknown" else 0
            row["past_history"] = 1 if past_history else 0

            # one-hot encode tx_type consistent with training columns (we used get_dummies drop_first)
            for t in ["Cash-out","Credit","Debit","Payment","Transfer"]:
                colname = f"type_{t}"
                if any(c.startswith("type_") for c in features):
                    # training used columns like 'type_Cash-out' etc (drop_first=True)
                    row[colname] = 1 if tx_type == t else 0

            # Build a feature vector aligned to features
            Xrow = []
            for f in features:
                Xrow.append(row.get(f, 0))

            model = st.session_state["model"]
            probs = model.predict_proba([Xrow])[0]
            fraud_prob = probs[1] * 100  # percent

            # Risk level
            if fraud_prob >= 75:
                risk = "High"
            elif fraud_prob >= 30:
                risk = "Medium"
            else:
                risk = "Low"

            # Decision suggestions & reason patterns (heuristic)
            reasons = []
            if amount > df["amount"].mean() + 2 * df["amount"].std():
                reasons.append("Amount unusually high")
            if receiver_type == "Unknown":
                reasons.append("Receiver unknown")
            if step < 5:
                reasons.append("Multiple rapid transactions (short time window)")
            if past_history:
                reasons.append("Sender has past fraud history")

            # Output card
            st.markdown("### 📍 Prediction Result")
            if fraud_prob >= 50:
                st.markdown(f"<div style='padding:10px;border-radius:10px;background:#2b0b0b;border:1px solid rgba(255,80,80,0.12)'><h3 style='color:#ffb4b4'>🚨 Fraud Detected</h3><b>Fraud Probability:</b> {fraud_prob:.2f}%  •  <b>Risk Level:</b> {risk}</div>", unsafe_allow_html=True)
                st.markdown("*Decision Suggestion:* ACTION NEEDED: block account & verify identity.", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='padding:10px;border-radius:10px;background:rgba(8,50,20,0.18);border:1px solid rgba(80,255,150,0.08)'><h3 style='color:#b4ffda'>✅ Safe Transaction</h3><b>Fraud Probability:</b> {fraud_prob:.2f}%  •  <b>Risk Level:</b> {risk}</div>", unsafe_allow_html=True)
                st.markdown("*Suggestion:* Allow transaction; monitor sender behavior.", unsafe_allow_html=True)

            # Show reasons
            st.write("#### Reason Patterns:")
            if reasons:
                for r in reasons:
                    st.write(f"- {r}")
            else:
                st.write("- Matches normal transaction patterns")

            # Probability bar (plotly gauge-like bar)
            fig = go.Figure(go.Bar(marker_color=['#ff4b4b'], x=[fraud_prob], y=['Fraud Risk'], orientation='h', width=0.4, text=[f"{fraud_prob:.1f}%"], textposition='inside'))
            fig.update_layout(xaxis=dict(range=[0,100], showticklabels=False), height=120, margin=dict(l=0,r=0,t=10,b=10))
            st.plotly_chart(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Small sample fraud history table (as per your example)
    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("📜 Fraud History Samples")
    samples = df[df["isFraud"]==1].head(6).copy()
    if not samples.empty:
        samples_table = samples.reset_index().rename(columns={"index":"ID","amount":"Amount","type":"Type","isFraud":"Status"})
        # Create display-friendly ID and Status
        samples_table["ID"] = samples_table["ID"].apply(lambda x: f"{x:05d}")
        samples_table["Status"] = samples_table["Status"].apply(lambda v: "🚨 Fraud" if v==1 else "✔ Safe")
        display_cols = ["ID","Amount","Type","Status"]
        st.table(samples_table[display_cols].assign(Amount=lambda d: "₹" + d["Amount"].round(2).astype(str)))
    else:
        st.info("No fraud samples in dataset.")
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------ Footer spacing ------------------
st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)