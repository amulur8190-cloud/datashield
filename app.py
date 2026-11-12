import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix

from core.data_loader import load_and_split
from core.attacks import run_secure
from core.federated import init_weights, local_train, evaluate, predict_proba
from core.privacy import run_private

st.set_page_config(page_title="DATASHIELD", page_icon="🛡️", layout="wide")

# ---------- STYLE ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(160deg, #070d1f 0%, #0b1530 60%, #0d1b3d 100%); color: #e6edf7; }
[data-testid="stSidebar"] { background: #0a1128; border-right: 1px solid #1c2c55; }
#MainMenu, footer { visibility: hidden; }
h1, h2, h3 { color: #e6edf7; font-weight: 800; }
.hero { padding: 34px 38px; border-radius: 18px; margin-bottom: 22px;
  background: radial-gradient(circle at 85% 20%, #1d4ed8 0%, #0f2a6b 40%, #0a1533 100%);
  border: 1px solid #2a4a9a; }
.hero h1 { font-size: 46px; margin: 6px 0; letter-spacing: 2px; color: #fff; }
.hero p { color: #b9c9ec; font-size: 17px; margin: 0; }
.badge { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 12px;
  background: #12306e; color: #7fb6ff; border: 1px solid #2f5bb8; letter-spacing: 1px; }
.card { background: #101c3d; border: 1px solid #243a73; border-radius: 14px;
  padding: 18px 20px; margin-bottom: 12px; }
.card-title { color: #8fa5d4; font-size: 13px; text-transform: uppercase; letter-spacing: 1px; }
.card-value { font-size: 30px; font-weight: 800; margin-top: 4px; white-space: nowrap; }
.card-sub { color: #7d90bb; font-size: 12px; margin-top: 2px; }
.pill { display: inline-block; padding: 3px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; }
.green { background: #0f3d2b; color: #4ade80; border: 1px solid #1f7a52; }
.red { background: #4a1420; color: #ff6b81; border: 1px solid #a02a42; }
.yellow { background: #43360f; color: #fbbf24; border: 1px solid #8a6d1a; }
.flow { display: flex; gap: 10px; flex-wrap: wrap; margin: 8px 0 18px 0; }
.step { flex: 1; min-width: 140px; background: #101c3d; border: 1px solid #243a73;
  border-radius: 12px; padding: 14px; text-align: center; font-size: 13px; color: #cfdcf7; }
.step b { display: block; color: #4da3ff; font-size: 22px; }
div.stButton > button { background: linear-gradient(90deg, #2563eb, #1d4ed8); color: white;
  border: none; border-radius: 10px; padding: 10px 26px; font-weight: 600; }
div.stButton > button:hover { background: linear-gradient(90deg, #3b82f6, #2563eb); color: white; }
</style>
""", unsafe_allow_html=True)


# ---------- HELPERS ----------
def card(title, value, sub="", color="#4da3ff"):
    st.markdown(f'<div class="card"><div class="card-title">{title}</div>'
                f'<div class="card-value" style="color:{color}">{value}</div>'
                f'<div class="card-sub">{sub}</div></div>', unsafe_allow_html=True)


def pill(text, kind):
    return f'<span class="pill {kind}">{text}</span>'


def explain(text):
    st.markdown(f'<div class="card" style="border-left:4px solid #4da3ff">'
                f'<div class="card-title">💡 What judges should know</div>'
                f'<div style="color:#cfdcf7;font-size:14px;margin-top:6px">{text}</div></div>',
                unsafe_allow_html=True)


def header(title, subtitle):
    st.markdown(f"## {title}")
    st.caption(subtitle)


def style_fig(fig):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter"),
                      margin=dict(l=10, r=10, t=50, b=10))
    return fig


@st.cache_data
def get_data():
    return load_and_split()


banks, X_test, y_test = get_data()


@st.cache_data
def local_baselines():
    out = {}
    for i, (name, (X, y)) in enumerate(banks.items()):
        out[name] = local_train(init_weights(X.shape[1]), X, y, epochs=30, seed=i)
    return out


for key, default in [("results", {}), ("runs", []), ("privacy", None), ("check", None)]:
    if key not in st.session_state:
        st.session_state[key] = default

# ---------- SIDEBAR ----------
st.sidebar.markdown("# 🛡️ DATASHIELD")
st.sidebar.caption("Privacy-Preserving Distributed Intelligence")
page = st.sidebar.radio("Navigation", [
    "Home", "Organizations", "Federated Training", "Security Monitoring",
    "Privacy Protection", "Model Evaluation", "Live Fraud Check", "Training History"],
    label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.caption("AARAMBH 2026 · PS09 · REVA University")

res = st.session_state.results

# ================= HOME =================
if page == "Home":
    st.markdown('<div class="hero"><span class="badge">AARAMBH 2026 · PS09</span>'
                '<h1>🛡️ DATASHIELD</h1>'
                '<p>Banks learn together without sharing data, and the platform catches '
                'any participant that cheats.</p></div>', unsafe_allow_html=True)

    last_f1 = f"{res['hist'][-1]['f1']:.3f}" if res else "Pending"
    c1, c2, c3, c4 = st.columns(4)
    with c1: card("Organizations", len(banks), "Simulated banks")
    with c2: card("Training rounds", 10, "Per federated run")
    with c3: card("Latest global F1", last_f1, "Measured, not hard-coded", "#38bdf8")
    with c4:
        if not res:
            card("Security status", "Ready", "Run training to evaluate", "#fbbf24")
        elif res["attack"] and not res["defence"]:
            card("Security status", "AT RISK", "Attack with no defence", "#ff6b81")
        else:
            card("Security status", "PROTECTED", "Defence active", "#4ade80")

    explain("Organizations = the simulated banks. Training rounds = how many times the banks and "
            "the server exchange updates. Latest global F1 = measured from your last run, not typed in. "
            "Security status changes to PROTECTED or AT RISK based on the last run.")

    st.markdown("#### How it works")
    st.markdown('<div class="flow">'
                '<div class="step"><b>1</b>Server sends global model</div>'
                '<div class="step"><b>2</b>Each bank trains on its own data</div>'
                '<div class="step"><b>3</b>Only weights are shared</div>'
                '<div class="step"><b>4</b>Server filters bad updates</div>'
                '<div class="step"><b>5</b>Server averages into new model</div>'
                '</div>', unsafe_allow_html=True)

    st.markdown("#### What is real and what is simplified")
    st.dataframe(pd.DataFrame([
        {"Mechanism": "Federated learning (FedAvg)", "Status": "Implemented"},
        {"Mechanism": "Model poisoning attack simulation", "Status": "Implemented"},
        {"Mechanism": "Outlier-based malicious update detection", "Status": "Implemented"},
        {"Mechanism": "Differential privacy (clipping + Gaussian noise)", "Status": "Implemented (no formal epsilon)"},
        {"Mechanism": "Secure aggregation (pairwise masking)", "Status": "Simplified demonstration"},
        {"Mechanism": "Homomorphic encryption", "Status": "Future scope"},
    ]), hide_index=True)

# ================= ORGANIZATIONS =================
elif page == "Organizations":
    header("Organizations", "Each bank holds its own private data slice. Nothing is sent out.")
    explain("Each bank is a separate slice of a real credit card dataset. A bank trains only on its "
            "own slice. Only model weights leave, never transactions.")
    cols = st.columns(3)
    for col, (name, (X, y)) in zip(cols, banks.items()):
        with col:
            st.markdown(f"### 🏦 {name}")
            card("Transactions", f"{len(y):,}", "Private local dataset")
            card("Fraud cases", int(y.sum()), f"{y.mean() * 100:.3f}% of its data", "#ff6b81")
            st.markdown(pill("Raw data leaves bank: NO", "green") + " " +
                        pill("Status: Ready", "yellow"), unsafe_allow_html=True)

# ================= FEDERATED TRAINING =================
elif page == "Federated Training":
    header("Federated Training", "Local training at each bank, then central aggregation, repeated for 10 rounds.")
    explain("Each round: the server sends the global model, each bank trains locally, the server "
            "filters bad updates and averages the rest (FedAvg). Toggle the attack and defence to "
            "see the difference.")
    c1, c2 = st.columns(2)
    attack = c1.toggle("Simulate malicious Bank Gamma")
    defence = c2.toggle("Enable robust defence", value=True)

    if st.button("▶ Start Training"):
        with st.spinner("3 banks are training locally... server is aggregating..."):
            w, hist, log = run_secure(banks, X_test, y_test, attack=attack, defence=defence)
        st.session_state.results = {"w": w, "hist": hist, "log": log,
                                    "attack": attack, "defence": defence}
        scen = ("Attack" if attack else "Normal") + (" + Defence" if defence else " (no defence)")
        st.session_state.runs.append({"name": f"Run {len(st.session_state.runs) + 1}",
                                      "scenario": scen, "hist": hist, "log": log})
        res = st.session_state.results
        st.success("Training complete. Global model updated.")

    if res:
        h = pd.DataFrame(res["hist"])
        last = h.iloc[-1]
        a, b, c, d = st.columns(4)
        with a: card("Precision", f"{last['precision']:.3f}")
        with b: card("Recall", f"{last['recall']:.3f}")
        with c: card("F1-score", f"{last['f1']:.3f}")
        with d: card("Final loss", f"{last['loss']:.4f}", color="#fbbf24")
        l, r = st.columns(2)
        l.plotly_chart(style_fig(px.line(h, x="round", y="loss", markers=True,
                                         title="Training loss per round")))
        r.plotly_chart(style_fig(px.line(h, x="round", y=["precision", "recall", "f1"],
                                         markers=True, title="Global model metrics per round")))
    else:
        st.info("Choose your options and click Start Training.")

# ================= SECURITY =================
elif page == "Security Monitoring":
    header("Security Monitoring", "Poisoned updates are detected by their distance from the typical update.")
    explain("Honest banks send similar updates. The server measures how far each update is from the "
            "typical one and blocks any that is more than 3 times farther. This needs an honest majority.")
    if not res or not res["log"]:
        st.warning("Run training with the defence enabled first (Federated Training page).")
    else:
        log = pd.DataFrame(res["log"])
        cols = st.columns(3)
        for col, name in zip(cols, banks.keys()):
            n_blocked = int((log[log["bank"] == name]["status"] == "BLOCKED").sum())
            with col:
                card(name, "BLOCKED" if n_blocked else "TRUSTED",
                     f"{n_blocked} blocked update(s) in 10 rounds",
                     "#ff6b81" if n_blocked else "#4ade80")
        blocked = log[log["status"] == "BLOCKED"]
        if len(blocked):
            st.error(f"🚨 SECURITY ALERT: {', '.join(sorted(blocked['bank'].unique()))} "
                     f"sent suspicious updates and was excluded from aggregation.")
        else:
            st.success("No suspicious updates detected. All participants trusted.")
        st.plotly_chart(style_fig(px.bar(log, x="round", y="distance", color="bank",
                                         barmode="group", title="Update distance from the typical update")))
        st.markdown("#### Security log")
        st.dataframe(log, hide_index=True)

# ================= PRIVACY =================
elif page == "Privacy Protection":
    header("Privacy Protection", "What leaves a bank is clipped, noised and (in the demo) masked.")
    explain("We clip each update to a maximum size and add random noise, which is the core idea of "
            "differential privacy. More noise means more privacy and usually lower accuracy. The "
            "masking table is a simplified secure aggregation demo.")
    st.markdown("**Protected:** raw transactions never leave any bank. "
                "**Shared:** only model weight updates, which we clip and add noise to.")

    st.markdown("#### 1. Differential privacy experiment (real)")
    clip = st.slider("Clipping bound (max update size)", 0.1, 5.0, 1.0, 0.1)
    if st.button("Run privacy vs accuracy experiment"):
        rows = []
        with st.spinner("Training with different noise levels (about 1 minute)..."):
            for nz in [0.0, 0.02, 0.05, 0.1, 0.2]:
                _, h = run_private(banks, X_test, y_test, rounds=5, clip=clip, noise=nz)
                rows.append({"noise": nz, "precision": h[-1]["precision"],
                             "recall": h[-1]["recall"], "f1": h[-1]["f1"]})
        st.session_state.privacy = pd.DataFrame(rows)
    if st.session_state.privacy is not None:
        p = st.session_state.privacy
        st.plotly_chart(style_fig(px.line(p, x="noise", y=["recall", "f1"], markers=True,
                                          title="More noise = more privacy, usually less accuracy")))
        st.dataframe(p, hide_index=True)
        st.caption("This shows the trade-off. It is not a formal privacy guarantee (no epsilon accounting).")

    st.markdown("#### 2. Secure aggregation demonstration (simplified)")
    base = local_baselines()
    vecs = [w[:5].copy() for w in base.values()]
    masked = [v.copy() for v in vecs]
    rng = np.random.default_rng(7)
    for i in range(3):
        for j in range(i + 1, 3):
            m = rng.normal(0, 10, size=5)
            masked[i] += m
            masked[j] -= m
    demo = pd.DataFrame({"Bank": list(base.keys()),
                         "Real update (first 5 weights)": [str(np.round(v, 2)) for v in vecs],
                         "What the server sees (masked)": [str(np.round(v, 2)) for v in masked]})
    st.dataframe(demo, hide_index=True)
    st.success(f"Sum of real updates: {np.round(sum(vecs), 3)}  |  "
               f"Sum of masked updates: {np.round(sum(masked), 3)}  (identical)")
    st.caption("Each pair of banks adds and subtracts the same random mask, so masks cancel in the sum. "
               "Simplified: no key exchange or dropout handling.")

# ================= EVALUATION =================
elif page == "Model Evaluation":
    header("Model Evaluation", "Individual local models versus the federated global model.")
    explain("Fraud is rare, so accuracy is misleading. Recall is how many real frauds we catch. "
            "Precision is how many of our fraud alerts are correct. F1 combines both.")
    if not res:
        st.warning("Run training first (Federated Training page).")
    else:
        base = local_baselines()
        rows = []
        for name, w in base.items():
            m = evaluate(w, X_test, y_test)
            rows.append({"Model": f"{name} (local only)", "Precision": m["precision"],
                         "Recall": m["recall"], "F1": m["f1"]})
        mg = evaluate(res["w"], X_test, y_test)
        rows.append({"Model": "Federated global", "Precision": mg["precision"],
                     "Recall": mg["recall"], "F1": mg["f1"]})
        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True)
        long = df.melt(id_vars="Model", var_name="Metric", value_name="Score")
        st.plotly_chart(style_fig(px.bar(long, x="Metric", y="Score", color="Model",
                                         barmode="group", title="Local vs global model")))
        pred = (predict_proba(res["w"], X_test) >= 0.5).astype(int)
        cm = confusion_matrix(y_test, pred)
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                        x=["Predicted normal", "Predicted fraud"],
                        y=["Actually normal", "Actually fraud"], title="Confusion matrix (global model)")
        st.plotly_chart(style_fig(fig))
        st.caption("Local models here use the same NumPy training code (30 epochs) for a fair comparison. "
                   "Fraud is rare, so focus on recall and F1, not accuracy.")

# ================= LIVE FRAUD CHECK =================
elif page == "Live Fraud Check":
    header("Live Fraud Check", "Pick a transaction and let the global model decide.")
    explain("The global model from your last training run scores a transaction from the test set "
            "(data no bank trained on). It outputs a fraud probability. Above 50% it is flagged. "
            "Precision is low, so some normal transactions get flagged. Try this after an attack "
            "with no defence to see the damaged model fail.")
    if not res:
        st.warning("Run training first (Federated Training page).")
    else:
        mode = st.radio("Choose a transaction",
                        ["Random normal transaction", "Random fraud transaction",
                         "Enter transaction number"], horizontal=True)
        number = st.number_input(f"Transaction number (0 to {len(y_test) - 1}, "
                                 f"used only for the third option)",
                                 min_value=0, max_value=len(y_test) - 1, value=0, step=1)
        if st.button("🔍 Check transaction"):
            if mode == "Random normal transaction":
                idx = int(np.random.choice(np.where(y_test == 0)[0]))
            elif mode == "Random fraud transaction":
                idx = int(np.random.choice(np.where(y_test == 1)[0]))
            else:
                idx = int(number)
            st.session_state.check = idx

        idx = st.session_state.get("check")
        if idx is not None:
            prob = float(predict_proba(res["w"], X_test[idx:idx + 1])[0])
            flagged = prob >= 0.5
            actual = int(y_test[idx])
            a, b, c = st.columns(3)
            with a: card("Transaction #", idx, "From the shared test set")
            with b: card("Fraud probability", f"{prob * 100:.1f}%",
                         "Model's confidence", "#ff6b81" if flagged else "#4ade80")
            with c: card("Actual label", "FRAUD" if actual else "NORMAL",
                         "Ground truth from the dataset", "#ff6b81" if actual else "#4ade80")
            if flagged and actual:
                st.success("✅ Correct: fraud caught.")
            elif flagged and not actual:
                st.warning("⚠️ False alarm: the model flagged a normal transaction.")
            elif not flagged and actual:
                st.error("❌ Missed fraud.")
            else:
                st.success("✅ Correct: normal transaction passed.")
            st.caption("The 29 input features are anonymized numbers from the dataset (V1 to V28 and Amount), "
                       "so real customer details are never shown.")

# ================= HISTORY =================
elif page == "Training History":
    header("Training History", "Every run in this session, with performance and security logs.")
    explain("Every run is saved for this session so you can compare normal, attacked and defended "
            "training side by side.")
    runs = st.session_state.runs
    if not runs:
        st.warning("No runs yet. Go to Federated Training and start one.")
    else:
        summary = pd.DataFrame([{"Run": r["name"], "Scenario": r["scenario"],
                                 "Precision": r["hist"][-1]["precision"],
                                 "Recall": r["hist"][-1]["recall"],
                                 "F1": r["hist"][-1]["f1"],
                                 "Loss": r["hist"][-1]["loss"]} for r in runs])
        st.dataframe(summary, hide_index=True)
        curves = pd.concat([pd.DataFrame(r["hist"]).assign(Run=f"{r['name']} · {r['scenario']}")
                            for r in runs])
        st.plotly_chart(style_fig(px.line(curves, x="round", y="f1", color="Run",
                                          markers=True, title="F1 per round, all runs")))
        logs = [pd.DataFrame(r["log"]).assign(Run=r["name"]) for r in runs if r["log"]]
        if logs:
            st.markdown("#### Security logs")
            st.dataframe(pd.concat(logs), hide_index=True)