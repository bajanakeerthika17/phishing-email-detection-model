"""
app.py
------
Streamlit Web Application: 🛡️ Phishing Email Detection System
A professional, modern machine learning dashboard for detecting phishing emails,
performing safe static URL inspection, and exploring ML performance metrics.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st

from predict import predict_email, load_model_bundle
from feature_extraction import extract_email_url_features, extract_linguistic_features
from preprocessing import clean_text

# -----------------------------------------------------------------------------
# 1. Page Configuration & White/Blue Modern Theme Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Phishing Email Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom White & Blue Theme CSS
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main Container Background */
    .stApp {
        background-color: #F8FAFC;
    }

    /* Header styling */
    .hero-header {
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 50%, #3B82F6 100%);
        border-radius: 16px;
        padding: 32px 36px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(37, 99, 235, 0.25);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        opacity: 0.92;
        margin-top: 8px;
        font-weight: 400;
        max-width: 800px;
    }

    /* Metric cards */
    .metric-card {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px 22px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 14px rgba(37, 99, 235, 0.08);
    }
    .metric-number {
        font-size: 2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-top: 4px;
    }
    .metric-label {
        font-size: 0.88rem;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    /* Result Banners */
    .banner-phishing {
        background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
        border: 2px solid #EF4444;
        border-radius: 14px;
        padding: 24px;
        color: #991B1B;
        margin-top: 18px;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.12);
    }
    .banner-safe {
        background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%);
        border: 2px solid #22C55E;
        border-radius: 14px;
        padding: 24px;
        color: #166534;
        margin-top: 18px;
        box-shadow: 0 4px 12px rgba(34, 197, 94, 0.12);
    }
    .banner-title {
        font-size: 1.6rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 6px;
    }

    /* URL Badges */
    .url-badge-high {
        background-color: #FEE2E2;
        color: #B91C1C;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.82rem;
        border: 1px solid #FCA5A5;
        display: inline-block;
    }
    .url-badge-medium {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.82rem;
        border: 1px solid #FCD34D;
        display: inline-block;
    }
    .url-badge-low {
        background-color: #DCFCE7;
        color: #15803D;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.82rem;
        border: 1px solid #86EFAC;
        display: inline-block;
    }

    /* Info callout */
    .info-disclaimer {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        color: #1E40AF;
        font-size: 0.9rem;
        margin: 16px 0;
    }

    /* Section card container */
    .section-card {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 22px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03);
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. Resource Caching & Artifact Loading
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
IMAGES_DIR = os.path.join(BASE_DIR, "images")
PRIMARY_MODEL_PATH = os.path.join(MODELS_DIR, "phishing_model.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "model_metrics.json")
DATASET_PATH = os.path.join(BASE_DIR, "phishing_email_dataset.csv")

@st.cache_resource
def get_artifacts():
    """
    Loads model bundle and metrics. Trains automatically if files are missing.
    """
    if not os.path.exists(PRIMARY_MODEL_PATH):
        from train_model import train_and_evaluate
        bundle = train_and_evaluate()
    else:
        bundle = joblib.load(PRIMARY_MODEL_PATH)

    metrics = {}
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        except Exception:
            metrics = bundle.get("results", {})
    else:
        metrics = bundle.get("results", {})

    return bundle, metrics


@st.cache_data
def load_dataset():
    """Loads the phishing email dataset for dashboard exploration."""
    if os.path.exists(DATASET_PATH):
        return pd.read_csv(DATASET_PATH)
    elif os.path.exists(os.path.join(BASE_DIR, "dataset.csv")):
        return pd.read_csv(os.path.join(BASE_DIR, "dataset.csv"))
    return pd.DataFrame()


bundle, all_metrics = get_artifacts()
df_data = load_dataset()

best_model_name = bundle.get("best_model_name", "Naive Bayes")
best_model = bundle.get("best_model")
vectorizer = bundle.get("vectorizer")
best_stats = all_metrics.get(best_model_name, {
    "accuracy": 0.9643,
    "precision": 0.9412,
    "recall": 1.0000,
    "f1_score": 0.9697
})


# -----------------------------------------------------------------------------
# 3. Sidebar Navigation & System Information
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.title("Phishing Detector")
    st.caption("AI-Powered Cyber Defense System")

    st.markdown("---")
    menu = st.radio(
        "Navigation",
        [
            "🔍 Email & URL Detector",
            "📊 Model Performance & Charts",
            "📈 Dataset Dashboard",
            "📖 Phishing Awareness Guide"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown("### ⚙️ Active Model Info")
    st.info(f"**Algorithm:** {best_model_name}\n\n"
            f"**F1-Score:** {best_stats.get('f1_score', 0) * 100:.1f}%\n\n"
            f"**Accuracy:** {best_stats.get('accuracy', 0) * 100:.1f}%\n\n"
            f"**Vocabulary:** {bundle.get('feature_count', 2200)} n-grams")

    st.markdown("---")
    st.markdown(
        "<div style='font-size: 0.8rem; color: #64748B;'>"
        "<strong>Note:</strong> Statistical ML classification estimate. Always verify suspicious communications."
        "</div>",
        unsafe_allow_html=True
    )


# -----------------------------------------------------------------------------
# 4. View 1: Email & URL Detector (Core Feature)
# -----------------------------------------------------------------------------
if menu == "🔍 Email & URL Detector":
    # Header Banner
    st.markdown("""
    <div class="hero-header">
        <h1 class="hero-title">🛡️ Phishing Email Detection System</h1>
        <p class="hero-subtitle">
            Inspect email content and embedded links using machine learning text vectorization
            and static heuristic URL analysis to identify malicious communications.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Pre-filled Samples Section
    st.markdown("#### ⚡ Quick Test Samples")
    st.caption("Click any button below to instantly load a realistic test email into the scanner:")

    sample_col1, sample_col2, sample_col3, sample_col4 = st.columns(4)

    sample_1_text = (
        "URGENT: Your Chase Bank account has been temporarily suspended due to unauthorized access attempts "
        "from IP 194.26.29.112. Click here immediately to verify your identity and restore access: "
        "http://chase-security-verify-login.xyz/auth"
    )
    sample_2_text = (
        "Final Notice: Your Microsoft 365 corporate password expires in 2 hours! To keep your current password, "
        "please validate your account credentials at: http://login-microsoft365-verify.net/auth"
    )
    sample_3_text = (
        "Hi Team, please find attached the agenda for our sprint retrospective meeting tomorrow at 10:00 AM. "
        "Let me know if you would like to add any discussion topics to the board before then."
    )
    sample_4_text = (
        "GitHub Notification: Pull request #184 (Feature: Add password hashing utility) has been approved by "
        "senior engineer. All automated CI/CD unit tests passed successfully. Ready for merge to main branch."
    )

    if "email_input_area" not in st.session_state:
        st.session_state["email_input_area"] = ""

    with sample_col1:
        if st.button("🚨 Bank Suspension Scam", use_container_width=True):
            st.session_state["email_input_area"] = sample_1_text

    with sample_col2:
        if st.button("⚠️ Microsoft 365 Password Scam", use_container_width=True):
            st.session_state["email_input_area"] = sample_2_text

    with sample_col3:
        if st.button("✅ Safe Sprint Meeting", use_container_width=True):
            st.session_state["email_input_area"] = sample_3_text

    with sample_col4:
        if st.button("✅ Safe GitHub PR Alert", use_container_width=True):
            st.session_state["email_input_area"] = sample_4_text

    st.markdown("<br>", unsafe_allow_html=True)

    # Input Area
    user_email = st.text_area(
        "Paste Email Subject & Content Below:",
        value=st.session_state["email_input_area"],
        height=170,
        placeholder="Paste full email text or subject line here (e.g., 'URGENT: Your account has been locked...')"
    )

    col_btn1, col_btn2 = st.columns([1, 5])
    with col_btn1:
        predict_clicked = st.button("🔍 Scan & Analyze Email", type="primary", use_container_width=True)
    with col_btn2:
        if st.button("🧹 Clear Text", use_container_width=False):
            st.session_state["email_input_area"] = ""
            st.rerun()

    # Trigger Analysis
    if predict_clicked:
        if not user_email or not user_email.strip():
            st.warning("⚠️ Please paste or type an email before clicking Predict.")
        else:
            with st.spinner("Analyzing email semantics, linguistic patterns, and embedded URLs..."):
                result = predict_email(user_email, best_model, vectorizer)

            if result["prediction"] == "Invalid Input":
                st.error(f"❌ {result.get('explanation', 'Invalid input provided.')}")
            else:
                pred = result["prediction"]
                conf = result["confidence"]
                phish_prob = result["phishing_prob"]
                safe_prob = result["safe_prob"]
                url_data = result["url_analysis"]
                ling_data = result["linguistic_analysis"]

                # Result Banner
                if pred == "Phishing":
                    st.markdown(f"""
                    <div class="banner-phishing">
                        <div class="banner-title">
                            🚨 PHISHING DETECTED
                        </div>
                        <p style="margin: 0; font-size: 1.05rem;">
                            This email exhibits classic social-engineering characteristics associated with phishing, credential harvesting, or financial scams.
                        </p>
                        <div style="margin-top: 14px; font-weight: 600; font-size: 1.1rem;">
                            Model Confidence: <span style="font-size: 1.35rem; color: #DC2626;">{conf:.1f}%</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="banner-safe">
                        <div class="banner-title">
                            ✅ SAFE / LEGITIMATE EMAIL
                        </div>
                        <p style="margin: 0; font-size: 1.05rem;">
                            No significant phishing indicators or malicious signatures were detected in this email.
                        </p>
                        <div style="margin-top: 14px; font-weight: 600; font-size: 1.1rem;">
                            Model Confidence: <span style="font-size: 1.35rem; color: #16A34A;">{conf:.1f}%</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Caution flag if high risk URLs found in safe email
                if result.get("warning_flag"):
                    st.warning(f"⚠️ {result['warning_flag']}")

                # Explanation & Probability Progress
                st.markdown("""
                <div class="info-disclaimer">
                    ℹ️ <strong>Model Confidence Interpretation:</strong> The percentage above represents the machine learning model's 
                    statistical probability estimate calculated via TF-IDF vectorization. It is not an absolute security guarantee. 
                    Always verify unexpected emails through known official channels.
                </div>
                """, unsafe_allow_html=True)

                # Probability Breakdown Bars
                prob_col1, prob_col2 = st.columns(2)
                with prob_col1:
                    st.markdown(f"**Phishing Probability: `{phish_prob:.1f}%`**")
                    st.progress(min(float(phish_prob) / 100.0, 1.0))
                with prob_col2:
                    st.markdown(f"**Safe / Legitimate Probability: `{safe_prob:.1f}%`**")
                    st.progress(min(float(safe_prob) / 100.0, 1.0))

                st.markdown("<br>", unsafe_allow_html=True)

                # Feature 2: Static URL Analysis Section
                st.markdown("### 🔗 Static URL Security Inspection")
                urls_list = url_data.get("analyzed_urls", [])

                if len(urls_list) == 0:
                    st.success("✅ **No external URLs detected.** The email does not contain any links.")
                else:
                    st.info(f"🔎 Found **{len(urls_list)} link(s)**. Performing safe static inspection (without visiting or querying):")
                    
                    for idx, u in enumerate(urls_list, start=1):
                        risk_level = u["risk_level"]
                        badge_class = (
                            "url-badge-high" if risk_level == "High Risk"
                            else "url-badge-medium" if risk_level == "Medium Risk"
                            else "url-badge-low"
                        )

                        with st.expander(f"Link #{idx}: {u['url']} — {risk_level}", expanded=(risk_level != "Low Risk")):
                            st.markdown(f"<span class='{badge_class}'>{risk_level} (Score: {u['risk_score']}/100)</span>", unsafe_allow_html=True)
                            
                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("URL Length", f"{u['length']} chars")
                            c2.metric("Protocol", "HTTPS" if u["is_https"] else "HTTP (Insecure)")
                            c3.metric("Raw IP Address", "YES (Risky)" if u["has_ip"] else "No (Domain)")
                            c4.metric("Subdomain Dots", f"{u['num_dots']}")

                            if u["flags"]:
                                st.markdown("**Identified Security Red Flags:**")
                                for flag in u["flags"]:
                                    st.markdown(f"- ⚠️ {flag}")
                            else:
                                st.markdown("✅ No anomalous URL heuristics detected.")

                # Linguistic Pattern Indicators
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### 📝 Linguistic & Content Signals")
                l1, l2, l3, l4 = st.columns(4)
                l1.metric("Urgency Triggers", f"{ling_data['urgency_count']}")
                l2.metric("Financial Keywords", f"{ling_data['financial_count']}")
                l3.metric("Uppercase Ratio", f"{ling_data['capital_ratio']}%")
                l4.metric("Exclamation Marks", f"{ling_data['exclamation_count']}")

                if ling_data["urgency_keywords"]:
                    st.caption(f"Detected Urgency Words: {', '.join(ling_data['urgency_keywords'])}")
                if ling_data["financial_keywords"]:
                    st.caption(f"Detected Financial Terms: {', '.join(ling_data['financial_keywords'])}")


# -----------------------------------------------------------------------------
# 5. View 2: Model Performance & Comparative Charts
# -----------------------------------------------------------------------------
elif menu == "📊 Model Performance & Charts":
    st.markdown("""
    <div class="hero-header">
        <h1 class="hero-title">📊 Machine Learning Model Performance</h1>
        <p class="hero-subtitle">
            Evaluation metrics, confusion matrix, and performance comparison across
            Logistic Regression, Naive Bayes, and Random Forest classifiers.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 4 Key Metrics Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Model Accuracy</div>
            <div class="metric-number">{best_stats.get('accuracy', 0) * 100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Precision Score</div>
            <div class="metric-number">{best_stats.get('precision', 0) * 100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Recall Score</div>
            <div class="metric-number">{best_stats.get('recall', 0) * 100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">F1-Score</div>
            <div class="metric-number">{best_stats.get('f1_score', 0) * 100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Comparative Performance Table
    st.markdown("### 🏆 Algorithm Comparison Table")
    table_rows = []
    for model_name, stats in all_metrics.items():
        is_best = " ⭐ (Selected)" if model_name == best_model_name else ""
        table_rows.append({
            "Algorithm": model_name + is_best,
            "Accuracy": f"{stats['accuracy'] * 100:.2f}%",
            "Precision": f"{stats['precision'] * 100:.2f}%",
            "Recall": f"{stats['recall'] * 100:.2f}%",
            "F1-Score": f"{stats['f1_score'] * 100:.2f}%"
        })
    df_perf = pd.DataFrame(table_rows)
    st.dataframe(df_perf, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Graphs side by side
    col_graph1, col_graph2 = st.columns([1, 1])

    with col_graph1:
        st.markdown("### 🎯 Confusion Matrix")
        cm_path = os.path.join(IMAGES_DIR, "confusion_matrix.png")
        if os.path.exists(cm_path):
            st.image(cm_path, use_container_width=True, caption=f"Confusion Matrix: {best_model_name}")
        else:
            st.info("Confusion matrix plot generating...")

    with col_graph2:
        st.markdown("### 📈 Algorithm Benchmark Chart")
        comp_path = os.path.join(IMAGES_DIR, "model_comparison.png")
        if os.path.exists(comp_path):
            st.image(comp_path, use_container_width=True, caption="Accuracy, Precision, Recall & F1 across Models")
        else:
            st.info("Comparison chart generating...")

    st.markdown("<br>", unsafe_allow_html=True)

    # Detailed Classification Report
    st.markdown("### 📋 Detailed Classification Report")
    with st.expander("View Full Text Classification Report", expanded=True):
        st.code(best_stats.get("classification_report_text", "Report unavailable."), language="text")


# -----------------------------------------------------------------------------
# 6. View 3: Dataset Dashboard
# -----------------------------------------------------------------------------
elif menu == "📈 Dataset Dashboard":
    st.markdown("""
    <div class="hero-header">
        <h1 class="hero-title">📈 Dataset Explorer & Dashboard</h1>
        <p class="hero-subtitle">
            Explore the training dataset distribution, email categories, and sample records
            used to build and benchmark the phishing detection model.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if not df_data.empty:
        total_records = len(df_data)
        phishing_count = int((df_data["label"] == 1).sum())
        safe_count = int((df_data["label"] == 0).sum())

        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Records</div>
                <div class="metric-number">{total_records}</div>
            </div>
            """, unsafe_allow_html=True)
        with d2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Phishing Emails</div>
                <div class="metric-number" style="color: #DC2626;">{phishing_count}</div>
            </div>
            """, unsafe_allow_html=True)
        with d3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Safe Emails</div>
                <div class="metric-number" style="color: #16A34A;">{safe_count}</div>
            </div>
            """, unsafe_allow_html=True)
        with d4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Selected Model</div>
                <div class="metric-number" style="font-size: 1.35rem; color: #2563EB;">{best_model_name}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Category Breakdown
        if "category" in df_data.columns:
            st.markdown("### 🏷️ Threat & Category Distribution")
            cat_counts = df_data["category"].value_counts().reset_index()
            cat_counts.columns = ["Category", "Count"]
            st.bar_chart(data=cat_counts.set_index("Category"))

        st.markdown("<br>", unsafe_allow_html=True)

        # Interactive Table
        st.markdown("### 📄 Browse Dataset Records")
        filter_option = st.selectbox("Filter by Label:", ["All Emails", "Phishing Only (Label: 1)", "Safe Only (Label: 0)"])
        
        filtered_df = df_data.copy()
        if filter_option == "Phishing Only (Label: 1)":
            filtered_df = filtered_df[filtered_df["label"] == 1]
        elif filter_option == "Safe Only (Label: 0)":
            filtered_df = filtered_df[filtered_df["label"] == 0]

        st.dataframe(filtered_df, use_container_width=True, hide_index=True)
    else:
        st.warning("No dataset file found at `phishing_email_dataset.csv`.")


# -----------------------------------------------------------------------------
# 7. View 4: Phishing Awareness Guide
# -----------------------------------------------------------------------------
elif menu == "📖 Phishing Awareness Guide":
    st.markdown("""
    <div class="hero-header">
        <h1 class="hero-title">📖 Cyber Security & Phishing Awareness</h1>
        <p class="hero-subtitle">
            Learn common phishing attack techniques, key indicators of compromise,
            and best practices to protect organizational and personal digital assets.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.markdown("### 🚩 Common Phishing Red Flags")
        st.markdown("""
        - **Artificial Urgency & Fear Appeals:** Phishing attacks routinely threaten imminent account deactivation, legal action, or bank lockouts within tight windows (e.g., *'within 2 hours'*).
        - **Mismatched Sender & Link Domains:** The email might display a trusted brand (e.g. *PayPal*, *Microsoft*), but hover over the link reveals an unrelated domain (e.g. `paypal-verify.xyz`).
        - **Unusual Requests for Gift Cards or Wire Transfers:** CEO fraud or BEC attacks often claim the executive is in a confidential meeting and needs urgent gift cards or wires.
        - **Generic Greetings:** *'Dear Customer'* or *'Valued Employee'* instead of your actual name or internal username.
        - **Suspicious Top-Level Domains:** Domains ending in `.xyz`, `.top`, `.cc`, `.buzz` are frequently used in automated phishing infrastructure due to low registration fees.
        """)

    with col_g2:
        st.markdown("### 🛡️ Best Defense Practices")
        st.markdown("""
        - **Inspect URLs Before Clicking:** Always check the exact destination domain without opening the link.
        - **Enable Multi-Factor Authentication (MFA):** Hardware security keys or authenticator apps stop password theft attacks cold.
        - **Verify Out-of-Band:** If an email from an executive or bank requests money or sensitive information, contact them directly on a verified internal channel.
        - **Keep Systems and Browsers Updated:** Modern browsers include built-in reputation filters like Google Safe Browsing and Microsoft SmartScreen.
        - **Report Phishing to IT/Security:** Early reports allow security teams to block malicious sender domains and quarantine emails across the entire company.
        """)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="info-disclaimer">
        ⚖️ <strong>Disclaimer:</strong> This application is built for educational, training, and defensive screening purposes. 
        Automated classification tools should always be paired with human vigilance, zero-trust policies, and multi-layered email defenses.
    </div>
    """, unsafe_allow_html=True)


# Footer
st.markdown("<br><hr style='border: none; border-top: 1px solid #E2E8F0;'>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align: center; color: #94A3B8; font-size: 0.85rem; padding: 12px 0;'>"
    "🛡️ Phishing Email Detection System &bull; Machine Learning with Scikit-Learn & Streamlit &bull; 2026"
    "</div>",
    unsafe_allow_html=True
)
