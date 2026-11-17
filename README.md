# 🛡️ Phishing Email Detection Model Using Machine Learning

A complete, beginner-friendly machine learning and cybersecurity project built with **Python**, **Scikit-learn**, and **Streamlit**. This system accurately classifies emails as **PHISHING** or **SAFE** based on email text semantics and embedded URL heuristic signals.

---

## 📌 Table of Contents
1. [Project Overview](#-project-overview)
2. [Problem Statement](#-problem-statement)
3. [Key Objectives](#-key-objectives)
4. [Technologies & Libraries Used](#-technologies--libraries-used)
5. [System Architecture & Workflow](#-system-architecture--workflow)
6. [Machine Learning Algorithms Compared](#-machine-learning-algorithms-compared)
7. [Project Directory Structure](#-project-directory-structure)
8. [Dataset Description](#-dataset-description)
9. [Step-by-Step Installation Guide](#-step-by-step-installation-guide)
10. [How to Run the Project](#-how-to-run-the-project)
11. [Model Evaluation & Benchmark Results](#-model-evaluation--benchmark-results)
12. [Features of the Streamlit Web Application](#-features-of-the-streamlit-web-application)
13. [Limitations & Ethical Considerations](#-limitations--ethical-considerations)
14. [Future Enhancements](#-future-enhancements)
15. [License](#-license)

---

## 📖 Project Overview

Email phishing remains the primary attack vector for cybercriminals to compromise credentials, deploy ransomware, and execute financial fraud. Traditional rule-based filters (like static keyword blocklists) often fail when attackers slightly alter wording or register novel spoofed domains.

This project implements an end-to-end, reproducible Machine Learning pipeline that:
1. Cleans and normalizes email content.
2. Extracts high-dimensional textual features using **TF-IDF (Term Frequency-Inverse Document Frequency)**.
3. Conducts safe, static inspection of embedded URLs (detecting IP hostnames, abusive TLDs, credential theft characters, and length anomalies) without visiting the links.
4. Trains and benchmarks three distinct supervised classification algorithms: **Logistic Regression**, **Multinomial Naive Bayes**, and **Random Forest**.
5. Deploys the best-performing model into an interactive, modern **Streamlit web application** with a clean White & Blue dashboard aesthetic.

---

## 🎯 Problem Statement

Every day, billions of phishing emails are sent targeting organizations and consumers. Common attack vectors include:
- **Banking Suspensions & Fraud Alerts:** Panicking users into inputting credentials.
- **Corporate SSO & Office 365 Expiration:** Stealing workplace logins.
- **CEO Fraud & Business Email Compromise (BEC):** Trick employees into urgent wire transfers or gift card purchases.
- **Fake Delivery & Customs Notices:** Enticing package tracking clicks.

The challenge is to distinguish these deceptive messages from legitimate workplace memos, engineering pull requests, e-commerce receipts, and routine customer service tickets.

---

## 🚀 Key Objectives

- Build a modular, well-documented machine learning codebase from scratch.
- Preprocess unstructured email text and extract structured URL indicators.
- Train and compare three classification models using rigorous train/test splits that prevent data leakage.
- Generate high-resolution evaluation charts (Confusion Matrix heatmap and Algorithm Comparison bar chart).
- Deliver a responsive Streamlit dashboard with real-time prediction, sample test buttons, and educational cyber hygiene tips.
- Ensure the project is 100% beginner-friendly and GitHub-ready.

---

## 🛠️ Technologies & Libraries Used

| Technology | Purpose |
| :--- | :--- |
| **Python 3.10 - 3.12** | Core programming language |
| **Scikit-learn** | ML model training, TF-IDF vectorization, train-test split, and evaluation metrics |
| **Pandas** | Tabular dataset manipulation, cleaning, and preprocessing |
| **NumPy** | High-performance matrix operations and mathematical calculations |
| **Matplotlib** | Static plot generation and benchmark visualizations |
| **Seaborn** | High-resolution confusion matrix heatmaps |
| **Streamlit** | Modern, interactive web user interface |
| **Joblib** | Model and vectorizer serialization/deserialization |
| **Regex (`re`)** | Pattern-based URL extraction and text normalization |

---

## 🔄 System Architecture & Workflow

```text
  Raw Email Text (Subject & Body)
               │
               ▼
  ┌─────────────────────────┐
  │   Data Preprocessing    │ ───► Lowercasing, HTML stripping, punctuation removal
  └─────────────────────────┘
         │             │
         ▼             ▼
  ┌──────────────┐  ┌────────────────────────┐
  │    TF-IDF    │  │ Static URL Inspection  │
  │ Vectorization│  │ (Safe heuristic scan   │
  │ (1-2 ngrams) │  │  without visiting link)│
  └──────────────┘  └────────────────────────┘
         │                     │
         ▼                     ▼
  ┌──────────────────────────────────────────┐
  │ Machine Learning Classification Engine   │
  │   - Multinomial Naive Bayes (Best Model) │
  │   - Logistic Regression                  │
  │   - Random Forest Classifier             │
  └──────────────────────────────────────────┘
               │
               ▼
  ┌──────────────────────────────────────────┐
  │ Streamlit Web Dashboard Output           │
  │  - Prediction (PHISHING vs SAFE)         │
  │  - Probabilistic Confidence Score        │
  │  - Detailed URL Security Breakdown       │
  │  - Urgency & Financial Keyword Signals   │
  └──────────────────────────────────────────┘
```

---

## 🧠 Machine Learning Algorithms Compared

To find the optimal classifier for text-based phishing detection, we train and evaluate three diverse algorithms:

### 1. Multinomial Naive Bayes (Selected Best Model)
- **Concept:** A probabilistic classifier based on Bayes' Theorem with strong independence assumptions between features.
- **Why it fits:** Naive Bayes is renowned for high efficiency and superior performance in text classification and spam detection, particularly when paired with TF-IDF unigram and bigram counts.
- **Benchmark F1-Score:** **96.97%** | **Accuracy:** **96.43%**

### 2. Logistic Regression
- **Concept:** A linear model that estimates the probability of binary outcomes using a sigmoid link function.
- **Why it fits:** Provides highly interpretable feature coefficients and fast training convergence.
- **Benchmark F1-Score:** **93.75%** | **Accuracy:** **92.86%**

### 3. Random Forest Classifier
- **Concept:** An ensemble meta-estimator that fits numerous decision tree classifiers on various sub-samples of the dataset and uses averaging to control over-fitting.
- **Why it fits:** Captures non-linear interactions between combinations of words and phrases.
- **Benchmark F1-Score:** **83.33%** | **Accuracy:** **78.57%**

---

## 📂 Project Directory Structure

```text
phishing_email_detection/
│
├── app.py                     # Streamlit interactive web dashboard
├── train_model.py             # Model training, evaluation & chart generation
├── preprocessing.py           # Text cleaning, normalization & vectorizer setup
├── feature_extraction.py      # Safe static URL analysis & linguistic heuristic scanner
├── predict.py                 # CLI & Python module inference utility
├── create_dataset.py          # Benchmark dataset generator
├── phishing_email_dataset.csv # Primary balanced dataset (Phishing & Safe emails)
├── dataset.csv                # Compatibility copy of dataset
├── requirements.txt           # Python dependency specifications
├── run_app.bat                # Windows double-click launcher
├── .gitignore                 # Standard Python/ML git ignore rules
├── README.md                  # Comprehensive project documentation
│
├── models/
│   ├── phishing_model.pkl     # Primary model bundle (estimator, vectorizer, metrics)
│   └── model_metrics.json     # Performance scores for all 3 algorithms
│
└── images/
    ├── confusion_matrix.png   # Seaborn heatmap of test evaluation
    └── model_comparison.png   # Grouped bar chart comparing all 3 models
```

---

## 📊 Dataset Description

The dataset `phishing_email_dataset.csv` contains realistic emails curated across multiple distinct threat and legitimate categories:

### Phishing Categories (Label: 1)
- **Banking Scam:** Unauthorized access alerts (Chase, Wells Fargo, Bank of America, Citibank).
- **Payment Scam:** Account limitations and pending holds (PayPal, Venmo, Stripe).
- **Credential Harvesting:** Expiring Microsoft 365 passwords, Google Workspace revoke alerts, Zoom recording links, Dropbox file shares.
- **Invoice & Antivirus Scam:** Overdue invoices, Geek Squad, Norton LifeLock auto-renewals.
- **CEO Fraud / BEC:** Urgent confidential wire transfer requests, CEO gift card queries.
- **Delivery Scams:** Package holds and redelivery fees (USPS, FedEx, DHL, UPS).
- **Government Impersonation:** IRS refunds, SSA suspension threats, DMV toll notices.

### Legitimate / Safe Categories (Label: 0)
- **Workplace Memos:** Agile sprint planning, retrospective notes, Q3 business reviews, slide reviews.
- **Developer & CI Tools:** GitHub PR approvals, GitLab CI pipeline passes, Docker image builds.
- **E-Commerce & Orders:** Amazon delivery updates, Apple Store receipts, Uber/DoorDash ride summaries.
- **Service & Travel:** Delta flight receipts, Marriott booking confirmations, library hold notices.
- **Security & System Notices:** User-requested password resets, AWS budget notifications.

---

## 💻 Step-by-Step Installation Guide

### Step 1: Clone or Download the Repository
Open your terminal (Command Prompt, PowerShell, or macOS/Linux Terminal) and navigate to the project directory:
```bash
cd "path/to/phishing_email_detection"
```

### Step 2: (Recommended) Create a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 🏃 How to Run the Project

### 1. Train the Machine Learning Models
To train all three models, evaluate them on the test set, generate evaluation plots, and save the model artifacts:
```bash
python train_model.py
```
*Expected Output:*
- Prints training progression, dataset split counts, and test set metrics.
- Automatically selects the best model.
- Saves `models/phishing_model.pkl`.
- Generates `images/confusion_matrix.png` and `images/model_comparison.png`.

---

### 2. Launch the Streamlit Web Application
To start the browser dashboard:
```bash
streamlit run app.py
```
*(On Windows, you can also simply double-click `run_app.bat`!)*

Once started, Streamlit will open your default browser at:
`http://localhost:8501`

---

### 3. Run Command-Line Predictions (CLI Tool)
You can test any email string directly from your terminal:
```bash
# Test a phishing email
python predict.py "URGENT: Your PayPal account has been suspended! Click http://paypal-verify-alert.xyz to restore access."

# Test a legitimate email
python predict.py "Hi Team, our sprint planning meeting is at 10 AM on Monday in Room B. Please review the agenda."

# Interactive mode
python predict.py
```

---

## 📈 Model Evaluation & Benchmark Results

On an 80/20 stratified train/test split:

| Algorithm | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes ⭐** | **96.43%** | **94.12%** | **100.00%** | **96.97%** |
| **Logistic Regression** | **92.86%** | **93.75%** | **93.75%** | **93.75%** |
| **Random Forest** | **78.57%** | **75.00%** | **93.75%** | **83.33%** |

### Evaluation Graphs
The training pipeline automatically outputs high-resolution charts saved in the `images/` directory:
- `images/confusion_matrix.png`: Shows True Positives, True Negatives, False Positives, and False Negatives.
- `images/model_comparison.png`: Bar chart benchmarking Accuracy, Precision, Recall, and F1 across all models.

---

## 🌟 Features of the Streamlit Web Application

1. **Email & URL Scanner:**
   - Large text area to paste email headers, subject, and body.
   - Quick-sample buttons to test realistic scenarios in 1 click.
   - Clear visual verdict card (Red for Phishing, Green for Safe).
   - Probabilistic confidence score with an educational disclaimer.
2. **Safe Static URL Inspection:**
   - Extracts all links using regular expressions.
   - Inspects URLs without visiting or querying remote servers.
   - Flags raw numeric IPs, suspicious TLDs (`.xyz`, `.top`, `.cc`), obfuscation characters (`@`), and excessive subdomains.
3. **Linguistic Heuristic Breakdown:**
   - Measures urgency triggers (*'immediately'*, *'suspended'*, *'within 24 hours'*).
   - Flags financial indicators (*'wire'*, *'bitcoin'*, *'invoice'*).
   - Computes uppercase ratio to flag deceptive high-pressure shouting.
4. **Interactive Model Performance Dashboard:**
   - Displays real-time Accuracy, Precision, Recall, and F1 cards.
   - Embedded Confusion Matrix and Algorithm Comparison charts.
   - Filterable view of the full classification report.
5. **Dataset Explorer:**
   - Browse the training dataset with label filters.
   - View category and threat distribution graphs.
6. **Cybersecurity Awareness Guide:**
   - Educational resource covering common social engineering tactics and defense best practices.

---

## ⚠️ Limitations & Ethical Considerations

- **Statistical Estimate, Not an Absolute Guarantee:** Machine learning models predict based on patterns learned from training data. Attackers constantly invent novel evasion strategies.
- **False Positives:** Legitimate marketing newsletters or urgent system alerts (e.g. *'Database Server Out of Memory - Immediate Action Required'*) may occasionally be flagged as suspicious due to urgency keywords.
- **False Negatives:** Highly customized spear phishing emails lacking overt keywords or external links may appear benign to pure text models.
- **Defensive Use Only:** This model and inspection toolkit are designed strictly for educational, defensive, and email screening purposes.

---

## 🔮 Future Enhancements

- [ ] Integrate deep learning transformers (e.g., DistilBERT or RoBERTa) for contextual embeddings.
- [ ] Connect with live domain reputation APIs (Google Safe Browsing, VirusTotal) for real-time URL reputation checks.
- [ ] Add header analysis (SPF, DKIM, DMARC verification).
- [ ] Support automated `.eml` and `.msg` file uploads with attachment virus scanning.
- [ ] Implement a Chrome or Outlook browser extension plugin.

---

## 📜 License
This project is open-source and available under the **MIT License**. Free for educational, personal, and research use.
