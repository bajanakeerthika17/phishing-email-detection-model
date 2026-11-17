"""
predict.py
----------
Inference and prediction module for Phishing Email Detection.
Accepts raw email text, applies preprocessing and TF-IDF transformation,
extracts URL features, and outputs classification results and confidence scores.

Usage:
  1. CLI command:
     python predict.py "URGENT: Click here to verify your PayPal account: http://paypal-fake.xyz"

  2. Interactive mode:
     python predict.py
"""

import os
import sys
import joblib

from preprocessing import clean_text
from feature_extraction import extract_email_url_features, extract_linguistic_features

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
PRIMARY_MODEL_PATH = os.path.join(MODELS_DIR, "phishing_model.pkl")
FALLBACK_MODEL_PATH = os.path.join(BASE_DIR, "phishing_model.joblib")
FALLBACK_VEC_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer.joblib")


def load_model_bundle():
    """
    Loads trained machine learning model, vectorizer, and metadata.
    """
    if os.path.exists(PRIMARY_MODEL_PATH):
        bundle = joblib.load(PRIMARY_MODEL_PATH)
        return bundle["best_model"], bundle["vectorizer"], bundle.get("best_model_name", "Best Model")
    elif os.path.exists(FALLBACK_MODEL_PATH) and os.path.exists(FALLBACK_VEC_PATH):
        model = joblib.load(FALLBACK_MODEL_PATH)
        vectorizer = joblib.load(FALLBACK_VEC_PATH)
        return model, vectorizer, "Trained Classifier"
    else:
        print("[!] No trained model found. Executing training pipeline...")
        from train_model import train_and_evaluate
        bundle = train_and_evaluate()
        return bundle["best_model"], bundle["vectorizer"], bundle["best_model_name"]


def predict_email(email_text: str, model=None, vectorizer=None):
    """
    Analyzes email text and outputs:
    - Prediction: "Phishing" or "Safe"
    - Confidence score (0-100%)
    - Probabilities for each class
    - URL analysis report
    - Linguistic indicators report

    Parameters:
        email_text (str): Raw email content.
        model: Optional pre-loaded classifier.
        vectorizer: Optional pre-loaded TF-IDF vectorizer.

    Returns:
        dict: Complete diagnostic analysis report.
    """
    if model is None or vectorizer is None:
        model, vectorizer, _ = load_model_bundle()

    # Handle empty or whitespace-only input
    if not email_text or not email_text.strip():
        return {
            "prediction": "Invalid Input",
            "label": -1,
            "confidence": 0.0,
            "phishing_prob": 0.0,
            "safe_prob": 0.0,
            "explanation": "No text provided. Please enter or paste email text to analyze.",
            "url_analysis": {"url_count": 0, "has_url": 0, "analyzed_urls": []},
            "linguistic_analysis": {"urgency_count": 0, "financial_count": 0, "capital_ratio": 0.0}
        }

    # 1. Preprocess text
    cleaned = clean_text(email_text, remove_stopwords=False)

    if not cleaned or len(cleaned.strip()) == 0:
        return {
            "prediction": "Invalid Input",
            "label": -1,
            "confidence": 0.0,
            "phishing_prob": 0.0,
            "safe_prob": 0.0,
            "explanation": "The text provided does not contain recognizable words. Please provide valid email text.",
            "url_analysis": extract_email_url_features(email_text),
            "linguistic_analysis": extract_linguistic_features(email_text)
        }

    # 2. Extract URL & Linguistic Features
    url_analysis = extract_email_url_features(email_text)
    linguistic_analysis = extract_linguistic_features(email_text)

    # 3. Vectorize text
    text_vector = vectorizer.transform([cleaned])

    # 4. Predict class and probabilities
    pred_label = int(model.predict(text_vector)[0])

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(text_vector)[0]
        safe_prob = round(float(probabilities[0]) * 100, 2)
        phishing_prob = round(float(probabilities[1]) * 100, 2)
    else:
        # Fallback if classifier doesn't implement predict_proba
        decision = model.decision_function(text_vector)[0]
        phishing_prob = 100.0 if decision > 0 else 0.0
        safe_prob = 100.0 - phishing_prob

    # 5. Heuristic URL Boost:
    # If the email contains a high-risk URL (e.g. IP address or phishing TLD) but ML model was borderline,
    # raise security caution flags
    if url_analysis["high_risk_url_count"] > 0 and pred_label == 0:
        warning_flag = "Caution: While the text wording scored as Safe, high-risk URLs were detected in the message!"
    else:
        warning_flag = None

    prediction_str = "Phishing" if pred_label == 1 else "Safe"
    confidence = phishing_prob if pred_label == 1 else safe_prob

    return {
        "prediction": prediction_str,
        "label": pred_label,
        "confidence": confidence,
        "phishing_prob": phishing_prob,
        "safe_prob": safe_prob,
        "warning_flag": warning_flag,
        "url_analysis": url_analysis,
        "linguistic_analysis": linguistic_analysis,
        "cleaned_text": cleaned
    }


def main():
    print("=" * 65)
    print("      PHISHING EMAIL DETECTION - PREDICTION TOOL")
    print("=" * 65)

    model, vectorizer, model_name = load_model_bundle()
    print(f"[+] Loaded Model: {model_name}")

    if len(sys.argv) > 1:
        # Email text passed via CLI argument
        email_input = " ".join(sys.argv[1:])
        result = predict_email(email_input, model, vectorizer)
        print("\n--- ANALYSIS RESULT ---")
        print(f"Prediction:       {result['prediction'].upper()}")
        print(f"Confidence:       {result['confidence']:.2f}%")
        print(f"Phishing Prob:    {result['phishing_prob']:.2f}%")
        print(f"Safe Prob:        {result['safe_prob']:.2f}%")
        print(f"URLs Found:       {result['url_analysis']['url_count']}")
        if result['warning_flag']:
            print(f"Security Alert:   {result['warning_flag']}")
    else:
        print("\n[Interactive Mode] Paste email text below and press Enter (or type 'exit' to quit):")
        while True:
            try:
                user_input = input("\nEnter Email Text > ").strip()
                if user_input.lower() in ("exit", "quit", "q"):
                    print("Exiting prediction tool.")
                    break
                if not user_input:
                    print("[!] Please enter some text.")
                    continue

                res = predict_email(user_input, model, vectorizer)
                tag = "[!] PHISHING" if res["label"] == 1 else "[+] SAFE"
                print(f"\n{tag} (Confidence: {res['confidence']:.2f}%)")
                print(f"Phishing Probability: {res['phishing_prob']:.2f}% | Safe Probability: {res['safe_prob']:.2f}%")
                if res["url_analysis"]["url_count"] > 0:
                    print(f"URLs Detected: {res['url_analysis']['url_count']}")
                    for u in res["url_analysis"]["analyzed_urls"]:
                        print(f"  * {u['url']} -> Risk: {u['risk_level']} (Score: {u['risk_score']}/100)")
                        for flag in u["flags"]:
                            print(f"      - {flag}")
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break


if __name__ == "__main__":
    main()
