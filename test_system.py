"""
test_system.py
--------------
Automated test suite verifying all modules, artifacts, and edge cases.
"""

import os
import joblib
import pandas as pd
from preprocessing import clean_text, clean_dataset, tokenize_text
from feature_extraction import extract_urls, analyze_url, extract_email_url_features, extract_linguistic_features
from predict import predict_email, load_model_bundle

def run_tests():
    print("=" * 65)
    print("      RUNNING PHISHING DETECTION AUTOMATED TEST SUITE")
    print("=" * 65)
    
    # Test 1: Check required files and directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    required_files = [
        "app.py",
        "train_model.py",
        "preprocessing.py",
        "feature_extraction.py",
        "predict.py",
        "phishing_email_dataset.csv",
        "requirements.txt",
        "README.md",
        "run_app.bat",
        ".gitignore",
        os.path.join("models", "phishing_model.pkl"),
        os.path.join("models", "model_metrics.json"),
        os.path.join("images", "confusion_matrix.png"),
        os.path.join("images", "model_comparison.png")
    ]
    
    print("\n[Test 1] Verifying File Structure...")
    for f in required_files:
        full_path = os.path.join(base_dir, f)
        assert os.path.exists(full_path), f"Missing required file: {f}"
        print(f"  [PASS] Found: {f}")

    # Test 2: Dataset integrity
    print("\n[Test 2] Verifying Dataset...")
    df = pd.read_csv(os.path.join(base_dir, "phishing_email_dataset.csv"))
    assert "email_text" in df.columns, "Missing 'email_text' column"
    assert "label" in df.columns, "Missing 'label' column"
    assert len(df) >= 100, f"Dataset too small ({len(df)} records)"
    assert set(df["label"].unique()) == {0, 1}, "Labels must be binary (0 and 1)"
    print(f"  [PASS] Dataset valid with {len(df)} records. Phishing: {(df['label']==1).sum()}, Safe: {(df['label']==0).sum()}")

    # Test 3: Preprocessing
    print("\n[Test 3] Testing Text Preprocessing...")
    raw_sample = "<b>URGENT:</b> Please <a href='http://bad.xyz'>CLICK HERE</a> to verify!"
    cleaned = clean_text(raw_sample)
    assert "<b>" not in cleaned and "</b>" not in cleaned, "HTML tags not stripped"
    assert "http" not in cleaned, "Raw URLs not removed from text tokens"
    tokens = tokenize_text("Important Notice from CEO")
    assert isinstance(tokens, list) and len(tokens) > 0, "Tokenization failed"
    print(f"  [PASS] Text cleaning and tokenization functional. Cleaned: '{cleaned}'")

    # Test 4: Feature Extraction (URLs & Linguistics)
    print("\n[Test 4] Testing Safe URL Inspection & Linguistics...")
    test_url_text = "Check out http://192.168.1.50/login and also http://fake-bank-update.xyz/auth?user=victim@test.com"
    extracted_urls = extract_urls(test_url_text)
    assert len(extracted_urls) == 2, f"Expected 2 URLs, got {len(extracted_urls)}"
    
    ip_analysis = analyze_url("http://192.168.1.50/login")
    assert ip_analysis["has_ip"] is True, "Failed to detect raw IP in URL"
    assert ip_analysis["risk_level"] == "High Risk", "Raw IP should be High Risk"

    tld_analysis = analyze_url("http://fake-bank-update.xyz/auth")
    assert tld_analysis["suspicious_tld"] == ".xyz", "Failed to detect .xyz TLD"
    assert len(tld_analysis["keywords_found"]) > 0, "Failed to detect 'auth' or 'bank' keyword"

    ling = extract_linguistic_features("URGENT: TRANSFER $5,000 IMMEDIATELY!")
    assert ling["urgency_count"] >= 2, "Urgency words not counted properly"
    assert ling["capital_ratio"] > 50, "Capital ratio calculation incorrect"
    print("  [PASS] URL extraction, static heuristic inspection, and linguistic detectors verified.")

    # Test 5: Model Inference on diverse test inputs
    print("\n[Test 5] Testing Model Predictions & Edge Cases...")
    model, vectorizer, model_name = load_model_bundle()
    assert model is not None and vectorizer is not None, "Model or vectorizer failed to load"
    print(f"  Loaded active model: {model_name}")

    # Case A: Real phishing sample
    res_phish = predict_email(
        "URGENT: Your Chase Bank account has been temporarily suspended due to unauthorized access attempts. Click here: http://chase-security-verify-login.xyz/auth",
        model, vectorizer
    )
    assert res_phish["prediction"] == "Phishing", f"Expected Phishing, got {res_phish['prediction']}"
    assert res_phish["label"] == 1
    assert res_phish["confidence"] > 50.0
    print(f"  [PASS] Phishing Email detected accurately: {res_phish['prediction']} ({res_phish['confidence']:.1f}%)")

    # Case B: Real safe sample
    res_safe = predict_email(
        "Hi Team, please find attached the agenda for our sprint retrospective meeting tomorrow at 10:00 AM.",
        model, vectorizer
    )
    assert res_safe["prediction"] == "Safe", f"Expected Safe, got {res_safe['prediction']}"
    assert res_safe["label"] == 0
    assert res_safe["confidence"] > 50.0
    print(f"  [PASS] Safe Email detected accurately: {res_safe['prediction']} ({res_safe['confidence']:.1f}%)")

    # Case C: Empty string
    res_empty = predict_email("", model, vectorizer)
    assert res_empty["prediction"] == "Invalid Input", "Empty string should return Invalid Input"
    print("  [PASS] Empty input properly handled.")

    # Case D: Whitespace only
    res_ws = predict_email("      \t\n   ", model, vectorizer)
    assert res_ws["prediction"] == "Invalid Input", "Whitespace input should return Invalid Input"
    print("  [PASS] Whitespace-only input properly handled.")

    # Case E: Punctuation & symbols only
    res_sym = predict_email("$$$ @@@ !!! %%%", model, vectorizer)
    assert res_sym["prediction"] == "Invalid Input", "Symbol-only input should return Invalid Input"
    print("  [PASS] Non-lexical symbol input properly handled.")

    print("\n" + "=" * 65)
    print("         ALL AUTOMATED SYSTEM TESTS PASSED PERFECTLY!")
    print("=" * 65)

if __name__ == "__main__":
    run_tests()
