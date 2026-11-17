"""
feature_extraction.py
---------------------
Module for extracting URL-related features and heuristic indicators from email text.
Performs safe, static inspection of URLs WITHOUT opening, fetching, or visiting them.

Key Functions:
1. extract_urls: Extracts all URLs from email text using regular expressions.
2. analyze_url: Performs static inspection of an individual URL for phishing markers.
3. extract_email_url_features: Computes email-level URL metrics.
4. extract_linguistic_features: Detects urgency triggers, financial keywords, and styling clues.
"""

import re
from urllib.parse import urlparse

# Suspicious Top-Level Domains (TLDs) frequently abused in phishing campaigns
SUSPICIOUS_TLDS = {
    ".xyz", ".top", ".cc", ".buzz", ".info", ".online", ".club",
    ".tk", ".ml", ".ga", ".cf", ".gq", ".work", ".fit", ".vip",
    ".icu", ".live", ".biz", ".site", ".website"
}

# Suspicious keywords often present in phishing URLs
URL_SUSPICIOUS_KEYWORDS = [
    "login", "verify", "account", "banking", "secure", "update",
    "signin", "confirm", "password", "support", "auth", "billing",
    "wallet", "claim", "refund", "invoice", "recover", "security",
    "free", "gift", "bonus", "alert", "portal", "resolution"
]

# Urgency keywords indicating psychological pressure
URGENCY_KEYWORDS = [
    "urgent", "immediately", "suspended", "unauthorized", "24 hours",
    "48 hours", "locked", "action required", "final notice", "threat",
    "warning", "limited", "expire", "expires", "deactivated", "wiped"
]

# Financial keywords commonly seen in scams
FINANCIAL_KEYWORDS = [
    "bank", "wire", "transfer", "invoice", "payment", "refund",
    "gift card", "bitcoin", "crypto", "ssn", "social security",
    "routing number", "credit card", "checking account", "overdue", "payout"
]

# Regex pattern for extracting URLs safely
URL_PATTERN = re.compile(
    r'(?:https?://|www\.)[^\s<>"\'\]\[\)\(]+',
    re.IGNORECASE
)

# Regex pattern for detecting IPv4 address in domain
IPV4_PATTERN = re.compile(
    r'^(?:https?://)?(?:www\.)?(\d{1,3}\.){3}\d{1,3}(?::\d+)?(?:/.*)?$',
    re.IGNORECASE
)


def extract_urls(text: str) -> list:
    """
    Extracts all URLs from the provided text using regex.

    Parameters:
        text (str): Email text.

    Returns:
        list: List of extracted URL strings.
    """
    if not isinstance(text, str):
        return []
    urls = URL_PATTERN.findall(text)
    # Strip trailing punctuation like '.' or ',' often caught at sentence ends
    clean_urls = [u.rstrip(".,;:!?") for u in urls]
    return clean_urls


def analyze_url(url: str) -> dict:
    """
    Performs safe, static heuristic analysis of a single URL.
    Does NOT connect to or visit the target URL.

    Parameters:
        url (str): The URL string to inspect.

    Returns:
        dict: Inspection report containing metrics, red flags, and risk rating.
    """
    url_lower = url.lower()
    flags = []
    risk_score = 0

    # 1. URL Length Analysis
    url_length = len(url)
    if url_length > 75:
        flags.append(f"Abnormally long URL ({url_length} characters, often masks redirects)")
        risk_score += 25
    elif url_length > 50:
        risk_score += 10

    # 2. IP Address in Hostname
    has_ip = bool(IPV4_PATTERN.match(url_lower))
    if has_ip:
        flags.append("Uses raw numeric IP address instead of legitimate domain name")
        risk_score += 40

    # 3. Obfuscation character '@' in URL
    has_at_symbol = "@" in url
    if has_at_symbol:
        flags.append("Contains '@' symbol (often used for credential theft and browser URL spoofing)")
        risk_score += 35

    # 4. HTTPS vs Insecure HTTP
    is_https = url_lower.startswith("https://")
    if not is_https and url_lower.startswith("http://"):
        flags.append("Unencrypted connection (uses HTTP instead of secure HTTPS)")
        risk_score += 15

    # 5. Extract domain / hostname for structural checks
    try:
        # Prepend http:// for urlparse if starting with www.
        parsed = urlparse(url if url_lower.startswith(("http://", "https://")) else f"http://{url}")
        hostname = parsed.hostname or ""
    except Exception:
        hostname = url_lower

    # 6. Check for Suspicious Top-Level Domains (TLDs)
    tld_found = None
    for tld in SUSPICIOUS_TLDS:
        if hostname.endswith(tld):
            tld_found = tld
            flags.append(f"Uses high-risk TLD associated with cheap phishing domains: '{tld}'")
            risk_score += 30
            break

    # 7. Check Subdomain Stacking (Excessive dots)
    num_dots = hostname.count(".")
    if num_dots >= 3:
        flags.append(f"Excessive subdomains ({num_dots} dots, possible brand impersonation)")
        risk_score += 20

    # 8. Check for Suspicious Phishing Keywords
    matched_keywords = [kw for kw in URL_SUSPICIOUS_KEYWORDS if kw in url_lower]
    if matched_keywords:
        flags.append(f"Contains security/sensitive keywords in URL: {', '.join(matched_keywords[:4])}")
        risk_score += min(len(matched_keywords) * 15, 35)

    # 9. Determine overall risk level
    if risk_score >= 50 or has_ip:
        risk_level = "High Risk"
    elif risk_score >= 25 or len(matched_keywords) > 0 or tld_found:
        risk_level = "Medium Risk"
    else:
        risk_level = "Low Risk"

    return {
        "url": url,
        "length": url_length,
        "is_https": is_https,
        "has_ip": has_ip,
        "has_at_symbol": has_at_symbol,
        "num_dots": num_dots,
        "suspicious_tld": tld_found,
        "keywords_found": matched_keywords,
        "flags": flags,
        "risk_score": min(risk_score, 100),
        "risk_level": risk_level
    }


def extract_email_url_features(text: str) -> dict:
    """
    Extracts summary URL features for an entire email.

    Parameters:
        text (str): Full email text.

    Returns:
        dict: Aggregated URL features and list of analyzed URLs.
    """
    urls = extract_urls(text)
    url_count = len(urls)
    has_url = 1 if url_count > 0 else 0

    analyzed_urls = [analyze_url(u) for u in urls]
    high_risk_urls = [u for u in analyzed_urls if u["risk_level"] == "High Risk"]
    medium_risk_urls = [u for u in analyzed_urls if u["risk_level"] == "Medium Risk"]

    max_url_length = max([u["length"] for u in analyzed_urls]) if analyzed_urls else 0
    has_ip = any([u["has_ip"] for u in analyzed_urls]) if analyzed_urls else False

    return {
        "url_count": url_count,
        "has_url": has_url,
        "max_url_length": max_url_length,
        "has_ip_in_any_url": has_ip,
        "high_risk_url_count": len(high_risk_urls),
        "medium_risk_url_count": len(medium_risk_urls),
        "analyzed_urls": analyzed_urls
    }


def extract_linguistic_features(text: str) -> dict:
    """
    Extracts linguistic clues such as urgency keywords, financial terms,
    and capitalization intensity.

    Parameters:
        text (str): Email text.

    Returns:
        dict: Linguistic indicators report.
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    text_lower = text.lower()

    # Detect urgency words
    urgency_hits = [w for w in URGENCY_KEYWORDS if w in text_lower]

    # Detect financial terms
    financial_hits = [w for w in FINANCIAL_KEYWORDS if w in text_lower]

    # Capitalization ratio (ratio of uppercase letters among all alphabetic chars)
    alpha_chars = [c for c in text if c.isalpha()]
    upper_chars = [c for c in alpha_chars if c.isupper()]
    capital_ratio = (len(upper_chars) / len(alpha_chars) * 100) if alpha_chars else 0.0

    # Count exclamation points
    exclamation_count = text.count("!")

    return {
        "urgency_keywords": urgency_hits,
        "urgency_count": len(urgency_hits),
        "financial_keywords": financial_hits,
        "financial_count": len(financial_hits),
        "capital_ratio": round(capital_ratio, 1),
        "exclamation_count": exclamation_count
    }
