"""
create_dataset.py
-----------------
Generates the comprehensive benchmark dataset for Phishing Email Detection:
'phishing_email_dataset.csv'

Features:
- Realistic phishing emails covering modern threat vectors:
  * Banking / Financial suspensions
  * PayPal & Payment alerts
  * Corporate SSO / Office 365 credential harvesting
  * CEO Fraud & Business Email Compromise (BEC)
  * Fake Invoices & Billing scams (Norton, Geek Squad, QuickBooks)
  * Package delivery notifications (FedEx, DHL, USPS, UPS)
  * Crypto giveaways & lottery scams
  * Government impersonation (IRS, SSA, DMV)
  * Social media security warnings (Meta, Instagram, LinkedIn)
  * HR & Payroll portal migrations
- Realistic legitimate (Safe) emails:
  * Internal project memos and sprint updates
  * Code review requests & GitHub / GitLab CI notifications
  * E-commerce receipts & shipping notices (Amazon, Apple)
  * University course notifications
  * Calendar invitations & meeting agendas
  * Customer support ticket responses
  * Industry newsletters & tech digests
  * Company HR all-hands & benefits notifications
  * Legitimate automated password resets & system alerts
"""

import os
import pandas as pd

def build_dataset():
    data = [
        # ==========================================
        # PHISHING EMAILS (Label: 1)
        # ==========================================
        # 1. Banking & Finance
        {
            "email_text": "URGENT: Your Chase Bank account has been temporarily suspended due to suspicious login attempts from IP 194.26.29.112. Click here immediately to verify your identity and restore access: http://chase-security-verify-login.xyz/auth",
            "label": 1,
            "category": "Banking Scam"
        },
        {
            "email_text": "Wells Fargo Alert: Suspicious transaction of $1,850.00 detected on your checking account. If you did not authorize this, cancel the transfer immediately at: http://wellsfargo-fraud-prevention-desk.online",
            "label": 1,
            "category": "Banking Scam"
        },
        {
            "email_text": "Bank of America Warning: Direct debit failure on your mortgage payment. Avoid penalty fees by confirming your routing and account number now: http://boa-portal-account-check.top/login",
            "label": 1,
            "category": "Banking Scam"
        },
        {
            "email_text": "SECURITY NOTIFICATION: Unusual withdrawal of $3,200.00 from your Citibank savings account. Verify your identity or report fraud here: http://citibank-alert-verification.com",
            "label": 1,
            "category": "Banking Scam"
        },
        {
            "email_text": "Capital One Fraud Protection: We locked your credit card due to unauthorized online activity. Unlock your card now by validating your Social Security Number at: http://capitalone-card-unlock.info",
            "label": 1,
            "category": "Banking Scam"
        },

        # 2. PayPal & Online Payment
        {
            "email_text": "Your PayPal account has been limited! We detected unusual login attempts from an unknown device in Moscow. Confirm your credentials within 24 hours to avoid permanent closure: http://paypal-resolution-center-update.cc",
            "label": 1,
            "category": "Payment Scam"
        },
        {
            "email_text": "PayPal Billing Issue: Your payment of $499.00 to CryptoTraders Ltd could not be processed. Update your billing address and credit card details here: http://paypal-billing-fix.top/update",
            "label": 1,
            "category": "Payment Scam"
        },
        {
            "email_text": "Venmo Security Notice: Someone attempted to withdraw $750 to a new debit card. If this was not authorized by you, reverse the transfer immediately: http://venmo-cancel-unauthorized-trans.xyz",
            "label": 1,
            "category": "Payment Scam"
        },
        {
            "email_text": "Stripe Merchant Alert: Your payout of $4,850.00 has been held due to missing tax documents. Verify your business registration and bank account at: http://stripe-merchant-verification.net",
            "label": 1,
            "category": "Payment Scam"
        },

        # 3. Corporate Credential Harvesting
        {
            "email_text": "Final Notice: Your Microsoft 365 corporate password expires in 2 hours! To keep your current password, validate your account information at: http://login-microsoft365-verify.net/auth",
            "label": 1,
            "category": "Credential Harvesting"
        },
        {
            "email_text": "IT Helpdesk Alert: System server migration scheduled tonight. All unconfirmed email accounts will be permanently deactivated. Keep your mailbox active by logging in here: http://corporate-it-webmail-sync.biz",
            "label": 1,
            "category": "Credential Harvesting"
        },
        {
            "email_text": "Google Workspace Security: A third-party malicious application was granted full access to your Gmail account. Revoke access immediately: http://google-security-auth-check.xyz/manage",
            "label": 1,
            "category": "Credential Harvesting"
        },
        {
            "email_text": "Your email storage quota has exceeded 99.8% capacity. You will stop receiving incoming emails in 60 minutes unless you increase your free quota at: http://webmail-storage-upgrade.biz",
            "label": 1,
            "category": "Credential Harvesting"
        },
        {
            "email_text": "Zoom Video Communications: You missed an urgent recorded video conference call from Senior HR. Log in with your corporate email password to listen to the recording: http://zoom-meeting-recordings-archive.biz",
            "label": 1,
            "category": "Credential Harvesting"
        },
        {
            "email_text": "Slack Notification: You were removed from #executives workspace. To appeal this administrative change, re-authenticate your workplace profile at: http://slack-workplace-reauth.info",
            "label": 1,
            "category": "Credential Harvesting"
        },

        # 4. Invoice & Billing Scams
        {
            "email_text": "Dear Customer, please find attached overdue invoice #INV-94021 for $2,450.00. Failure to remit payment within 24 hours will result in immediate legal collection action. Click here to remit payment: http://pay-invoice-secure.xyz",
            "label": 1,
            "category": "Invoice Scam"
        },
        {
            "email_text": "Geek Squad Best Buy: Your annual auto-renewal subscription of $499.99 has been debited. If you did not authorize this charge, call our cancellation desk immediately at +1-800-FAKE-NUM or click link to claim full refund: http://geeksquad-refund-portal.net",
            "label": 1,
            "category": "Invoice Scam"
        },
        {
            "email_text": "Norton LifeLock: Receipt for your order #NL-849201. We have charged $379.00 from your checking account. To dispute this automatic payment, click here within 48 hours: http://norton-refund-desk.xyz",
            "label": 1,
            "category": "Invoice Scam"
        },
        {
            "email_text": "McAfee Antivirus Protection: Your annual protection expired today! Your PC is now vulnerable to 5 trojan viruses. Renew your license now with a 70% discount: http://mcafee-protect-renew.top",
            "label": 1,
            "category": "Invoice Scam"
        },
        {
            "email_text": "QuickBooks Online: You received a new bill from Industrial Supplies Inc for $4,120.00. Review the document and approve payment via this link: http://quickbooks-invoice-view.info/download",
            "label": 1,
            "category": "Invoice Scam"
        },

        # 5. CEO Fraud & BEC
        {
            "email_text": "Are you at your desk right now? I am currently in a confidential executive board meeting and cannot take calls. I need you to process an urgent wire transfer for an acquisition vendor right away. Send me your cell number. Sent from iPhone.",
            "label": 1,
            "category": "CEO Fraud / BEC"
        },
        {
            "email_text": "Quick request from the CEO: Can you purchase 5 Apple gift cards ($100 each) for employee appreciation awards today? Email me the voucher codes immediately as I need them before 3 PM for our client.",
            "label": 1,
            "category": "CEO Fraud / BEC"
        },
        {
            "email_text": "CONFIDENTIAL REQUEST: We are closing a discreet corporate takeover today. Please prepare a wire of $45,000 to the attached escrow routing account. Keep this strictly off public Slack channels.",
            "label": 1,
            "category": "CEO Fraud / BEC"
        },
        {
            "email_text": "From Management: Please update our vendor bank wire instructions for Horizon Logistics. They changed banks; use the attached routing form immediately for all pending disbursements.",
            "label": 1,
            "category": "CEO Fraud / BEC"
        },

        # 6. HR & Payroll Scams
        {
            "email_text": "Confidential Payroll Update from HR: We are migrating to a new direct deposit payroll system. Please click the link below to input your bank account number and routing number before Friday: http://company-payroll-portal.cc",
            "label": 1,
            "category": "HR Scam"
        },
        {
            "email_text": "HR Department: Mandatory 2024 Employee Benefits and Bonus Enrollment is closing at 5 PM today. Confirm your personal details and tax withholding exemptions here: http://hr-benefits-portal-login.online",
            "label": 1,
            "category": "HR Scam"
        },
        {
            "email_text": "URGENT HR MEMO: Please review the updated employee code of conduct and sign your termination acknowledgment: http://docusign-hr-termination-policy.net",
            "label": 1,
            "category": "HR Scam"
        },

        # 7. Package Delivery & Logistics
        {
            "email_text": "USPS Notification: We were unable to deliver your package #US-849103 due to an incomplete home address. Please pay the $1.95 redelivery fee and update your address here: http://usps-track-package-resolve.com",
            "label": 1,
            "category": "Delivery Scam"
        },
        {
            "email_text": "FedEx Express: Your parcel is on hold at our distribution facility awaiting customs clearance. Confirm your identity and pay customs duty of $3.50: http://fedex-parcel-clearance.net",
            "label": 1,
            "category": "Delivery Scam"
        },
        {
            "email_text": "DHL Express: Delivery attempt failed for shipment tracking code #DHL-938201. Reschedule your parcel delivery and download the shipping label here: http://dhl-express-track-update.top",
            "label": 1,
            "category": "Delivery Scam"
        },
        {
            "email_text": "UPS Tracking Alert: Important document package returned to depot. Confirm your recipient address within 24 hours: http://ups-package-confirm-address.biz",
            "label": 1,
            "category": "Delivery Scam"
        },

        # 8. Crypto & Lottery Scams
        {
            "email_text": "CONGRATULATIONS! You have been selected as the lucky winner of $1,500,000 in the International Coca-Cola Mega Sweepstakes! Send your full name, home address, and bank account number to claim your prize.",
            "label": 1,
            "category": "Lottery Scam"
        },
        {
            "email_text": "Binance Security Alert: Your crypto wallet has received 1.25 BTC ($85,000.00). To release the escrow funds to your personal wallet, confirm your private key here: http://binance-escrow-claim.net",
            "label": 1,
            "category": "Crypto Scam"
        },
        {
            "email_text": "Exclusive Crypto Opportunity: Earn guaranteed 500% daily returns with automated Bitcoin trading AI. Deposit $250 today to start trading immediately: http://crypto-ai-multiplier-yield.org",
            "label": 1,
            "category": "Crypto Scam"
        },
        {
            "email_text": "Coinbase Alert: Unusual withdrawal request of 3.4 ETH detected. If you did not initiate this transaction, cancel it immediately by entering your seed phrase: http://coinbase-auth-cancel.xyz",
            "label": 1,
            "category": "Crypto Scam"
        },

        # 9. Government & Legal Threats
        {
            "email_text": "Internal Revenue Service (IRS): You have an unclaimed tax refund of $1,840.50 pending for the tax year. Submit your direct deposit details here: http://irs-tax-refund-portal.us.com",
            "label": 1,
            "category": "Government Impersonation"
        },
        {
            "email_text": "Social Security Administration: Your Social Security Number (SSN) has been suspended due to suspicious criminal activities. Contact our federal agent or click here to prevent asset freezing: http://ssa-gov-verification-agent.net",
            "label": 1,
            "category": "Government Impersonation"
        },
        {
            "email_text": "Department of Motor Vehicles: Final notice regarding unpaid speeding toll violations. Pay your $45.00 balance today to avoid suspension of your driver's license: http://dmv-toll-settlement.online",
            "label": 1,
            "category": "Government Impersonation"
        },
        {
            "email_text": "Legal Notice: A formal lawsuit has been filed against you in Civil Court. Download your subpoena notice and court summons by logging in here: http://court-summons-document.top",
            "label": 1,
            "category": "Government Impersonation"
        },

        # 10. Social Media & Tech Platforms
        {
            "email_text": "Meta Security Team: Your Facebook Business Page has been reported for trademark violations. Submit your verification documents here within 24 hours to prevent page deletion: http://meta-business-appeal-desk.com",
            "label": 1,
            "category": "Social Media Scam"
        },
        {
            "email_text": "Action Required: Your Apple ID is locked for security reasons. You must unlock your account within 12 hours or your iCloud data will be permanently wiped: http://apple-id-recovery.info",
            "label": 1,
            "category": "Tech Scam"
        },
        {
            "email_text": "Urgent: Unauthorized login detected on your Netflix account. Your subscription has been suspended due to billing error. Update your credit card details immediately: http://netflix-billing-resolve.com",
            "label": 1,
            "category": "Tech Scam"
        },
        {
            "email_text": "Immediate Attention: Your Instagram account will be deleted within 24 hours due to copyright infringement. Submit your official appeal immediately at: http://instagram-copyright-appeals-helpdesk.cc",
            "label": 1,
            "category": "Social Media Scam"
        },
        {
            "email_text": "Dropbox Notification: John Miller shared 'Quarterly_Salary_Bonus_Sheet.xlsx' with you. Sign in with your email provider to view this spreadsheet: http://dropbox-shared-file-cloud.top",
            "label": 1,
            "category": "Credential Harvesting"
        },
        {
            "email_text": "DocuSign: Please review and sign 'CONFIDENTIAL_TERMINATION_NOTICE.pdf'. Access this time-sensitive legal agreement via the secure portal: http://docusign-document-review-auth.info",
            "label": 1,
            "category": "Credential Harvesting"
        },
        {
            "email_text": "Amazon Shopper Survey: Congratulations! You were chosen for our $500 Amazon Gift Voucher promotion. Click here to claim your reward before the timer runs out: http://amazon-vip-rewards-club.cc",
            "label": 1,
            "category": "Shopping Scam"
        },
        {
            "email_text": "Job Offer: You have been accepted for a Remote Virtual Assistant role paying $65/hr. No interview required! Provide your social security number and bank details to receive your home office check.",
            "label": 1,
            "category": "Job Scam"
        },
        {
            "email_text": "SECURITY WARNING: Unknown foreign login detected in your Steam gaming account from Kiev, Ukraine. Confirm your identity to prevent permanent ban: http://steamcommunity-account-protect.xyz",
            "label": 1,
            "category": "Tech Scam"
        },
        {
            "email_text": "LinkedIn: 4 recruiters viewed your profile and sent urgent job proposals with starting compensation over $180,000. Unlock messages: http://linkedin-pro-job-matches.biz",
            "label": 1,
            "category": "Social Media Scam"
        },

        # ==========================================
        # LEGITIMATE / SAFE EMAILS (Label: 0)
        # ==========================================
        # 1. Workplace & Engineering
        {
            "email_text": "Hi Team, please find attached the agenda for our sprint retrospective meeting tomorrow at 10:00 AM. Let me know if you would like to add any discussion topics to the board before then.",
            "label": 0,
            "category": "Workplace Memo"
        },
        {
            "email_text": "Quarterly Business Review: Reminder that our Q3 presentation is scheduled for Thursday at 2:00 PM in Conference Room B. A Google Meet link has also been added to the calendar invite for remote teammates.",
            "label": 0,
            "category": "Workplace Memo"
        },
        {
            "email_text": "Sprint Planning Notes: Great job finishing all user stories for sprint 24! Our backlog refinement session will take place this Wednesday afternoon at 3:30 PM.",
            "label": 0,
            "category": "Workplace Memo"
        },
        {
            "email_text": "Hi Alex, could you please review the updated slide deck for the client pitch on Friday? I have incorporated your feedback on slides 4, 7, and 12.",
            "label": 0,
            "category": "Workplace Memo"
        },
        {
            "email_text": "Weekly Team Standup: Here is the summary of project deliverables and roadblocks discussed in this morning's sync. Please update your Jira tickets by end of day.",
            "label": 0,
            "category": "Workplace Memo"
        },
        {
            "email_text": "GitHub Notification: Pull request #184 (Feature: Add password hashing utility) has been approved by senior engineer. All automated CI/CD unit tests passed successfully. Ready for merge to main branch.",
            "label": 0,
            "category": "Developer Tool"
        },
        {
            "email_text": "GitLab CI: Pipeline #84920 for branch 'develop' finished successfully in 4 minutes. Code coverage is currently at 94.2%. Staging deployment complete.",
            "label": 0,
            "category": "Developer Tool"
        },
        {
            "email_text": "Hi team, we will be performing scheduled database indexing maintenance on the staging cluster tonight from 11 PM to midnight. Production environments will not be impacted.",
            "label": 0,
            "category": "IT Maintenance"
        },
        {
            "email_text": "Code review request: Hi David, I just opened PR #512 for the user authentication bug fix. Could you take a look when you have a spare 15 minutes?",
            "label": 0,
            "category": "Developer Tool"
        },
        {
            "email_text": "Docker Hub: Automated build for repository 'org/api-service:v2.4.1' finished successfully. Docker image pushed to private container registry.",
            "label": 0,
            "category": "Developer Tool"
        },

        # 2. General Company & HR Announcements
        {
            "email_text": "All Hands Meeting: Please join us this Friday at 11:00 AM EST for our monthly company-wide update. The executive leadership team will share our annual product roadmap and answer open questions.",
            "label": 0,
            "category": "HR Announcement"
        },
        {
            "email_text": "Holiday Schedule Notice: Please note that our offices will be closed on Monday in observance of Memorial Day. If you have on-call production responsibilities, please review the rotation schedule.",
            "label": 0,
            "category": "HR Announcement"
        },
        {
            "email_text": "Wellness Committee: Join us for the annual 5K Charity Fun Run next month! Sign up sheet is on the kitchen bulletin board. T-shirts and snacks provided for all participants.",
            "label": 0,
            "category": "HR Announcement"
        },
        {
            "email_text": "Employee Referral Program: Reminder that we have open engineering positions for Senior React Developer and Cloud Architect. Check internal careers portal for bonus details.",
            "label": 0,
            "category": "HR Announcement"
        },
        {
            "email_text": "Building Management Notice: Window cleaning for the 4th and 5th floors will take place this Thursday from 9 AM to 1 PM. Please ensure your blinds are raised.",
            "label": 0,
            "category": "Facilities"
        },

        # 3. Legitimate E-Commerce & Service Receipts
        {
            "email_text": "Amazon.com: Your order #112-9482014-482910 has shipped! Estimated delivery date: Thursday, October 17. Track your package in the official Amazon mobile app.",
            "label": 0,
            "category": "Order Receipt"
        },
        {
            "email_text": "Apple Store Receipt: Thank you for your purchase of iCloud+ 200 GB Storage plan ($2.99/month). Your receipt is available in your standard Apple account subscription settings.",
            "label": 0,
            "category": "Order Receipt"
        },
        {
            "email_text": "Uber Receipt: Thanks for riding with us, Sarah. Your trip total was $18.45. You can view your route map and driver tip summary in your Uber account.",
            "label": 0,
            "category": "Service Receipt"
        },
        {
            "email_text": "DoorDash: Your dinner order from Green Garden Bistro is on the way! Your dasher Carlos is expected to arrive at your door in 15 minutes.",
            "label": 0,
            "category": "Service Receipt"
        },
        {
            "email_text": "Spotify: Your monthly Family Premium subscription renewal receipt for $16.99. No action required. Thank you for listening to music with Spotify.",
            "label": 0,
            "category": "Service Receipt"
        },

        # 4. Routine Customer Support & Travel
        {
            "email_text": "Delta Air Lines: Electronic Ticket Receipt for flight DL-1492 from Atlanta (ATL) to Seattle (SEA). Confirmation code: H8K92L. Check-in opens 24 hours before departure.",
            "label": 0,
            "category": "Travel"
        },
        {
            "email_text": "Marriott Bonvoy: Your reservation at Boston Marriott Copley Place is confirmed for November 12-15. Check-in time is 4:00 PM. We look forward to welcoming you.",
            "label": 0,
            "category": "Travel"
        },
        {
            "email_text": "Customer Support Ticket #49281: Your inquiry regarding dashboard metric exports has been resolved by our support engineer. Please reply to this email if you need further help.",
            "label": 0,
            "category": "Customer Support"
        },
        {
            "email_text": "Library Notice: The book 'Hands-On Machine Learning with Scikit-Learn' is now ready for pickup at the University Library front circulation desk. Please claim within 5 days.",
            "label": 0,
            "category": "Education"
        },
        {
            "email_text": "Doctor Appointment Reminder: This is a friendly reminder of your upcoming dental cleaning appointment on Tuesday at 9:30 AM with Dr. Emily Watson. Call us if you need to reschedule.",
            "label": 0,
            "category": "Healthcare"
        },
        {
            "email_text": "Bank Statement Notification: Your monthly checking account statement for September is now available for download in your online banking documents portal. This is an informational alert.",
            "label": 0,
            "category": "Banking Notification"
        },
        {
            "email_text": "Python Weekly Newsletter: Issue #642 - Exploring Scikit-learn 1.5 features, FastAPI async best practices, and building clean data pipelines with Pandas.",
            "label": 0,
            "category": "Newsletter"
        },
        {
            "email_text": "Medium Daily Digest: Recommended stories for you in Machine Learning, Python Data Science, and Cybersecurity trends this week.",
            "label": 0,
            "category": "Newsletter"
        },
        {
            "email_text": "Coursera: Congratulations on finishing Module 2 of 'Supervised Machine Learning: Classification and Regression'. Continue your learning journey with Module 3.",
            "label": 0,
            "category": "Education"
        },
        {
            "email_text": "LinkedIn Job Alert: 3 new Machine Learning Engineer jobs matching your search criteria in Seattle, WA. View opportunities on the official LinkedIn mobile app.",
            "label": 0,
            "category": "Job Notification"
        }
    ]

    # Additional diverse samples to provide a rich dataset
    extra_phishing = [
        ("URGENT: Your Netflix membership is on hold because your automatic payment failed. Click here to update your card and avoid disruption: http://netflix-member-renew.xyz", 1, "Tech Scam"),
        ("IRS Notice: Your tax return for last year contains an error resulting in a refund balance of $620.00. Confirm your SSN and direct deposit bank: http://irs-refund-claim-2024.net", 1, "Government Impersonation"),
        ("Chase Fraud Prevention: A transfer of $950 to foreign account was flagged. Cancel transaction within 15 mins: http://chase-bank-verify-fraud.info", 1, "Banking Scam"),
        ("Apple iCloud Storage Full: All incoming photos and emails will be rejected. Upgrade 50GB storage free today: http://apple-cloud-verification.online", 1, "Tech Scam"),
        ("DocuSign Signature Request: 'Employment_Severance_Document.pdf' awaits your digital signature. Click link to access: http://docusign-confidential-sign.top", 1, "Credential Harvesting"),
        ("DHL Express Parcel Tracking: Incomplete shipping address for package #948201. Pay $2.40 processing fee: http://dhl-parcel-delivery-redirection.biz", 1, "Delivery Scam"),
        ("Internal IT Alert: Mandatory security update for all Windows and Mac laptops. Download patch here immediately: http://it-support-system-patch.cc", 1, "Credential Harvesting"),
        ("PayPal Alert: You sent a payment of $820 to CoinDesk Global. If you did not make this purchase, click here to reverse: http://paypal-dispute-resolution.xyz", 1, "Payment Scam"),
        ("HR Direct Deposit: Missing tax documentation for upcoming bonus payroll. Please submit your routing number: http://payroll-hr-employee-portal.info", 1, "HR Scam"),
        ("Walmart Rewards: You won a $500 shopping voucher! Claim your card before 11:59 PM tonight: http://walmart-survey-giftcards.top", 1, "Shopping Scam"),
        ("Cryptocurrency Escrow: 0.5 BTC is waiting for release to your wallet. Authenticate your private keys: http://crypto-escrow-blockchain.online", 1, "Crypto Scam"),
        ("FedEx Shipment Delay: Package held at customs depot. Update recipient identification: http://fedex-customs-release.net", 1, "Delivery Scam"),
        ("Facebook Security Notice: Your fan page will be disabled permanently due to policy violations. Request review: http://facebook-meta-appeal-form.biz", 1, "Social Media Scam"),
        ("Best Buy Order Confirmation: Thank you for ordering Apple MacBook Pro ($2,199.00). If you did not place this order, call refund desk or cancel: http://bestbuy-order-dispute.top", 1, "Invoice Scam"),
        ("Microsoft Account Alert: Unusual sign-in activity detected from Nigeria. Secure your account now: http://login-live-security-check.xyz", 1, "Credential Harvesting"),
        ("Zoom Notification: Manager left an urgent audio memo for you. Click link to listen: http://zoom-audio-memo-play.online", 1, "Credential Harvesting"),
        ("Wire Transfer Request from CFO: Please execute wire transfer of $12,400 to new consulting vendor before close of business today.", 1, "CEO Fraud / BEC"),
        ("Amazon Prime: Your membership will auto-renew for $139.99 today. To dispute or cancel subscription, verify credentials: http://amazon-prime-cancellation.biz", 1, "Shopping Scam"),
        ("LinkedIn Recruiter: Executive position with salary $220,000 available. Submit resume and bank details for signing bonus: http://linkedin-careers-portal.info", 1, "Job Scam"),
        ("Norton Lifelock Auto-Debit: Invoice #NL-94819 charged $429.99. Cancel transaction immediately: http://norton-refund-desk-call.cc", 1, "Invoice Scam"),
        ("Wells Fargo Security: Suspicious ATM withdrawal of $500 detected in Las Vegas. Lock your debit card: http://wellsfargo-debit-lock.xyz", 1, "Banking Scam"),
        ("USPS Tracking Alert: Package #938104 delivery failed. Re-schedule delivery at: http://usps-reschedule-address.top", 1, "Delivery Scam"),
        ("Bank of America Alert: Your online banking profile is temporarily restricted. Confirm SSN: http://boa-account-restore.online", 1, "Banking Scam"),
        ("DocuSign Contract: Please review and sign 'Vendor_Master_Service_Agreement.pdf': http://docusign-sign-document-portal.biz", 1, "Credential Harvesting"),
        ("Dropbox Shared Folder: 'Confidential_Financial_Forecast.xlsx' shared with you. Sign in to view: http://dropbox-cloud-files-view.net", 1, "Credential Harvesting"),
        ("Google Security Alert: Access granted to unverified third party application. Revoke privileges immediately: http://myaccount-google-security-review.info", 1, "Credential Harvesting"),
        ("AT&T Mobility: Your wireless bill is overdue by 15 days. Pay online today to avoid cellular service termination: http://att-billpay-resolve.biz", 1, "Invoice Scam"),
        ("Apple Support: A new device signed into your iCloud from Beijing, China. If this was not you, lock your credentials: http://appleid-protect-device.xyz", 1, "Tech Scam"),
        ("Payroll Advisory: Important notice regarding direct deposit routing change for upcoming pay period. Validate credentials: http://payroll-deposit-update.cc", 1, "HR Scam")
    ]

    extra_safe = [
        ("Hi Team, sprint planning starts at 10 AM on Monday. Please ensure your backlog tasks are estimated.", 0, "Workplace Memo"),
        ("Quarterly financial report is now available on the intranet corporate portal for all department heads.", 0, "Workplace Memo"),
        ("Reminder: Lunch and Learn session on Kubernetes container orchestration this Thursday at 12:30 PM in Cafeteria A.", 0, "Workplace Memo"),
        ("GitHub: Automated unit tests passed for pull request #324 (Fix database connection pooling timeout).", 0, "Developer Tool"),
        ("GitLab CI: Pipeline #91024 completed successfully in 3 minutes on branch feature/oauth2-support.", 0, "Developer Tool"),
        ("Amazon.com: Your package with order #104-9281928 has been delivered to your front porch. Thank you for shopping.", 0, "Order Receipt"),
        ("Apple: Your weekly screen time report is ready. Average screen time decreased by 12% compared to last week.", 0, "Device Notification"),
        ("HR Update: Open enrollment for health insurance benefits begins next Monday. Informational webinar on Wednesday.", 0, "HR Announcement"),
        ("Calendar Invite: 1-on-1 biweekly check-in with Sarah Jenkins on Friday at 2:30 PM via Google Meet.", 0, "Workplace Memo"),
        ("Uber Eats: Your order from Tokyo Sushi Bar has been accepted by the restaurant and is being prepared.", 0, "Service Receipt"),
        ("Delta Air Lines: Flight DL-391 boarding gate has been updated to Gate B22 at Salt Lake City Airport.", 0, "Travel"),
        ("University Registrar: Fall semester grades have been finalized and are accessible on the student portal.", 0, "Education"),
        ("Customer Feedback Survey: Thank you for contacting our customer service team. Rate your experience with agent Marcus.", 0, "Customer Support"),
        ("Tech Digest: This week's top software architecture articles, Microservices vs Monolith debate, and AI safety.", 0, "Newsletter"),
        ("Meeting Notes: Summary of client requirements for the Q4 mobile app redesign discussed in today's discovery call.", 0, "Workplace Memo"),
        ("Spotify Wrapped: Your top songs and favorite artists of the year are ready for you to explore in the app.", 0, "Service Receipt"),
        ("Library Reminder: The borrowed book 'Designing Data-Intensive Applications' is due in 3 days. Renew online if needed.", 0, "Education"),
        ("Office Notice: The parking garage level 2 will be closed for repaving this Saturday. Alternative parking available on level 3.", 0, "Facilities"),
        ("Jira: Ticket DEV-1049 (Optimize image loading performance) moved from In Progress to In Review by Dev Team.", 0, "Developer Tool"),
        ("Doctor's Office: Lab test results from your annual physical are now ready to view in the patient portal.", 0, "Healthcare"),
        ("DoorDash Receipt: Total charge of $24.80 for Thai Spicy Noodles. Thank you for your order.", 0, "Service Receipt"),
        ("Slack Reminder: Reminder to complete your peer reviews for the mid-year performance cycle by Friday 5 PM.", 0, "Workplace Memo"),
        ("Marriott Hotel: Your invoice for your recent stay at San Francisco Marriott Marquis is ready. Thank you for staying with us.", 0, "Travel"),
        ("GitHub: User @alex-dev requested your review on pull request #410: Refactor authentication middleware.", 0, "Developer Tool"),
        ("Target Receipt: Order #982014 ready for curbside store pickup at Northgate location. Bring photo ID.", 0, "Order Receipt"),
        ("Password Reset Requested: We received a request to reset your password for Figma. If you made this request, please follow standard in-app prompts.", 0, "Security Notice"),
        ("AWS Cost Management: Your monthly Amazon Web Services budget alert for project staging environment is at 65% of forecasted threshold.", 0, "Developer Tool"),
        ("DigitalOcean: Your droplet droplet-prod-db snapshot backup completed successfully at 03:00 UTC.", 0, "Developer Tool"),
        ("Google Calendar: Reminder - Architecture review meeting with security compliance team starts in 15 minutes.", 0, "Workplace Memo")
    ]

    for text, label, category in extra_phishing:
        data.append({"email_text": text, "label": label, "category": category})

    for text, label, category in extra_safe:
        data.append({"email_text": text, "label": label, "category": category})

    df = pd.DataFrame(data)
    
    # Save to both phishing_email_dataset.csv (requested) and dataset.csv (compatibility)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    primary_path = os.path.join(base_dir, "phishing_email_dataset.csv")
    secondary_path = os.path.join(base_dir, "dataset.csv")

    df.to_csv(primary_path, index=False)
    df.to_csv(secondary_path, index=False)

    print(f"Generated dataset successfully with {len(df)} records!")
    print(f"Phishing emails: {(df['label'] == 1).sum()}")
    print(f"Safe emails:     {(df['label'] == 0).sum()}")
    print(f"Saved to: {primary_path}")

if __name__ == "__main__":
    build_dataset()
