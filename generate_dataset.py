"""
generate_dataset.py
-------------------
Generates a comprehensive, realistic sample dataset of Phishing and Safe emails
for training the Phishing Email Detection Model.

Label 1 = Phishing Email
Label 0 = Safe / Legitimate Email
"""

import pandas as pd
import os

def create_dataset():
    emails = [
        # ==========================================
        # PHISHING EMAILS (Label: 1)
        # ==========================================
        # Account Suspensions & Urgency
        (
            "URGENT: Your bank account has been temporarily suspended due to unauthorized access attempts. "
            "Click here immediately to verify your identity and restore access: http://secure-bank-verify-login.com",
            1
        ),
        (
            "Your PayPal account has been limited! We detected unusual login attempts from an unknown device in Russia. "
            "Please confirm your account credentials within 24 hours to avoid permanent closure: http://paypal-security-update.top",
            1
        ),
        (
            "Final Notice: Your Microsoft 365 password expires today! To keep your current password, "
            "please validate your account information at: http://login-microsoft365-verify.net/auth",
            1
        ),
        (
            "SECURITY ALERT: Someone accessed your Chase Bank account from an unrecognized location. "
            "If this was not you, lock your card and verify your Social Security Number here: http://chase-secure-alert.xyz",
            1
        ),
        (
            "Action Required: Your Apple ID is locked for security reasons. "
            "You must unlock your account within 12 hours or your iCloud data will be permanently wiped: http://apple-id-recovery.info",
            1
        ),
        (
            "Wells Fargo Alert: Suspicious transaction of $1,450.00 detected on your checking account. "
            "Authorize or cancel this transaction now by logging into: http://wellsfargo-fraud-prevention.online",
            1
        ),
        (
            "Your email inbox has exceeded its storage quota (99.8% full). "
            "You will stop receiving incoming emails in 2 hours unless you upgrade your free quota here: http://webmail-storage-upgrade.biz",
            1
        ),
        (
            "Urgent: Unauthorized login detected on your Netflix account. "
            "Your subscription has been suspended due to billing error. Update your credit card details immediately: http://netflix-billing-resolve.com",
            1
        ),
        (
            "Immediate Attention: Your Instagram account will be deleted within 24 hours due to copyright infringement. "
            "Submit your official appeal immediately at: http://instagram-copyright-appeals-helpdesk.cc",
            1
        ),
        (
            "Bank of America Warning: Direct debit failure on your mortgage payment. "
            "Avoid overdraft fees by immediately confirming your routing number and account details: http://boa-account-verification-portal.com",
            1
        ),

        # Fake Invoices & Financial Fraud
        (
            "Dear Customer, please find attached overdue invoice #INV-94021 for the amount of $2,450.00. "
            "Failure to pay within 24 hours will result in immediate legal collection action. Click here to remit payment.",
            1
        ),
        (
            "Invoice #83910 from Geek Squad Best Buy: Your annual auto-renewal subscription of $499.99 has been processed. "
            "If you did not authorize this charge, call our cancellation desk immediately at +1-800-FAKE-NUM or click link to refund.",
            1
        ),
        (
            "Norton LifeLock: Receipt for your order #NL-849201. We have charged $379.00 from your checking account. "
            "To dispute this automatic payment or claim a refund, click here within 48 hours: http://norton-refund-desk.xyz",
            1
        ),
        (
            "McAfee Antivirus Protection: Your annual protection expired today. "
            "Your computer is now at high risk of 5 trojan viruses. Renew your license now with a 70% discount: http://mcafee-protect-renew.top",
            1
        ),
        (
            "Wire Transfer Confirmation: You have initiated a wire transfer of $8,200.00 to Global Holdings Ltd. "
            "If you did not initiate this wire transfer, click here immediately to stop the transaction.",
            1
        ),
        (
            "QuickBooks Online: You received a new bill from Vendor Supplies Co for $3,120.00. "
            "Review the document and approve payment via this secure link: http://quickbooks-invoice-view.info",
            1
        ),

        # CEO Fraud & Business Email Compromise (BEC)
        (
            "Are you at your desk right now? I am currently in a confidential board meeting and cannot take calls. "
            "I need you to process an urgent wire transfer for an acquisition vendor right away. Send me your cell number. Sent from iPhone.",
            1
        ),
        (
            "Quick request from CEO: Can you purchase 5 Apple gift cards ($100 each) for employee appreciation awards today? "
            "Email me the voucher codes immediately as I need them before 3 PM.",
            1
        ),
        (
            "Confidential Payroll Update from HR: We are migrating to a new direct deposit payroll system. "
            "Please click the link below to input your bank account number and routing number before Friday: http://company-payroll-portal.cc",
            1
        ),
        (
            "Urgent task from Management: Need you to review the attached confidential merger agreement. "
            "Please log in with your corporate email password to decrypt the PDF document: http://onedrive-corporate-view.xyz",
            1
        ),

        # Tax, Government & Legal Scams
        (
            "Internal Revenue Service (IRS): You have an unclaimed tax refund of $1,840.50 pending for the 2023-2024 tax year. "
            "Submit your tax refund form and direct deposit details here: http://irs-tax-refund-portal.us.com",
            1
        ),
        (
            "Social Security Administration: Your Social Security Number (SSN) has been suspended due to suspicious criminal activities. "
            "Contact our federal agent immediately or click here to prevent asset freezing: http://ssa-gov-verification-agent.net",
            1
        ),
        (
            "Department of Motor Vehicles: Final notice regarding unpaid speeding toll violations. "
            "Pay your $45.00 balance today to avoid suspension of your driver's license: http://dmv-toll-settlement.online",
            1
        ),
        (
            "Legal Notice: A formal lawsuit has been filed against you in Civil Court. "
            "Download your subpoena notice and court summons by logging in here: http://court-summons-document.top",
            1
        ),

        # Delivery & Shipping Scams
        (
            "USPS Notification: We were unable to deliver your package #US-849103 due to an incomplete home address. "
            "Please pay the $1.95 redelivery fee and update your shipping address here: http://usps-track-package-resolve.com",
            1
        ),
        (
            "FedEx Express: Your parcel is on hold at our distribution facility awaiting clearance. "
            "Confirm your personal delivery details and pay customs duty of $3.50: http://fedex-parcel-clearance.net",
            1
        ),
        (
            "DHL Express: Delivery attempt failed for shipment tracking code #DHL-938201. "
            "Reschedule your parcel delivery and download the shipping label here: http://dhl-express-track-update.top",
            1
        ),
        (
            "UPS Tracking Alert: Important document package returned to depot. "
            "Confirm your recipient identity within 24 hours: http://ups-package-confirm-address.biz",
            1
        ),

        # Lottery, Prizes, Rewards & Crypto Scams
        (
            "CONGRATULATIONS! You have been selected as the lucky winner of $1,500,000 in the International Coca-Cola Mega Sweepstakes! "
            "Send your full name, home address, and bank account number to claim your prize.",
            1
        ),
        (
            "Official Notice: You have won a $1,000 Walmart Gift Card! "
            "Complete this brief 2-minute survey to claim your card before midnight: http://walmart-rewards-winner.xyz",
            1
        ),
        (
            "Binance Security Alert: Your crypto wallet has received 1.25 BTC ($85,000.00). "
            "To release the escrow funds to your personal wallet, confirm your private key here: http://binance-escrow-claim.net",
            1
        ),
        (
            "Exclusive Crypto Opportunity: Earn guaranteed 500% daily returns with automated Bitcoin trading AI. "
            "Deposit $250 today to start trading immediately: http://crypto-ai-multiplier-yield.org",
            1
        ),
        (
            "Amazon Shopper Survey: Congratulations! You were chosen for our $500 Amazon Gift Voucher promotion. "
            "Click here to claim your reward before the timer runs out: http://amazon-vip-rewards-club.cc",
            1
        ),

        # IT & System Support Scams
        (
            "IT Helpdesk Alert: System maintenance scheduled tonight. All inactive email accounts will be purged. "
            "Click here to keep your corporate email active: http://it-helpdesk-account-sync.net",
            1
        ),
        (
            "Zoom Video Communications: You missed an important recorded video conference call from HR. "
            "Log in with your corporate credentials to listen to the recording: http://zoom-meeting-recordings-archive.biz",
            1
        ),
        (
            "DocuSign: Please review and sign 'CONFIDENTIAL_TERMINATION_NOTICE.pdf'. "
            "Access this time-sensitive legal agreement via the secure portal: http://docusign-document-review-auth.info",
            1
        ),
        (
            "Google Workspace Security: A third-party application was granted full access to your Gmail account. "
            "If you did not authorize this, revoke access immediately: http://google-security-auth-check.xyz",
            1
        ),
        (
            "Dropbox Notification: John Miller shared 'Quarterly_Salary_Bonus_Sheet.xlsx' with you. "
            "Sign in with your email provider to view this spreadsheet: http://dropbox-shared-file-cloud.top",
            1
        ),
        (
            "Job Offer: You have been accepted for a Remote Virtual Assistant role paying $65/hr. "
            "No interview required! Provide your social security number and bank details to receive your home office check.",
            1
        ),
        (
            "Meta Security Team: Your Facebook Business Page has been reported for trademark violations. "
            "Submit your verification identity documents here within 24 hours to prevent page deletion: http://meta-business-appeal-desk.com",
            1
        ),

        # ==========================================
        # SAFE / LEGITIMATE EMAILS (Label: 0)
        # ==========================================
        # Work, Team & Sprint Meetings
        (
            "Hi Team, please find attached the agenda for our sprint retrospective meeting tomorrow at 10:00 AM. "
            "Let me know if you would like to add any discussion topics to the board before then.",
            0
        ),
        (
            "Quarterly Business Review: Reminder that our Q3 presentation is scheduled for Thursday at 2:00 PM in Conference Room B. "
            "A Google Meet link has also been added to the calendar invite for remote teammates.",
            0
        ),
        (
            "Sprint Planning Notes: Great job finishing all user stories for sprint 24! "
            "Our backlog refinement session will take place this Wednesday afternoon at 3:30 PM.",
            0
        ),
        (
            "Hi Alex, could you please review the updated slide deck for the client pitch on Friday? "
            "I have incorporated your feedback on slides 4, 7, and 12.",
            0
        ),
        (
            "Weekly Team Standup: Here is the summary of project deliverables and roadblocks discussed in this morning's sync. "
            "Please update your Jira tickets by end of day.",
            0
        ),

        # Code Reviews, Development & GitHub
        (
            "GitHub Notification: Pull request #184 (Feature: Add password hashing utility) has been approved by senior engineer. "
            "All automated CI/CD unit tests passed successfully. Ready for merge to main branch.",
            0
        ),
        (
            "GitLab CI: Pipeline #84920 for branch 'develop' finished successfully in 4 minutes. "
            "Code coverage is currently at 94.2%. Staging deployment complete.",
            0
        ),
        (
            "Hi team, we will be performing scheduled database indexing maintenance on the staging cluster tonight from 11 PM to midnight. "
            "Production environments will not be impacted.",
            0
        ),
        (
            "Code review request: Hi David, I just opened PR #512 for the user authentication bug fix. "
            "Could you take a look when you have a spare 15 minutes?",
            0
        ),
        (
            "Stack Overflow Newsletter: Top answers this week on Python list comprehensions, Scikit-learn model tuning, and Docker multi-stage builds.",
            0
        ),

        # E-Commerce, Shipping & Genuine Receipts
        (
            "Amazon Order Confirmation: Thank you for your order #112-948201-49201. "
            "Your package has shipped via UPS and is estimated to arrive on Wednesday, November 18. You can track your package in your Amazon account.",
            0
        ),
        (
            "Your receipt from Apple Store: Thank you for purchasing 50GB iCloud+ monthly storage plan for $0.99. "
            "This document is an electronic receipt for your records.",
            0
        ),
        (
            "Spotify Premium: Your monthly subscription payment of $10.99 was successfully processed. "
            "Thank you for being a premium subscriber. Next billing date: December 16.",
            0
        ),
        (
            "Uber Receipt: Thanks for riding with Uber! Your trip on Sunday evening totaled $18.42. "
            "Your driver was rated 5 stars. Download your PDF tax receipt anytime from your account history.",
            0
        ),
        (
            "Starbucks Rewards: You earned 25 bonus stars on your visit today! "
            "Check your mobile app to redeem your stars for free drinks and bakery treats.",
            0
        ),
        (
            "Target Order Ready for Pickup: Great news! Your online drive-up order is ready at the Shoreline store. "
            "Let us know when you are on your way using the Target app.",
            0
        ),

        # Newsletters, Education & Learning
        (
            "Python Weekly Digest #620: Exploring the latest features in Python 3.12, machine learning with scikit-learn, and building web apps with Streamlit.",
            0
        ),
        (
            "Coursera Course Update: New assignment available in 'Machine Learning Specialization: Supervised Learning with Logistic Regression'. "
            "Due date: Sunday, Nov 22.",
            0
        ),
        (
            "Medium Daily Digest: Read 5 articles recommended for you: How TF-IDF text vectorization works, and Introduction to NLP algorithms.",
            0
        ),
        (
            "O'Reilly Radar: Trends in Artificial Intelligence, data engineering best practices, and the future of open-source software libraries.",
            0
        ),
        (
            "IEEE Spectrum: This month's featured articles on quantum computing breakthroughs and energy-efficient microprocessor design.",
            0
        ),

        # Personal, Casual & Coworker Chat
        (
            "Hey Sarah! Are we still on for lunch at the cafeteria around 12:30 PM today? "
            "Let me know if that time works or if you'd prefer 1 PM instead.",
            0
        ),
        (
            "Hi everyone, delicious homemade chocolate chip cookies are available in the 3rd floor breakroom. "
            "Help yourselves while supplies last!",
            0
        ),
        (
            "Out of Office Notice: I will be out of the office on annual leave from Nov 20 to Nov 27 with limited access to email. "
            "For urgent matters during my absence, please reach out to Karen from operations.",
            0
        ),
        (
            "Hey buddy, thanks for sharing that podcast episode yesterday! "
            "I listened to it during my morning commute and found the discussion on data privacy fascinating.",
            0
        ),
        (
            "Hi team, just a reminder that the office will be closed next Thursday and Friday in observance of the Thanksgiving holiday. Enjoy your time off!",
            0
        ),

        # Legitimate Account Notices & Appointments
        (
            "Appointment Confirmation: Reminder that your routine dental checkup with Dr. Watson is scheduled for Monday at 10:15 AM. "
            "Please call our front desk if you need to reschedule.",
            0
        ),
        (
            "Library Notice: The books you requested ('Introduction to Algorithms' and 'Pattern Recognition') are now ready for pickup at the front circulation desk.",
            0
        ),
        (
            "GitHub Security: Your personal access token 'deploy-token-ci' will expire in 7 days. "
            "If you still need this token, generate a replacement from your personal developer settings.",
            0
        ),
        (
            "Google Account: Your password was changed on Monday, Nov 16 at 5:10 PM. "
            "If you made this change, you can safely ignore this notification email.",
            0
        ),
        (
            "Customer Support: We have resolved your inquiry regarding ticket #49201. "
            "We hope this was helpful! Please feel free to reply if you need any further assistance.",
            0
        ),
        (
            "Your monthly electric utility statement from Pacific Power is now available online. "
            "Your current balance of $74.20 will be automatically debited on Nov 28.",
            0
        ),
        (
            "Delta Air Lines: Your flight reservation DL402 from Seattle to New York is confirmed. "
            "Check-in opens 24 hours prior to departure. Have a wonderful trip!",
            0
        ),
        (
            "HR Welcome Announcement: Please join us in welcoming Marcus to the Software Engineering team starting Monday! "
            "He will be sitting on the second floor next to the QA team.",
            0
        ),
        (
            "Gym Membership: Your monthly fitness membership has been renewed. "
            "Group yoga classes have resumed on Tuesday and Thursday evenings at 6 PM.",
            0
        ),
        (
            "University Registrar: Spring semester course registration opens next Monday for graduate students. "
            "Please review your degree audit before selecting your elective courses.",
            0
        ),
    ]

    # Convert to pandas DataFrame
    df = pd.DataFrame(emails, columns=["email_text", "label"])
    
    # Map numerical label to readable category
    df["category"] = df["label"].map({1: "Phishing", 0: "Safe"})
    
    # Save to CSV
    output_path = os.path.join(os.path.dirname(__file__), "dataset.csv")
    df.to_csv(output_path, index=False, encoding="utf-8")
    
    print(f"Dataset created successfully!")
    print(f"Location: {output_path}")
    print(f"Total samples: {len(df)}")
    print(f"Phishing emails: {(df['label'] == 1).sum()}")
    print(f"Safe emails:     {(df['label'] == 0).sum()}")
    return df

if __name__ == "__main__":
    create_dataset()
