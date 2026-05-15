import imaplib
import smtplib
import email
import time
import joblib
import re
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

BOT_EMAIL    = "CONNECT YOUR BOT EMAIL HERE"   
APP_PASSWORD = "Genrate an app password from your email settings and put it here"        

# Load model and vectorizer
model      = joblib.load("artifacts/spam_model.pkl")
vectorizer = joblib.load("artifacts/vectorizer.pkl")

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict(text: str):
    cleaned   = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    pred      = model.predict(vectorized)[0]
    proba     = model.predict_proba(vectorized)[0]
    conf      = float(proba.max())
    label     = "SPAM 🚨" if pred == 1 else "HAM ✅ (Not Spam)"
    return label, conf

def send_reply(to_email, subject, prediction, confidence):
    msg = MIMEMultipart()
    msg['From']    = BOT_EMAIL
    msg['To']      = to_email
    msg['Subject'] = f"Re: {subject}"

    body = (
        f"📧 Email Classification Result\n"
        f"{'='*35}\n"
        f"Prediction : {prediction}\n"
        f"Confidence : {confidence:.2%}\n"
        f"{'='*35}\n\n"
        f"This is an automated response from the ML Email Bot.\n"
        f"Powered by Naive Bayes + TF-IDF"
    )
    msg.attach(MIMEText(body, 'plain'))

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(BOT_EMAIL, APP_PASSWORD)
        server.send_message(msg)
    print(f"  ✅ Reply sent to {to_email}")

def check_inbox():
    mail = imaplib.IMAP4_SSL('imap.gmail.com')
    mail.login(BOT_EMAIL, APP_PASSWORD)
    mail.select('inbox')

    _, messages = mail.search(None, 'UNSEEN')
    email_ids = messages[0].split()

    if not email_ids:
        print("  No new emails.")
        return

    for eid in email_ids:
        _, msg_data = mail.fetch(eid, '(RFC822)')
        raw = msg_data[0][1]
        msg = email.message_from_bytes(raw)

        sender  = msg['From']
        subject = msg['Subject'] or "(no subject)"

        # Get email body
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == 'text/plain':
                    body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                    break
        else:
            body = msg.get_payload(decode=True).decode('utf-8', errors='ignore')

        print(f"\n  📨 New email from: {sender}")
        print(f"  Subject: {subject}")
        print(f"  Body preview: {body[:80]}...")

        # Predict
        prediction, confidence = predict(subject + " " + body)
        print(f"  🤖 Prediction: {prediction} ({confidence:.2%})")

        # Reply
        send_reply(sender, subject, prediction, confidence)

    mail.logout()


# MAIN Loop checks every 30 seconds

print("🤖 Email Bot started! Checking inbox every 30 seconds...")
print(f"   Bot email: {BOT_EMAIL}")
print("   Press Ctrl+C to stop\n")

while True:
    try:
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] Checking inbox...")
        check_inbox()
    except Exception as e:
        print(f"  ❌ Error: {e}")
    time.sleep(30)