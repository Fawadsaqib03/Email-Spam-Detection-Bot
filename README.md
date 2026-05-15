# Email Spam Detection Bot using Naive Bayes and TF-IDF

An automated Gmail bot that classifies incoming emails as spam or not spam using a Naive Bayes and TF-IDF model and replies with the prediction result.

---

## About the Project

This project is a fully automated email bot that reads unread emails from a Gmail inbox every 30 seconds, classifies them as spam or not spam using a trained machine learning model, and sends an automated reply back to the sender with the classification result and confidence score.

```
Model       :  Multinomial Naive Bayes
Vectorizer  :  TF-IDF (max 5000 features)
Dataset     :  SMS Spam Collection Dataset
Input       :  Email Subject + Body
Output      :  SPAM 🚨 or HAM ✅ with confidence score
```

---

## Project Structure

```
Email Bot/
│
├── email_bot.py         # Main bot — reads inbox and sends replies
├── load_data.py         # Loads SMS Spam dataset from URL
├── clean.py             # Text cleaning and preprocessing
├── vectorize.py         # TF-IDF vectorization and train/test split
├── train.py             # Trains Naive Bayes model
├── evaluate.py          # Evaluates model performance
├── save.py              # Runs full pipeline and saves artifacts
│
└── artifacts/
    ├── spam_model.pkl   # Saved Naive Bayes model
    └── vectorizer.pkl   # Saved TF-IDF vectorizer
```

---

## Step by Step Breakdown

**Step 1: Load Data**
The SMS Spam Collection dataset was loaded directly from a public GitHub URL. It contains messages labeled as either ham (not spam) or spam.

**Step 2: Text Cleaning**
Each message was cleaned by converting to lowercase, removing URLs, removing numbers, removing punctuation, and stripping extra whitespace. A numeric label column was also created where ham is 0 and spam is 1.

**Step 3: TF-IDF Vectorization**
The cleaned text was converted into numerical features using TF-IDF Vectorizer with a maximum of 5000 features and English stop words removed. Data was split into 80% training and 20% testing with stratification to maintain class balance.

**Step 4: Model Training**
A Multinomial Naive Bayes model was trained on the vectorized training data. Naive Bayes works well for text classification tasks like spam detection.

**Step 5: Model Evaluation**
The model was evaluated using accuracy score, a full classification report with precision, recall, and F1-score for both Ham and Spam classes, and a confusion matrix.

**Step 6: Save Artifacts**
The trained model and vectorizer were saved as pickle files in the artifacts folder so they can be loaded by the email bot without retraining.

**Step 7: Email Bot**
The email bot connects to Gmail using IMAP with SSL and checks the inbox for unread emails every 30 seconds. For each new email it reads the sender, subject, and body, combines the subject and body into one string, cleans the text, vectorizes it, and runs the prediction. It then sends an automated reply to the sender using SMTP with the classification result and confidence score.

---

## Bot Reply Format

```
📧 Email Classification Result
===================================
Prediction : SPAM 🚨  or  HAM ✅ (Not Spam)
Confidence : 97.43%
===================================

This is an automated response from the ML Email Bot.
Powered by Naive Bayes + TF-IDF
```

---

## Technologies Used

* Python 3
* Scikit-learn
* TF-IDF Vectorizer
* Multinomial Naive Bayes
* IMAP (imaplib)
* SMTP (smtplib)
* Joblib
* Pandas
* Regex

---

## Model Performance

| Metric | Score |
|---|---|
| Accuracy | High (Naive Bayes performs well on spam tasks) |
| Evaluated On | Accuracy, Precision, Recall, F1-Score |
| Confusion Matrix | Yes |

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Email-Spam-Bot.git
   ```

2. Install required libraries:
   ```bash
   pip install scikit-learn pandas joblib
   ```

3. Train the model and save artifacts:
   ```bash
   python save.py
   ```

4. Add your Gmail address and App Password in email_bot.py:
   ```python
   BOT_EMAIL    = "your-email@gmail.com"
   APP_PASSWORD = "your-app-password"
   ```

5. Run the email bot:
   ```bash
   python email_bot.py
   ```

6. The bot will check your inbox every 30 seconds and auto-reply to any new email with the spam prediction result.

> **Note:** Make sure IMAP is enabled in your Gmail settings and that you are using a Gmail App Password, not your regular password.

---

## 📝 License

This project is intended for **academic and non-commercial use only**.

---

## 📧 Author

**Fawad Saqib**
💬 Reach out via GitHub for feedback or collaboration!
