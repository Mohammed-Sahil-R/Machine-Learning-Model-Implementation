import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings('ignore')

# --- 1. Create and Load Dummy Text Dataset ---
data = {
    'Label': ['ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam'],
    'Message': [
        'Hey, want to grab lunch?',
        'WINNER! Claim your FREE cash prize now!',
        'Can we reschedule the meeting?',
        'URGENT! Your account has been suspended. Click here.',
        'Please review the attached document.',
        'Congratulations! You won a brand new iPhone.',
        'See you tomorrow at 9 AM.',
        'Money back guarantee, limited time offer!!!',
        'Did you finish the report?',
        'VlP Offer! Only for today, huge discount.'
    ]
}
df = pd.DataFrame(data)

# Map text labels to numerical format (0=Ham, 1=Spam)
df['Label_Code'] = df['Label'].map({'ham': 0, 'spam': 1})

X = df['Message']
y = df['Label_Code']

print("--- Sample of the Text Dataset ---")
print(df.head())
print("\n")

# --- 2. Data Splitting ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Total samples: {len(X)}")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print("\n")

# --- 3. Feature Engineering (Text to Numbers) ---
# Convert text data into a matrix of token counts
vectorizer = CountVectorizer()

# Fit the vectorizer on the training data and transform it
X_train_transformed = vectorizer.fit_transform(X_train)

# Transform the test data using the fitted vectorizer
X_test_transformed = vectorizer.transform(X_test)

print(f"Feature Vector Size: {X_train_transformed.shape[1]}")
print("\n")

# --- 4. Model Training (Multinomial Naive Bayes) ---
model = MultinomialNB()

model.fit(X_train_transformed, y_train)

print(f"Model trained successfully: {type(model).__name__}")
print("\n")

# --- 5. Make Predictions ---
y_pred = model.predict(X_test_transformed)

# --- 6. Model Evaluation ---
print("--- Model Evaluation Metrics ---")

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {accuracy:.4f}")

# Display detailed Classification Report (0=Ham, 1=Spam)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Ham (0)', 'Spam (1)']))
print("\n")

# --- 7. Demonstration: Prediction on New Data ---
new_emails = ["meeting details", "claim your prize"]
new_emails_transformed = vectorizer.transform(new_emails)

predictions = model.predict(new_emails_transformed)

print("--- New Email Predictions ---")
for email, pred in zip(new_emails, predictions):
    label = 'Spam' if pred == 1 else 'Ham'
    print(f"Email: '{email}' -> Predicted Label: {label}")

