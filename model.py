import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -----------------------------
# DATASET
# -----------------------------
data = {
    "text": [
        "forgot my password",
        "password reset not working",
        "cannot login to account",
        "incorrect password issue",
        "account login problem",
        "login failed",
        "unable to access account",
        "reset my password",
        "how to check leave balance",
        "show my leave details",
        "leave balance not visible",
        "how many leaves do i have",
        "leave request information",
        "leave policy details",
        "how to apply leave"
    ],
    "category": [
        "login","login","login","login","login","login","login","login",
        "hr","hr","hr","hr","hr","hr","hr"
    ]
}

df = pd.DataFrame(data)

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["category"], test_size=0.2, random_state=42
)

# -----------------------------
# VECTORIZATION
# -----------------------------
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -----------------------------
# MODEL TRAINING
# -----------------------------
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# -----------------------------
# ACCURACY
# -----------------------------
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.2f}")

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict_ticket(text):
    text_vec = vectorizer.transform([text])
    return model.predict(text_vec)[0]
