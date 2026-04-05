import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# -----------------------------
#Create dataset inside code (no CSV needed)
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
        "how to check leave balance",
        "show my leave details",
        "leave balance not visible",
        "how many leaves do i have",
        "leave request information",
        "leave policy details"
    ],
    "category": [
        "login",
        "login",
        "login",
        "login",
        "login",
        "login",
        "login",
        "hr",
        "hr",
        "hr",
        "hr",
        "hr",
        "hr"
    ]
}

df = pd.DataFrame(data)

# -----------------------------
#  Split data
# -----------------------------
X = df["text"]
y = df["category"]

# -----------------------------
#  Convert text to numbers
# -----------------------------
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# -----------------------------
#  Train model
# -----------------------------
model = MultinomialNB()
model.fit(X_vectorized, y)

# -----------------------------
#  Prediction function
# -----------------------------
def predict_ticket(text):
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)
    return prediction[0]

# -----------------------------
#Test examples
# -----------------------------
print("Testing model...\n")
print("Input: I forgot my password")
print("Output:", predict_ticket("I forgot my password"))

print("\nInput: show my leave balance")
print("Output:", predict_ticket("show my leave balance"))

# -----------------------------
#Interactive chatbot mode 
# -----------------------------
print("\nAI Ticket Classifier is running (type 'exit' to stop)\n")

while True:
    user_input = input("Enter your issue: ")

    if user_input.lower() == "exit":
        print("Exiting...")
        break

    category = predict_ticket(user_input)

    # Auto-response 
    if category == "login":
        response = "It seems like a login issue. Please try resetting your password or contact support."
    elif category == "hr":
        response = "This looks like an HR query. Please check your HR portal for leave details."
    else:
        response = "Sorry, I couldn't understand your issue."

    print("Category:", category)
    print("Response:", response)
    print("-" * 50)