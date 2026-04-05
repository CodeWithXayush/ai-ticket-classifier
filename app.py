import streamlit as st
from model import predict_ticket

st.set_page_config(page_title="AI Ticket Classifier")

st.title("🤖 AI Ticket Classifier")
st.write("Enter your issue and get instant solution")

user_input = st.text_input("Enter your issue:")

if user_input:
    category = predict_ticket(user_input)

    if category == "login":
        response = "🔐 Login Issue: Try resetting your password or contact support."
    elif category == "hr":
        response = "📊 HR Query: Please check your HR portal for leave details."
    else:
        response = "❓ Could not understand the issue."

    st.write("### Category:", category)
    st.write("### Response:", response)
